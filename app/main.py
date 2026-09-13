"""FastAPI application exposing the detector and the reasoning layer.

    GET  /health   readiness, loaded weights, whether the language model is on
    POST /detect   image -> boxes, labels, confidences
    POST /ask      image + question -> plain-language answer, or an honest refusal

Run:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import os
import time
import uuid

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from app import reasoning
from app.detector import CLASSES, DEFAULT_CONF, DetectorError, detector
from app.scene import build_scene
from app.schemas import (AskResponse, DetectResponse, ErrorResponse, HealthResponse)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("aquarium.api")

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(12 * 1024 * 1024)))

app = FastAPI(
    title="Aquarium Animal Census API",
    version="1.0.0",
    description=(
        "RT-DETR fine-tuned on aquarium imagery for seven animal classes (fish, "
        "jellyfish, penguin, puffin, shark, starfish, stingray), with a hand-written "
        "reasoning layer that answers counting, ranking and presence questions from "
        "the detections and refuses when crowding, label conflicts or weak "
        "detections make the answer unreliable."
    ),
)


@app.middleware("http")
async def add_request_context(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    started = time.time()
    response = await call_next(request)
    elapsed = (time.time() - started) * 1000
    response.headers["x-request-id"] = request_id
    log.info("%s %s -> %s in %.0fms [%s]", request.method, request.url.path,
             response.status_code, elapsed, request_id)
    return response


def _routing(decision) -> dict:
    """Route dataclass -> the RoutingOut shape. The field is named question_kind
    in the response because `kind` alone is ambiguous to an API consumer."""
    return {
        "needs_detection": decision.needs_detection,
        "question_kind": decision.kind,
        "targets": decision.targets,
        "rationale": decision.rationale,
        "decided_by": decision.decided_by,
    }


def _error(request_id: str, status: int, error: str, detail: str):
    return JSONResponse(
        status_code=status,
        content=ErrorResponse(request_id=request_id, error=error, detail=detail).model_dump(),
    )


async def _read_image(upload: UploadFile) -> bytes:
    data = await upload.read()
    if not data:
        raise DetectorError("the uploaded file was empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise DetectorError(
            "image is larger than the {} MB limit".format(MAX_UPLOAD_BYTES // (1024 * 1024)))
    return data


@app.on_event("startup")
def warm_model():
    """Load weights at boot so the first real request is not the slow one."""
    try:
        detector.load()
    except DetectorError as exc:
        log.warning("model not loaded at startup: %s", exc)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if detector.ready else "degraded",
        model_loaded=detector.ready,
        weights=detector.weights,
        classes=CLASSES,
        llm_enabled=bool(os.getenv("ANTHROPIC_API_KEY")),
    )


@app.post("/detect", response_model=DetectResponse, responses={400: {"model": ErrorResponse}})
async def detect(request: Request,
                 file: UploadFile = File(..., description="JPEG or PNG image"),
                 confidence: float = Form(DEFAULT_CONF)):
    request_id = request.state.request_id
    try:
        image_bytes = await _read_image(file)
        detections, width, height, ms = detector.predict(image_bytes, conf=confidence)
    except DetectorError as exc:
        return _error(request_id, 400, "bad_request", str(exc))
    except Exception as exc:  # unexpected: log the trace, return a clean message
        log.exception("detect failed [%s]", request_id)
        return _error(request_id, 500, "inference_error", str(exc))

    counts = {}
    for d in detections:
        counts[d.label] = counts.get(d.label, 0) + 1

    return DetectResponse(
        request_id=request_id,
        image={"width": width, "height": height},
        confidence_threshold=confidence,
        count=len(detections),
        counts_by_class=counts,
        detections=[d.to_dict() for d in detections],
        inference_ms=ms,
    )


@app.post("/ask", response_model=AskResponse, responses={400: {"model": ErrorResponse}})
async def ask(request: Request,
              question: str = Form(..., description="natural-language question about the image"),
              file: UploadFile = File(None, description="optional; required for image questions"),
              confidence: float = Form(DEFAULT_CONF)):
    """Route, detect if needed, reason over the structured output, guard the answer."""
    request_id = request.state.request_id

    decision = reasoning.route(question)

    # Step 1: the question does not need the detector, so it is not called.
    if not decision.needs_detection:
        return AskResponse(
            request_id=request_id,
            question=question,
            answer=reasoning.insufficient_message(decision.kind, []),
            sufficient_information=False,
            routing=_routing(decision),
            detector_called=False,
            guardrail_reasons=[decision.rationale],
            answer_source="template",
        )

    if file is None:
        return _error(request_id, 400, "bad_request",
                      "this question needs an image, but no file was uploaded")

    # Step 2: call the detection model.
    try:
        image_bytes = await _read_image(file)
        detections, width, height, ms = detector.predict(image_bytes, conf=confidence)
    except DetectorError as exc:
        return _error(request_id, 400, "bad_request", str(exc))
    except Exception as exc:
        log.exception("ask failed during detection [%s]", request_id)
        return _error(request_id, 500, "inference_error", str(exc))

    facts = build_scene(detections, width, height, confidence)

    # Step 3: deterministic guardrail, before any language model call.
    guard = reasoning.guardrail(decision, facts)
    if not guard.sufficient:
        return AskResponse(
            request_id=request_id,
            question=question,
            answer=reasoning.insufficient_message(decision.kind, guard.reasons),
            sufficient_information=False,
            routing=_routing(decision),
            detector_called=True,
            guardrail_reasons=guard.reasons,
            answer_source="template",
            evidence=facts.to_dict(),
            inference_ms=ms,
        )

    # Step 4: reason over the structured facts and answer.
    answer, source = reasoning.compose(question, decision, facts)
    return AskResponse(
        request_id=request_id,
        question=question,
        answer=answer,
        sufficient_information=True,
        routing=_routing(decision),
        detector_called=True,
        guardrail_reasons=[],
        answer_source=source,
        evidence=facts.to_dict(),
        inference_ms=ms,
    )
