"""Response models. These are the contract the reviewer reads, so they are
explicit about provenance: every answer says how it was routed, whether the
detector ran, and why."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DetectionOut(BaseModel):
    label: str = Field(..., examples=["shark"])
    confidence: float = Field(..., examples=[0.91])
    box_xyxy: List[float] = Field(..., description="x1, y1, x2, y2 in pixels",
                                  examples=[[143.0, 58.2, 391.5, 201.7]])


class DetectResponse(BaseModel):
    request_id: str
    image: Dict[str, int]
    confidence_threshold: float
    count: int
    counts_by_class: Dict[str, int]
    detections: List[DetectionOut]
    inference_ms: float


class RoutingOut(BaseModel):
    needs_detection: bool
    question_kind: str
    targets: List[str] = Field(default_factory=list,
                               description="classes the question names, in order")
    rationale: str
    decided_by: str


class AskResponse(BaseModel):
    request_id: str
    question: str
    answer: str
    sufficient_information: bool
    routing: RoutingOut
    detector_called: bool
    guardrail_reasons: List[str] = []
    answer_source: str = Field(..., description="llm or template")
    evidence: Optional[Dict[str, Any]] = None
    inference_ms: Optional[float] = None


class HealthResponse(BaseModel):
    # "model_loaded" collides with pydantic's protected model_ namespace; the
    # field name is the clearer one for an API consumer, so keep it and opt out.
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
    weights: str
    classes: List[str]
    llm_enabled: bool


class ErrorResponse(BaseModel):
    request_id: str
    error: str
    detail: str
