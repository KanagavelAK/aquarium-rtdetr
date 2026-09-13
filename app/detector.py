"""RT-DETR inference wrapper.

Loaded once at process start. Everything downstream consumes the plain
dictionaries this module returns, never the Ultralytics result objects, so the
reasoning layer stays testable without a GPU or a checkpoint.
"""
from __future__ import annotations

import io
import logging
import os
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

log = logging.getLogger("aquarium.detector")

CLASSES = ["fish", "jellyfish", "penguin", "puffin", "shark", "starfish", "stingray"]
DEFAULT_WEIGHTS = os.getenv("MODEL_WEIGHTS", "artifacts/best.pt")
DEFAULT_CONF = float(os.getenv("CONF_THRESHOLD", "0.25"))
DEFAULT_IMGSZ = int(os.getenv("IMGSZ", "640"))
DEVICE = os.getenv("DETECTOR_DEVICE", "")   # "" lets Ultralytics choose; "cpu" pins it
WEIGHTS_URL = os.getenv("WEIGHTS_URL", "https://github.com/KanagavelAK/aquarium-rtdetr/releases/download/v1.0/best.pt")
WEIGHTS_KAGGLE_DATASET = os.getenv("WEIGHTS_KAGGLE_DATASET", "")


@dataclass
class Detection:
    label: str
    confidence: float
    box_xyxy: List[float]

    def to_dict(self):
        return asdict(self)


class DetectorError(RuntimeError):
    pass


class Detector:
    """Thin, thread-safe wrapper around a fine-tuned RT-DETR checkpoint."""

    def __init__(self, weights: str = DEFAULT_WEIGHTS, imgsz: int = DEFAULT_IMGSZ):
        self.weights = weights
        self.imgsz = imgsz
        self._lock = threading.Lock()
        self._model = None
        self._names = None

    def load(self):
        if self._model is not None:
            return
        path = Path(self.weights)
        if not path.exists() and (WEIGHTS_URL or WEIGHTS_KAGGLE_DATASET):
            log.info("weights missing at %s, downloading", path)
            try:
                from scripts.download_weights import download
                download(path, WEIGHTS_URL, WEIGHTS_KAGGLE_DATASET)
            except Exception as exc:
                log.warning("weights download failed: %s", exc)
        if not path.exists():
            raise DetectorError(
                "model weights not found at {}. Set MODEL_WEIGHTS to the checkpoint, "
                "or run scripts/download_weights.py.".format(path)
            )
        from ultralytics import RTDETR  # imported lazily, it is slow

        started = time.time()
        self._model = RTDETR(str(path))
        self._names = getattr(self._model, "names", None) or {
            i: n for i, n in enumerate(CLASSES)
        }
        log.info("loaded %s in %.1fs, classes=%s", path, time.time() - started, self._names)

    @property
    def ready(self) -> bool:
        return self._model is not None

    def label_for(self, index: int) -> str:
        if isinstance(self._names, dict):
            return str(self._names.get(index, index))
        try:
            return str(self._names[index])
        except Exception:
            return str(index)

    def predict(self, image_bytes: bytes, conf: float = DEFAULT_CONF):
        """Run detection on raw image bytes.

        Returns (detections, image_width, image_height, inference_ms).
        """
        self.load()
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise DetectorError("could not decode the uploaded file as an image") from exc

        array = np.array(image)[:, :, ::-1]  # PIL RGB -> BGR for ultralytics
        started = time.time()
        with self._lock:  # ultralytics predict is not reentrant
            result = self._model.predict(array, conf=conf, imgsz=self.imgsz, verbose=False,
                                         device=DEVICE or None)[0]
        elapsed_ms = (time.time() - started) * 1000.0

        detections = []
        for box in result.boxes:
            detections.append(Detection(
                label=self.label_for(int(box.cls.item())),
                confidence=round(float(box.conf.item()), 4),
                box_xyxy=[round(float(v), 1) for v in box.xyxy[0].tolist()],
            ))
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections, image.width, image.height, round(elapsed_ms, 1)


detector = Detector()
