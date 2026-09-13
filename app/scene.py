"""Turn a flat list of boxes into the census facts the reasoning layer needs.

The detector answers "where are the animals". A census question asks "how
many", "which is the most common", "are there more X than Y" -- and every one
of those is only as good as the boxes it is summed over. Three things make a
raw box list untrustworthy for counting, and this module measures each:

  1. Duplicate queries. RT-DETR has no NMS. When two decoder queries lock
     onto the same animal the API receives two boxes, usually one strong and
     one weak, almost on top of each other. Counting both is a plain error, so
     same-class boxes with IoU >= DUPLICATE_IOU are collapsed to the stronger
     one and the number collapsed is reported.

  2. Crowding. A school of fish or a huddle of penguins produces boxes that
     overlap their neighbours. Where a box overlaps another box of the same
     class by IoU >= CROWD_IOU the detector may have merged two animals into
     one or split one into two; the count for that class carries a crowding
     score (fraction of its boxes in that state) and the guardrail refuses
     counts when it is high.

  3. Identity conflicts. A shark and a stingray box on the same pixels means
     the detector is not sure what the animal is. Where boxes of different
     classes overlap by IoU >= CONFLICT_IOU the pair is recorded and neither
     class can be counted or ranked with confidence.

Nothing here consults a language model. These are arithmetic facts about the
boxes, computed the same way every time.
"""
from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List

DUPLICATE_IOU = 0.70   # same class, this much overlap: one animal, two queries
CROWD_IOU = 0.30       # same class, this much overlap: animals touching or overlapping
CONFLICT_IOU = 0.50    # different classes, this much overlap: one animal, two labels

# Detections weaker than this cannot support an assertion. Shared with the
# guardrail in app/reasoning.py, which imports it from here.
ASSERTION_FLOOR = float(os.getenv("LOW_CONFIDENCE_FLOOR", "0.45"))

PLURALS = {"fish": "fish", "jellyfish": "jellyfish", "starfish": "starfish"}


def plural(label: str, n: int) -> str:
    if n == 1:
        return label
    return PLURALS.get(label, label + "s")


def iou(a, b) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class ClassFacts:
    label: str
    count: int = 0
    confident: int = 0        # boxes at or above the assertion floor
    tentative: int = 0        # boxes below it
    max_confidence: float = 0.0
    mean_confidence: float = 0.0
    crowded: int = 0          # boxes overlapping another box of the same class
    crowding: float = 0.0     # crowded / count
    conflicts: int = 0        # boxes overlapping a box of a different class
    conflicts_with: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "count": self.count,
            "confident": self.confident,
            "tentative": self.tentative,
            "max_confidence": self.max_confidence,
            "mean_confidence": self.mean_confidence,
            "crowded": self.crowded,
            "crowding": self.crowding,
            "conflicts": self.conflicts,
            "conflicts_with": self.conflicts_with,
        }


@dataclass
class SceneFacts:
    image_width: int
    image_height: int
    confidence_floor: float
    assertion_floor: float = ASSERTION_FLOOR
    counts: Dict[str, int] = field(default_factory=dict)
    classes: Dict[str, ClassFacts] = field(default_factory=dict)
    total: int = 0
    kinds: int = 0
    ranking: List[tuple] = field(default_factory=list)   # (label, count), most common first
    duplicates_suppressed: int = 0
    conflicts: List[dict] = field(default_factory=list)
    crowding: float = 0.0
    max_confidence: float = 0.0
    mean_confidence: float = 0.0
    animals: List[dict] = field(default_factory=list)    # boxes kept after duplicate suppression
    detections: List[dict] = field(default_factory=list) # raw detector output

    def facts_for(self, label: str) -> ClassFacts:
        return self.classes.get(label) or ClassFacts(label=label)

    def to_dict(self):
        return {
            "image_size": {"width": self.image_width, "height": self.image_height},
            "confidence_floor": self.confidence_floor,
            "assertion_floor": self.assertion_floor,
            "animals_detected": self.total,
            "kinds_detected": self.kinds,
            "counts": self.counts,
            "ranking": [{"label": k, "count": v} for k, v in self.ranking],
            "duplicates_suppressed": self.duplicates_suppressed,
            "identity_conflicts": self.conflicts,
            "crowding": self.crowding,
            "max_confidence": self.max_confidence,
            "mean_confidence": self.mean_confidence,
            "classes": {k: v.to_dict() for k, v in self.classes.items()},
            "animals": self.animals,
            "detections": self.detections,
        }


def build_scene(detections, image_width: int, image_height: int,
                confidence_floor: float) -> SceneFacts:
    """Group raw detections into census facts."""
    dets = [d.to_dict() if hasattr(d, "to_dict") else dict(d) for d in detections]
    dets_sorted = sorted(dets, key=lambda d: -d["confidence"])

    # 1. Duplicate suppression, strongest box first.
    kept = []
    duplicates = 0
    for d in dets_sorted:
        twin = any(k["label"] == d["label"] and iou(k["box_xyxy"], d["box_xyxy"]) >= DUPLICATE_IOU
                   for k in kept)
        if twin:
            duplicates += 1
            continue
        kept.append({"label": d["label"], "confidence": d["confidence"],
                     "box_xyxy": list(d["box_xyxy"]), "crowded": False, "conflict_with": []})

    # 2. Crowding and 3. identity conflicts, over the kept boxes.
    conflicts = []
    for i, a in enumerate(kept):
        for j in range(i + 1, len(kept)):
            b = kept[j]
            overlap = iou(a["box_xyxy"], b["box_xyxy"])
            if a["label"] == b["label"]:
                if overlap >= CROWD_IOU:
                    a["crowded"] = b["crowded"] = True
            elif overlap >= CONFLICT_IOU:
                a["conflict_with"].append(b["label"])
                b["conflict_with"].append(a["label"])
                conflicts.append({"labels": [a["label"], b["label"]],
                                  "confidences": [a["confidence"], b["confidence"]],
                                  "iou": round(overlap, 3)})

    classes: Dict[str, ClassFacts] = {}
    for k in kept:
        cf = classes.setdefault(k["label"], ClassFacts(label=k["label"]))
        cf.count += 1
        if k["confidence"] >= ASSERTION_FLOOR:
            cf.confident += 1
        else:
            cf.tentative += 1
        cf.max_confidence = max(cf.max_confidence, k["confidence"])
        cf.mean_confidence += k["confidence"]
        if k["crowded"]:
            cf.crowded += 1
        if k["conflict_with"]:
            cf.conflicts += 1
            for other in k["conflict_with"]:
                if other not in cf.conflicts_with:
                    cf.conflicts_with.append(other)
    for cf in classes.values():
        cf.mean_confidence = round(cf.mean_confidence / cf.count, 4)
        cf.crowding = round(cf.crowded / cf.count, 3)
        cf.max_confidence = round(cf.max_confidence, 4)

    counts = {k: v.count for k, v in classes.items()}
    ranking = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    confidences = [k["confidence"] for k in kept]

    return SceneFacts(
        image_width=image_width,
        image_height=image_height,
        confidence_floor=confidence_floor,
        counts=counts,
        classes=classes,
        total=len(kept),
        kinds=len(classes),
        ranking=ranking,
        duplicates_suppressed=duplicates,
        conflicts=conflicts,
        crowding=round(sum(1 for k in kept if k["crowded"]) / len(kept), 3) if kept else 0.0,
        max_confidence=round(max(confidences), 4) if confidences else 0.0,
        mean_confidence=round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
        animals=kept,
        detections=dets,
    )
