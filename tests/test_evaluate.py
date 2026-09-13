"""The per-class metric extraction must map Ultralytics' position-indexed
arrays back to class ids. No model needed: a stub stands in for results.box."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate import per_class_metrics  # noqa: E402


class StubBox:
    """Only shark (4) and starfish (5) had ground truth: two rows, in that order."""
    ap_class_index = np.array([4, 5])

    def class_result(self, pos):
        return [(0.9, 0.8, 0.85, 0.5), (0.3, 0.2, 0.25, 0.1)][pos]


def test_per_class_metrics_follow_ap_class_index_not_class_id():
    out = per_class_metrics(StubBox())
    assert out["shark"]["mAP50"] == 0.85
    assert out["starfish"]["mAP50"] == 0.25
    assert "note" in out["fish"] and "note" in out["stingray"]


def test_empty_index_gives_notes_for_every_class():
    class Empty:
        ap_class_index = np.array([], dtype=int)
    out = per_class_metrics(Empty())
    assert all("note" in v for v in out.values())
