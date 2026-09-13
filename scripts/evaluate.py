"""Evaluate the fine-tuned model on the validation and the held-out test split.

Both are reported on purpose. The validation number was used for model
selection (best.pt is the best validation epoch), so it is optimistic; the
test split was never looked at during training and is the honest one. The
confusion matrix is exported as numbers, not only as a picture, because
"which class gets mistaken for which" is the question the census reasoning
layer has to defend against.

Usage:
    python scripts/evaluate.py --weights runs/rtdetr_aquarium/weights/best.pt \
        --data /kaggle/working/data/aquarium/data.yaml \
        --out  artifacts/metrics.json
"""
import argparse
import json
from pathlib import Path

from prepare_data import CLASSES


def confusion_as_dict(results):
    """Ultralytics' matrix is (nc+1) x (nc+1): rows are predictions, columns
    are ground truth, and the extra index is `background` (a miss or a false
    positive). Rewritten with class names so it reads without the docs."""
    try:
        matrix = results.confusion_matrix.matrix
    except Exception:
        return None
    labels = list(CLASSES) + ["background"]
    n = min(len(labels), matrix.shape[0])
    table = {}
    for i in range(n):
        row = {}
        for j in range(n):
            v = int(matrix[i, j])
            if v:
                row[labels[j]] = v
        table[labels[i]] = row
    # The two numbers a reader wants first.
    confusions = []
    for i in range(n - 1):
        for j in range(n - 1):
            if i != j and int(matrix[i, j]) > 0:
                confusions.append({"true": labels[j], "predicted": labels[i], "count": int(matrix[i, j])})
    confusions.sort(key=lambda c: -c["count"])
    return {"rows_are_predicted_cols_are_true": table, "off_diagonal_confusions": confusions}


def per_class_metrics(box):
    """Per-class P / R / mAP keyed by class name.

    Ultralytics stores per-class arrays only for classes that have ground
    truth in the split, ordered by `ap_class_index`, so `class_result()` takes
    a position in that list, not a class id. Indexing it by class id silently
    returns another class's numbers whenever one class is absent.
    """
    raw_index = getattr(box, "ap_class_index", None)   # numpy array; never test it with `or`
    ap_index = [int(c) for c in (raw_index if raw_index is not None else [])]
    per_class = {}
    for i, name in enumerate(CLASSES):
        if i not in ap_index:
            per_class[name] = {"note": "no ground-truth instances of this class in the split"}
            continue
        p, r, ap50, ap = box.class_result(ap_index.index(i))
        per_class[name] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "mAP50": round(float(ap50), 4),
            "mAP50_95": round(float(ap), 4),
        }
    return per_class


def run_split(model, data_yaml, split, imgsz, batch, tag, project):
    results = model.val(data=data_yaml, split=split, imgsz=imgsz, batch=batch,
                        plots=True, project=project, name=f"val_{tag}", exist_ok=True)
    box = results.box
    per_class = per_class_metrics(box)
    return {
        "split": split,
        "data": str(data_yaml),
        "mAP50": round(float(box.map50), 4),
        "mAP50_95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "per_class": per_class,
        "confusion": confusion_as_dict(results),
        "speed_ms_per_image": {k: round(float(v), 2) for k, v in (results.speed or {}).items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="artifacts/metrics.json")
    ap.add_argument("--project", default="runs/eval", help="where Ultralytics writes the val plots")
    args = ap.parse_args()

    from ultralytics import RTDETR

    model = RTDETR(args.weights)
    report = {
        "weights": args.weights,
        "classes": CLASSES,
        "validation": run_split(model, args.data, "val", args.imgsz, args.batch, "val", args.project),
        "test": run_split(model, args.data, "test", args.imgsz, args.batch, "test", args.project),
    }
    report["val_minus_test_mAP50"] = round(report["validation"]["mAP50"] - report["test"]["mAP50"], 4)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "validation"}, indent=2))


if __name__ == "__main__":
    main()
