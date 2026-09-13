"""Generate a self-contained Kaggle notebook for this repo.

Every script the notebook needs is embedded as a %%writefile cell, read from
the real source files at generation time so the notebook can never drift from
the code. One Save & Run All does everything: dependencies, dataset download,
split, training on both T4s, evaluation, failure mining, a reasoning-layer
check and packaging of the artifacts.

Usage:
    python notebooks/build_kaggle_notebook.py
    -> notebooks/kaggle_aquarium_rtdetr.ipynb   (upload this to Kaggle)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "kaggle_aquarium_rtdetr.ipynb"

EMBEDDED_FILES = [
    "scripts/__init__.py",
    "scripts/prepare_data.py",
    "scripts/train.py",
    "scripts/evaluate.py",
    "scripts/failure_cases.py",
    "app/__init__.py",
    "app/detector.py",
    "app/scene.py",
    "app/reasoning.py",
]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip("\n")}


def writefile_cell(rel_path):
    body = (ROOT / rel_path).read_text(encoding="utf-8")
    target = f"/kaggle/working/repo/{rel_path}"
    if not body.strip():
        # %%writefile rejects an empty cell body, so create empty files directly
        return code(f'from pathlib import Path\nPath("{target}").parent.mkdir(parents=True, exist_ok=True)\n'
                    f'Path("{target}").touch()\nprint("created empty", "{target}")')
    return code(f"%%writefile {target}\n{body}")


cells = []

cells.append(md("""
# Aquarium Animal Census — RT-DETR fine-tune on Kaggle

Fine-tunes RT-DETR-L to detect seven aquarium animals — **fish, jellyfish,
penguin, puffin, shark, starfish, stingray** (none of them a COCO class) — on
Roboflow's *Aquarium Combined* photographs, evaluates it on a held-out test
split, mines failure cases with measured evidence, checks the census reasoning
layer end to end, and packages the checkpoint for the FastAPI service in the repo.

**Notebook settings (right-hand panel):**

| Setting | Value |
|---|---|
| Accelerator | **GPU T4 x2** (recommended). P100 also works: Kaggle's current torch build has dropped sm_60 support, so cell 1 detects that and installs a compatible torch first. |
| Internet | **ON** — needed for pip, `kagglehub`, and the `rtdetr-l.pt` base weights |
| Datasets to attach | **None.** The dataset is downloaded by code below via `kagglehub`. |

Run through **Save Version → Save & Run All** so a browser disconnect does
not kill the run. Everything is written under `/kaggle/working`. Expect about
45 minutes end to end on 2 × T4.
"""))

cells.append(md("""
## 1. Environment

Installs the pinned dependencies, then checks that the preinstalled torch was
compiled for this GPU. Kaggle's torch 2.10 + CUDA 12.8 image no longer includes
sm_60 kernels, so on a P100 the model fails with `CUDA error: no kernel image
is available`. If that mismatch is detected, a torch build that still supports
the card is installed **before** torch is imported into this kernel.
"""))
cells.append(code("""
import os, sys, subprocess, json
from pathlib import Path
os.makedirs("/kaggle/working/repo", exist_ok=True)
os.chdir("/kaggle/working/repo")

def pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)

pip("ultralytics==8.3.40", "kagglehub")   # no numpy/opencv pins: Kaggle ships numpy 2 and cv2 already
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

# Probe in a subprocess so torch is not yet imported here if it has to be replaced.
probe = subprocess.run([sys.executable, "-c",
    "import torch; cap = torch.cuda.get_device_capability(); "
    "print('sm_%d%d' % cap); print(' '.join(torch.cuda.get_arch_list())); print(torch.__version__)"],
    capture_output=True, text=True)
gpu_sm, arch_list, torch_ver = (probe.stdout.strip().split("\\n") + ["", "", ""])[:3]
print(f"gpu {gpu_sm} | torch {torch_ver} compiled for: {arch_list}")

if gpu_sm and gpu_sm not in arch_list.split():
    print(f"{gpu_sm} is not supported by the preinstalled torch; installing torch 2.6.0 (cu126), ~2.5 GB ...")
    pip("--force-reinstall", "--no-deps",
        "torch==2.6.0", "torchvision==0.21.0",
        "--index-url", "https://download.pytorch.org/whl/cu126")
    pip("ultralytics==8.3.40")   # re-satisfy deps the --no-deps install skipped

import torch, ultralytics
sm = "sm_%d%d" % torch.cuda.get_device_capability()
assert sm in torch.cuda.get_arch_list(), f"{sm} still unsupported by torch {torch.__version__}: {torch.cuda.get_arch_list()}"
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| gpu", torch.cuda.get_device_name(0),
      "| ultralytics", ultralytics.__version__)
"""))

cells.append(md("""
## 2. Run configuration

Edit here, nowhere else. Set `DRY_RUN = True` the first time to measure
seconds per epoch, then multiply it out before committing to `EPOCHS`.
RT-DETR is attention heavy: **batch 8 per GPU at 640 px is the largest that
fits on a 16 GB card**. With two GPUs the batch is doubled and Ultralytics
trains with DDP across both; the per-card load stays at 8. The dataset is
small (about 450 training images, ~28 steps per epoch), so the budget goes
into epochs: 80 epochs with early stopping at 25 epochs of no improvement.
"""))
cells.append(code("""
DRY_RUN   = False   # True -> 2 epochs into runs/dryrun, then stop
EPOCHS    = 80
PATIENCE  = 25
N_GPU     = torch.cuda.device_count()
for i in range(N_GPU):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}  {torch.cuda.get_device_properties(i).total_memory/2**30:.1f} GB")
if N_GPU < 2:
    print(f"WARNING: only {N_GPU} GPU visible. Training will still run, ~2x slower. Pick 'GPU T4 x2' next time.")
DEVICE    = ",".join(str(i) for i in range(N_GPU)) or "cpu"   # "0,1" -> Ultralytics relaunches under DDP across both cards
BATCH     = 8 * max(N_GPU, 1)                                 # 8 per GPU is the 16 GB ceiling for RT-DETR-L at 640
WORKERS   = os.cpu_count()                           # Kaggle gives 4 cores; the dataloader is the bottleneck otherwise
CACHE     = "ram"                                    # 638 decoded images at 640 px is well under 1 GB
IMGSZ     = 640
LR0       = 1e-4
SEED      = 0

# Public Kaggle mirrors of Roboflow's "Aquarium Combined" (CC BY 4.0). The first
# that downloads is used; scripts/prepare_data.py reads any YOLO layout.
DATASET_CANDIDATES = ["slavkoprytula/aquarium-data-cots",
                      "sharansmenon/aquarium-dataset",
                      "sovitrath/aquarium-data"]

WORK  = "/kaggle/working"
DATA  = f"{WORK}/data/aquarium"
RUNS  = f"{WORK}/runs"
ARTS  = f"{WORK}/artifacts"
RUN_NAME = "dryrun" if DRY_RUN else "rtdetr_aquarium"
RUN_EPOCHS = 2 if DRY_RUN else EPOCHS
BEST = f"{RUNS}/{RUN_NAME}/weights/best.pt"
os.makedirs(ARTS, exist_ok=True)
print(json.dumps({k: v for k, v in globals().items() if k.isupper() and not k.startswith("_")}, indent=2, default=str))
"""))

cells.append(md("""
## 3. Repository code

The repo is embedded below so the notebook is self-contained. These cells are
generated from the real source files by `notebooks/build_kaggle_notebook.py`;
do not edit them here.
"""))
for rel in EMBEDDED_FILES:
    cells.append(writefile_cell(rel))

cells.append(md("""
## 4. Dataset

Downloaded by code, no manual attachment. Roboflow's **Aquarium Combined**
(CC BY 4.0): 638 photographs from the Henry Doorly Zoo (Omaha) and the
National Aquarium (Baltimore), 4,821 boxes over seven classes, shipped in
YOLO format. The Kaggle mirror is tried first; two other mirrors are
fallbacks. The tree is printed so the memo can quote exactly what was used.
"""))
cells.append(code("""
import kagglehub
from pathlib import Path

DATASET_ROOT, DATASET_SLUG = None, None
for slug in DATASET_CANDIDATES:
    try:
        DATASET_ROOT = Path(kagglehub.dataset_download(slug))
        DATASET_SLUG = slug
        break
    except Exception as exc:
        print(f"{slug}: {type(exc).__name__}: {str(exc)[:120]}")
assert DATASET_ROOT is not None, "no dataset mirror could be downloaded; check Internet is ON"
print("using", DATASET_SLUG, "->", DATASET_ROOT)

def tree(path, depth=0, max_depth=3):
    if depth > max_depth: return
    entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
    for p in entries[:12]:
        if p.is_dir():
            n_img = sum(1 for f in p.rglob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
            print("  " * depth + f"{p.name}/  ({n_img} images below)")
            tree(p, depth + 1, max_depth)
        else:
            print("  " * depth + f"{p.name}  ({p.stat().st_size/1e3:.0f} KB)")
    if len(entries) > 12:
        print("  " * depth + f"... {len(entries) - 12} more")
tree(DATASET_ROOT)
for y in list(DATASET_ROOT.rglob("*.yaml"))[:2]:
    print(f"\\n--- {y.relative_to(DATASET_ROOT)} ---\\n{y.read_text()[:600]}")
"""))

cells.append(md("""
## 5. Prepare the training data

Walks the download for images with labels, removes byte-identical
duplicates, maps class ids to this repo's fixed order **by name**, and
re-splits with a deterministic `md5(file stem)` bucket (70 / 15 / 15 at the
image level) so reruns never leak training images into the test split. The
class histogram shows the imbalance the memo has to talk about: fish are more
than half of all boxes, starfish under three percent.
"""))
cells.append(code("""
!python scripts/prepare_data.py --root "{DATASET_ROOT}" --out "{DATA}" --val-frac 0.15 --test-frac 0.15
"""))
cells.append(code("""
import matplotlib.pyplot as plt
stats = json.load(open(f"{DATA}/split_stats.json"))
classes = stats["classes"]
fig, ax = plt.subplots(figsize=(9, 3.2))
bottom = [0] * len(classes)
for split, colour in (("train", "#2a7f9e"), ("val", "#f0a04b"), ("test", "#7a5195")):
    vals = [stats["instances_per_split"][split].get(c, 0) for c in classes]
    ax.bar(classes, vals, bottom=bottom, label=split, color=colour)
    bottom = [b + v for b, v in zip(bottom, vals)]
for i, total in enumerate(bottom):
    ax.text(i, total + 15, str(total), ha="center", fontsize=9)
ax.set_ylabel("boxes"); ax.set_title("instances per class and split"); ax.legend(frameon=False)
plt.tight_layout(); plt.savefig(f"{ARTS}/class_distribution.png", dpi=130); plt.show()
print("images per split:", stats["images_per_split"], "| duplicates removed:", stats["byte_identical_duplicates_removed"])
"""))

cells.append(md("""
## 6. Train

`rtdetr-l.pt` (COCO-pretrained) is fetched automatically by Ultralytics on
first use. AMP is on (required to fit batch 8), decoded images are cached in
RAM. A `training_receipt.json` with hardware, hyperparameters and wall-clock
time is written next to the weights.
"""))
cells.append(code("""
import shutil
if Path(RUNS, RUN_NAME).exists():
    shutil.rmtree(Path(RUNS, RUN_NAME))   # a leftover run must never be mistaken for this one
    print("removed stale run directory", Path(RUNS, RUN_NAME))

!python scripts/train.py \\
    --data "{DATA}/data.yaml" \\
    --model rtdetr-l.pt \\
    --epochs {RUN_EPOCHS} --batch {BATCH} --imgsz {IMGSZ} --lr0 {LR0} --optimizer AdamW \\
    --patience {PATIENCE} --workers {WORKERS} --seed {SEED} --device {DEVICE} --cache {CACHE} \\
    --project "{RUNS}" --name {RUN_NAME}
"""))
cells.append(code("""
assert Path(BEST).exists(), f"training did not produce {BEST}"
receipt_path = Path(RUNS) / RUN_NAME / "training_receipt.json"
if not receipt_path.exists():
    # train.py did not reach its receipt step (e.g. the DDP parent crashed after
    # the workers had already saved best.pt). Rebuild it from Ultralytics' own files.
    import csv, yaml
    run_dir = Path(RUNS) / RUN_NAME
    rows = list(csv.DictReader(open(run_dir / "results.csv"))) if (run_dir / "results.csv").exists() else []
    train_args = yaml.safe_load(open(run_dir / "args.yaml")) if (run_dir / "args.yaml").exists() else {}
    last = {k.strip(): v.strip() for k, v in rows[-1].items()} if rows else {}
    receipt = {
        "status": "reconstructed",
        "note": "train.py exited before writing the receipt; values below come from results.csv and args.yaml",
        "hardware": {"gpu": [torch.cuda.get_device_name(i) for i in range(N_GPU)], "torch": torch.__version__},
        "model": train_args.get("model"), "epochs": train_args.get("epochs"), "epochs_completed": len(rows),
        "batch": train_args.get("batch"), "imgsz": train_args.get("imgsz"), "lr0": train_args.get("lr0"),
        "optimizer": train_args.get("optimizer"), "seed": train_args.get("seed"), "device": str(train_args.get("device")),
        "wall_clock_seconds": float(last.get("time", 0)) or None,
        "final_val_mAP50": last.get("metrics/mAP50(B)"), "final_val_mAP50_95": last.get("metrics/mAP50-95(B)"),
        "weights": BEST,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print("WARNING: receipt was reconstructed; check the end of the training cell output for the error.\\n")
print(receipt_path.read_text())
if DRY_RUN:
    print("\\nDRY_RUN is on. Read seconds/epoch above, set DRY_RUN = False and EPOCHS, then Save & Run All.")
"""))
cells.append(code("""
# Training curves: box / cls / giou losses and validation mAP per epoch.
import pandas as pd
res = pd.read_csv(f"{RUNS}/{RUN_NAME}/results.csv"); res.columns = [c.strip() for c in res.columns]
fig, axes = plt.subplots(1, 3, figsize=(14, 3.4))
for ax, cols, title in ((axes[0], ["train/giou_loss", "val/giou_loss"], "box (GIoU) loss"),
                        (axes[1], ["train/cls_loss", "val/cls_loss"], "classification loss"),
                        (axes[2], ["metrics/mAP50(B)", "metrics/mAP50-95(B)"], "validation mAP")):
    for c in cols:
        if c in res: ax.plot(res["epoch"], res[c], label=c.split("/")[-1])
    ax.set_title(title); ax.set_xlabel("epoch"); ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(f"{ARTS}/training_curves.png", dpi=130); plt.show()
best_epoch = int(res["metrics/mAP50(B)"].idxmax())
print(f"best validation mAP50 {res['metrics/mAP50(B)'].max():.3f} at epoch {res.loc[best_epoch, 'epoch']}, "
      f"{len(res)} epochs run")
"""))

cells.append(md("""
## 7. Evaluate

mAP50, mAP50-95, precision and recall per class on the **validation** split
(used for model selection, so optimistic) and on the **held-out test** split
(never looked at during training). The confusion matrix is exported as
numbers as well as a picture: which class the model mistakes for which is
what the census guardrail has to defend against.
"""))
cells.append(code("""
!python scripts/evaluate.py \\
    --weights "{BEST}" \\
    --data "{DATA}/data.yaml" \\
    --imgsz {IMGSZ} --batch {BATCH} \\
    --out  "{ARTS}/metrics.json" --project "{RUNS}/eval"
"""))
cells.append(code("""
m = json.load(open(f"{ARTS}/metrics.json"))
rows = []
for c in m["classes"]:
    v, t = m["validation"]["per_class"].get(c, {}), m["test"]["per_class"].get(c, {})
    rows.append([c, v.get("mAP50"), t.get("mAP50"), t.get("mAP50_95"), t.get("precision"), t.get("recall")])
rows.append(["**all**", m["validation"]["mAP50"], m["test"]["mAP50"], m["test"]["mAP50_95"], m["test"]["precision"], m["test"]["recall"]])
print("| class | val mAP50 | test mAP50 | test mAP50-95 | test P | test R |\\n|---|---|---|---|---|---|")
for r in rows:
    print("| " + " | ".join("—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x)) for x in r) + " |")
print("\\ntop confusions on the test split:", m["test"]["confusion"]["off_diagonal_confusions"][:6] if m["test"]["confusion"] else "n/a")

from IPython.display import Image as IPImage, display
for name in ("confusion_matrix_normalized.png", "PR_curve.png"):
    p = Path(RUNS) / "eval" / "val_test" / name
    if p.exists(): display(IPImage(filename=str(p), width=560))
"""))

cells.append(md("""
## 8. Failure mining

Worst test images by error count, each miss measured for blur, contrast,
scale, crowding and exposure, and each false positive checked for being a
duplicate query on an already-found animal. Confirm every hypothesis by
looking at the annotated image before it goes in the memo.
"""))
cells.append(code("""
!python scripts/failure_cases.py \\
    --weights "{BEST}" \\
    --data "{DATA}/data.yaml" \\
    --top 8 \\
    --out "{ARTS}/failures"
"""))
cells.append(code("""
import cv2
report = json.load(open(f"{ARTS}/failures/failure_report.json"))
worst = report["worst"][:5]
fig, axes = plt.subplots(max(len(worst), 1), 1, figsize=(10, 7 * max(len(worst), 1)))
for ax, rec in zip(axes if len(worst) > 1 else [axes], worst):
    img = cv2.cvtColor(cv2.imread(f"{ARTS}/failures/failure_{rec['image']}"), cv2.COLOR_BGR2RGB)
    ax.imshow(img); ax.axis("off")
    hyp = "; ".join(h for e in rec["errors"] for h in e["hypotheses"])
    ax.set_title(f"{rec['image']} — {rec['error_count']} errors (gt {rec['ground_truth_count']}, pred {rec['prediction_count']})\\n{hyp[:170]}", fontsize=9)
plt.tight_layout(); plt.show()
print("class confusions across the whole test split:", report["class_confusions_across_test_split"])
"""))

cells.append(md("""
## 9. Prediction gallery and README samples

Four test images with predictions drawn, saved as one figure for the README,
plus two individual test images copied into `samples/` so the API examples
work on a clean checkout.
"""))
cells.append(code("""
from ultralytics import RTDETR
model = RTDETR(BEST)
test_images = sorted(Path(f"{DATA}/images/test").glob("*"))
# pick the four test images with the most ground-truth boxes: busier scenes show more
def n_boxes(p):
    lbl = Path(f"{DATA}/labels/test/{p.stem}.txt")
    return len(lbl.read_text().splitlines()) if lbl.exists() else 0
gallery = sorted(test_images, key=lambda p: -n_boxes(p))[:4]
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
for ax, p in zip(axes.ravel(), gallery):
    r = model.predict(str(p), conf=0.25, verbose=False)[0]
    ax.imshow(cv2.cvtColor(r.plot(line_width=2, font_size=10), cv2.COLOR_BGR2RGB)); ax.axis("off")
    counts = {}
    for c in r.boxes.cls.tolist(): counts[model.names[int(c)]] = counts.get(model.names[int(c)], 0) + 1
    ax.set_title(", ".join(f"{v} {k}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])) or "no detections", fontsize=10)
plt.tight_layout(); plt.savefig(f"{ARTS}/prediction_gallery.png", dpi=110); plt.show()

os.makedirs(f"{ARTS}/samples", exist_ok=True)
for i, img in enumerate(gallery[:2], 1):
    shutil.copy(img, f"{ARTS}/samples/tank_{i:02d}{img.suffix.lower()}")
print("samples:", os.listdir(f"{ARTS}/samples"))
"""))

cells.append(md("""
## 10. End-to-end check of the reasoning layer

Runs the same code path as the API's `POST /ask` — route → detect → census
facts → guardrail → compose — on two test images, offline. No API key is
needed; without one the answer comes from templates over the same structured
facts. Watch for at least one honest refusal: a crowded fish school or a
shark/stingray conflict should produce "I do not have enough information".
"""))
cells.append(code("""
os.environ["MODEL_WEIGHTS"] = BEST
sys.path.insert(0, "/kaggle/working/repo")
import importlib
for m in ("app.detector", "app.scene", "app.reasoning"):
    if m in sys.modules: importlib.reload(sys.modules[m])
from app.detector import Detector
from app.scene import build_scene
from app import reasoning

det = Detector(weights=BEST, imgsz=IMGSZ)
QUESTIONS = ["How many fish are in this tank?", "What is the most common animal here?",
             "Are there any sharks?", "Are there more fish than jellyfish?",
             "How many different kinds of animal are there?", "What species of fish is that?",
             "What is the capital of France?"]
demo_log = []
for sample in gallery[:2]:
    image_bytes = sample.read_bytes()
    print("\\n=====", sample.name)
    for question in QUESTIONS:
        decision = reasoning.route(question)
        line = {"image": sample.name, "question": question, "kind": decision.kind, "detector_called": decision.needs_detection}
        if not decision.needs_detection:
            line["answer"] = reasoning.insufficient_message(decision.kind, [])
        else:
            dets, w, h, ms = det.predict(image_bytes, conf=0.25)
            facts = build_scene(dets, w, h, 0.25)
            guard = reasoning.guardrail(decision, facts)
            if not guard.sufficient:
                line["answer"] = reasoning.insufficient_message(decision.kind, guard.reasons)
                line["refused"] = True
            else:
                line["answer"], line["source"] = reasoning.compose(question, decision, facts)
            line["evidence"] = {"counts": facts.counts, "crowding": facts.crowding,
                                "duplicates_suppressed": facts.duplicates_suppressed,
                                "conflicts": len(facts.conflicts)}
        demo_log.append(line)
        print(f"\\nQ: {question}\\n   route: {decision.kind} | detector: {decision.needs_detection}")
        print("   A:", line["answer"])
        if "evidence" in line: print("   evidence:", line["evidence"])
json.dump(demo_log, open(f"{ARTS}/reasoning_demo.json", "w"), indent=2)
"""))

cells.append(md("""
## 11. Package the artifacts

Everything the API and the memo need lands in `/kaggle/working/artifacts`,
plus a single zip. Download from the notebook's **Output** tab, then:

- `best.pt` -> `artifacts/best.pt` in the repo (and a GitHub release asset)
- `samples/` -> `samples/` in the repo (the README curl examples use them)
- `metrics.json`, `training_receipt.json`, `split_stats.json`, `failures/`,
  `reasoning_demo.json` -> fill the README results and the memo
- `*.png` -> `artifacts/` and `docs/` for the README figures

then start the API with `uvicorn app.main:app --host 0.0.0.0 --port 8000`.
"""))
cells.append(code("""
shutil.copy(BEST, f"{ARTS}/best.pt")
shutil.copy(f"{RUNS}/{RUN_NAME}/weights/last.pt", f"{ARTS}/last.pt")
shutil.copy(f"{RUNS}/{RUN_NAME}/training_receipt.json", ARTS)
shutil.copy(f"{DATA}/split_stats.json", ARTS)
for plot in ("results.png", "confusion_matrix.png", "confusion_matrix_normalized.png", "PR_curve.png", "labels.jpg"):
    src = Path(RUNS) / RUN_NAME / plot
    if src.exists():
        shutil.copy(src, ARTS)
for plot in ("confusion_matrix_normalized.png", "PR_curve.png"):
    src = Path(RUNS) / "eval" / "val_test" / plot
    if src.exists():
        shutil.copy(src, f"{ARTS}/test_{plot}")

shutil.make_archive(f"{WORK}/aquarium_artifacts", "zip", ARTS)
for p in sorted(Path(ARTS).rglob("*")):
    if p.is_file():
        print(f"{p.stat().st_size/1e6:8.1f} MB  {p.relative_to(ARTS)}")
print(f"\\nzip: {WORK}/aquarium_artifacts.zip  {Path(f'{WORK}/aquarium_artifacts.zip').stat().st_size/1e6:.1f} MB")
"""))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "kaggle": {
            "accelerator": "gpu",
            "dataSources": [],
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT} ({len(cells)} cells, {len(EMBEDDED_FILES)} embedded files)")
