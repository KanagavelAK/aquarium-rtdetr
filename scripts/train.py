"""Fine-tune RT-DETR on the aquarium dataset.

Sized for the Kaggle free tier. RT-DETR is attention heavy, so batch 8 per
GPU at 640 px is the largest that reliably fits on a 16 GB card. With two
GPUs pass --device 0,1 and --batch 16; Ultralytics trains with DDP across
both. The dataset is small (~450 training images), so the budget is spent on
epochs rather than on data: 80 epochs is about 40 minutes on 2 x T4.

Usage:
    python scripts/train.py --data /kaggle/working/data/aquarium/data.yaml \
        --epochs 80 --batch 16 --device 0,1 --cache ram --project /kaggle/working/runs

Dry run first to measure seconds per epoch before committing the full budget:
    python scripts/train.py --data ... --epochs 2 --name dryrun
"""
import argparse
import json
import platform
import subprocess
import time
import traceback
from pathlib import Path


def hardware_report():
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
    except Exception as exc:  # pragma: no cover
        info["torch_error"] = str(exc)
    try:
        info["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"], text=True).strip()
    except Exception:
        pass
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="rtdetr-l.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--lr0", type=float, default=1e-4)
    ap.add_argument("--optimizer", default="AdamW")
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="0",
                    help="cuda device id(s), e.g. 0 or 0,1 for multi-GPU DDP, or cpu")
    ap.add_argument("--cache", default="",
                    help="'ram' or 'disk' to cache decoded images; empty for none")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="rtdetr_aquarium")
    args = ap.parse_args()

    from ultralytics import RTDETR, settings

    # Ultralytics auto-registers a Ray Tune callback whenever `ray` is importable.
    # Kaggle ships a newer ray whose private API that callback calls no longer
    # exists, which crashes training at the end of the first epoch.
    settings.update({"raytune": False})

    hw = hardware_report()
    print(json.dumps(hw, indent=2))

    model = RTDETR(args.model)
    run_dir = Path(args.project) / args.name
    started = time.time()
    error = None
    try:
        model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            lr0=args.lr0,
            optimizer=args.optimizer,
            patience=args.patience,
            workers=args.workers,
            seed=args.seed,
            device=args.device,
            project=args.project,
            name=args.name,
            exist_ok=True,
            amp=True,          # required to fit batch 8 on 16 GB
            cache=args.cache or False,
            plots=True,
            val=True,
            # Underwater frames are dim and blue-shifted; hue jitter would make
            # colour a less useful cue, so it stays at the Ultralytics default
            # (0.015). Flips are safe: no animal here has a canonical orientation.
            fliplr=0.5,
        )
    except BaseException as exc:
        # Under DDP the real training runs in subprocesses that already saved
        # best.pt; a crash in the parent must not lose the receipt.
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    elapsed = time.time() - started

    epochs_completed = None
    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        epochs_completed = max(0, len(results_csv.read_text(encoding="utf-8").strip().splitlines()) - 1)

    receipt = {
        "status": "failed" if error else "ok",
        "error": error,
        "epochs_completed": epochs_completed,
        "hardware": hw,
        "model": args.model,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "lr0": args.lr0,
        "optimizer": args.optimizer,
        "patience": args.patience,
        "seed": args.seed,
        "device": args.device,
        "workers": args.workers,
        "cache": args.cache or False,
        "amp": True,
        "data": args.data,
        "wall_clock_seconds": round(elapsed, 1),
        "wall_clock_human": f"{elapsed / 3600:.2f} h",
        "weights": str(run_dir / "weights" / "best.pt"),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    print("\nPaste the receipt above into the memo. Reproducibility is graded.")
    if error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
