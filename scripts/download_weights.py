"""Fetch the fine-tuned checkpoint.

Primary source is the GitHub release asset (a plain HTTPS download, no
account needed). A public Kaggle dataset can be used instead with --kaggle.

Usage:
    python scripts/download_weights.py                    # -> artifacts/best.pt
    python scripts/download_weights.py --out /tmp/w.pt
    python scripts/download_weights.py --kaggle owner/slug
"""
import argparse
import os
import shutil
import urllib.request
from pathlib import Path

DEFAULT_URL = os.getenv(
    "WEIGHTS_URL",
    "https://github.com/KanagavelAK/aquarium-rtdetr/releases/download/v1.0/best.pt")
DEFAULT_KAGGLE = os.getenv("WEIGHTS_KAGGLE_DATASET", "")


def download_url(url: str, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    tmp.replace(out)
    return out


def download_kaggle(dataset: str, out: Path) -> Path:
    import kagglehub

    root = Path(kagglehub.dataset_download(dataset))
    src = next(root.rglob("best.pt"), None)
    if src is None:
        raise SystemExit(f"no best.pt inside {dataset} (looked under {root})")
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out)
    return out


def download(out: Path, url: str = DEFAULT_URL, kaggle: str = DEFAULT_KAGGLE) -> Path:
    """Try the direct URL first, then the Kaggle dataset if one is configured."""
    errors = []
    if url:
        try:
            return download_url(url, out)
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    if kaggle:
        try:
            return download_kaggle(kaggle, out)
        except Exception as exc:
            errors.append(f"kaggle {kaggle}: {exc}")
    raise RuntimeError("could not download weights: " + "; ".join(errors))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL, help="direct download URL of best.pt")
    ap.add_argument("--kaggle", default=DEFAULT_KAGGLE, help="owner/slug of a Kaggle dataset holding best.pt")
    ap.add_argument("--out", default="artifacts/best.pt")
    args = ap.parse_args()
    path = download(Path(args.out), args.url, args.kaggle)
    print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
