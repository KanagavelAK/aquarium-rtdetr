"""Convert the Kaggle mirror of Roboflow's "Aquarium Combined" dataset into a
clean YOLO dataset with a deterministic, image-level train/val/test split.

Source: https://www.kaggle.com/datasets/slavkoprytula/aquarium-data-cots
        (Roboflow "Aquarium Combined", CC BY 4.0: 638 photographs taken at the
        Henry Doorly Zoo, Omaha and the National Aquarium, Baltimore)

The mirror ships YOLO txt labels in Roboflow's train/valid/test folders. This
script does not trust that layout: it walks the whole tree for images that
have a label file, removes byte-identical duplicates (Roboflow exports can
place the same frame in two splits under different names), maps class ids to
this repo's fixed class order by NAME, and re-splits everything with a hash
of the file stem so the split never moves between reruns.

Usage:
    python scripts/prepare_data.py \
        --root /kaggle/input/aquarium-data-cots \
        --out  /kaggle/working/data/aquarium
"""
import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path

# Fixed class order. The index is baked into the weights, so never reorder it.
CLASSES = ["fish", "jellyfish", "penguin", "puffin", "shark", "starfish", "stingray"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}

# How the source may spell each class. Everything else is dropped, never guessed.
ALIASES = {
    "fish": "fish",
    "jellyfish": "jellyfish", "jelly fish": "jellyfish", "jelly-fish": "jellyfish",
    "penguin": "penguin",
    "puffin": "puffin",
    "shark": "shark",
    "starfish": "starfish", "star fish": "starfish", "star-fish": "starfish", "sea star": "starfish",
    "stingray": "stingray", "sting ray": "stingray", "sting-ray": "stingray", "ray": "stingray",
}

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def read_source_names(root: Path):
    """Class names in the source's own id order, from any yaml under root.

    Returns a list (index -> name) or None if no yaml declares `names`.
    """
    for yaml_path in sorted(list(root.rglob("*.yaml")) + list(root.rglob("*.yml"))):
        text = yaml_path.read_text(encoding="utf-8", errors="replace")
        if "names" not in text:
            continue
        try:
            import yaml
            doc = yaml.safe_load(text) or {}
        except Exception:
            continue
        names = doc.get("names")
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names, key=int)]
        if isinstance(names, list):
            return [str(n) for n in names]
    return None


def find_pairs(root: Path):
    """Yield (image_path, label_path_or_None, source_split) for every image.

    Handles the two YOLO layouts in the wild -- `<split>/images/*.jpg` with
    `<split>/labels/*.txt`, and `images/<split>/*.jpg` with
    `labels/<split>/*.txt` -- by mirroring each image's path below `images/`
    into the sibling `labels/`. If no images/ + labels/ pair exists at all,
    falls back to any image with a same-stem .txt beside it.
    """
    seen = set()
    for images_dir in sorted(p for p in root.rglob("images") if p.is_dir()):
        labels_dir = images_dir.parent / "labels"
        if not labels_dir.is_dir():
            continue
        for image_path in sorted(images_dir.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_EXTS or image_path in seen:
                continue
            seen.add(image_path)
            rel = image_path.relative_to(images_dir)
            label_path = labels_dir / rel.with_suffix(".txt")
            source_split = (rel.parts[0] if len(rel.parts) > 1 else images_dir.parent.name).lower()
            yield image_path, (label_path if label_path.exists() else None), source_split
    if seen:
        return
    for image_path in sorted(root.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = image_path.with_suffix(".txt")
        if label_path.exists():
            yield image_path, label_path, image_path.parent.name.lower()


def split_for(stem: str, val_frac: float, test_frac: float) -> str:
    """Hash-based split.

    A pure function of the file stem, so re-running this script or adding
    images later never moves an existing image between splits. That is what
    stops train/test leakage from creeping in across reruns.
    """
    digest = hashlib.md5(stem.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < test_frac:
        return "test"
    if bucket < test_frac + val_frac:
        return "val"
    return "train"


def clean_stem(stem: str) -> str:
    """Roboflow appends `.rf.<hash>` to every file; hashing on the part before
    it keeps the split stable across Roboflow re-exports of the same image."""
    return re.split(r"\.rf\.[0-9a-f]+$", stem)[0]


def convert_label(label_path, source_names, dropped: Counter):
    """Source YOLO lines -> lines in this repo's class order. Coordinates are
    clipped to [0, 1]; degenerate boxes and unknown classes are dropped."""
    lines = []
    if label_path is None:
        return lines
    for raw in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = raw.split()
        if len(parts) < 5:
            continue
        try:
            src_id = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        if source_names is not None and 0 <= src_id < len(source_names):
            raw_name = source_names[src_id]
        else:
            raw_name = CLASSES[src_id] if 0 <= src_id < len(CLASSES) else str(src_id)
        name = ALIASES.get(raw_name.strip().lower())
        if name is None:
            dropped[raw_name] += 1
            continue
        x1, y1 = max(0.0, cx - bw / 2), max(0.0, cy - bh / 2)
        x2, y2 = min(1.0, cx + bw / 2), min(1.0, cy + bh / 2)
        if x2 - x1 <= 0.001 or y2 - y1 <= 0.001:
            dropped["degenerate box"] += 1
            continue
        lines.append("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(
            CLASS_TO_ID[name], (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="downloaded dataset root (any depth)")
    ap.add_argument("--out", required=True, help="output directory for the YOLO dataset")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    if not root.is_dir():
        raise SystemExit(f"{root} is not a directory")

    source_names = read_source_names(root)
    print("source class order:", source_names or "(no yaml found; assuming this repo's order)")

    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    per_split = Counter()
    per_split_class = {s: Counter() for s in ("train", "val", "test")}
    source_split_counts = Counter()
    dropped = Counter()
    content_hashes = {}
    duplicates = 0
    background = 0
    total = 0

    pairs = list(find_pairs(root))
    if not pairs:
        raise SystemExit(f"no images/ + labels/ folder pairs found under {root}")

    for image_path, label_path, source_split in pairs:
        data = image_path.read_bytes()
        digest = hashlib.md5(data).hexdigest()
        if digest in content_hashes:
            duplicates += 1  # same bytes already kept under another name
            continue
        content_hashes[digest] = image_path.name

        stem = clean_stem(image_path.stem)
        split = split_for(stem, args.val_frac, args.test_frac)
        lines = convert_label(label_path, source_names, dropped)
        if not lines:
            background += 1  # kept: an empty tank teaches the model restraint

        dst_name = stem + image_path.suffix.lower()
        dst_img = out / "images" / split / dst_name
        if not dst_img.exists():
            if args.copy:
                shutil.copy2(image_path, dst_img)
            else:
                try:
                    dst_img.symlink_to(image_path.resolve())
                except OSError:
                    shutil.copy2(image_path, dst_img)  # Windows without developer mode
        (out / "labels" / split / (stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")

        total += 1
        per_split[split] += 1
        source_split_counts[f"{source_split}->{split}"] += 1
        for line in lines:
            per_split_class[split][CLASSES[int(line.split()[0])]] += 1

    (out / "data.yaml").write_text(
        "\n".join([
            f"path: {out.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            *[f"  {i}: {n}" for i, n in enumerate(CLASSES)],
            "",
        ]),
        encoding="utf-8",
    )

    stats = {
        "source": "kaggle:slavkoprytula/aquarium-data-cots (Roboflow Aquarium Combined, CC BY 4.0)",
        "classes": CLASSES,
        "images_total": total,
        "images_per_split": dict(per_split),
        "instances_per_split": {s: dict(c) for s, c in per_split_class.items()},
        "instances_total": dict(sum(per_split_class.values(), Counter())),
        "background_only_images": background,
        "byte_identical_duplicates_removed": duplicates,
        "labels_dropped": dict(dropped),
        "source_split_to_our_split": dict(source_split_counts),
        "split_method": "md5(file stem without Roboflow's .rf.<hash> suffix) -> deterministic bucket, image level",
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
    }
    (out / "split_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"\nwrote {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
