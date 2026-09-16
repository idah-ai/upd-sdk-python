"""
COCO8 → UPD

ultralytics is used only for the one-time dataset download.
Images are loaded directly with PIL; labels are read from YOLO .txt files.

Prerequisites
-------------
    pip install upd ultralytics

Usage
-----
    python examples/coco8_to_upd.py --output coco8.upd
"""

from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="coco8.upd")
    return p.parse_args()


# COCO 80-class names
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def read_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Parse a YOLO .txt label file → list of (class_id, cx, cy, w, h)."""
    if not label_path.exists():
        return []
    rows = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            cls, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
            rows.append((cls, cx, cy, w, h))
    return rows


def image_to_jpeg(img) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=95)
    return buf.getvalue()


def run(output: str) -> None:
    from PIL import Image
    from ultralytics.data.utils import check_det_dataset
    from upd import UPD

    # Downloads COCO8 on first run (~6 MB), cached afterwards
    data         = check_det_dataset("coco8.yaml")
    dataset_root = Path(data["path"])

    with UPD.open(output) as upd:
        upd.metadata.set("Schema-Built-By", "coco8_to_upd v1.0")

        for split in ("train", "val"):
            img_dir   = dataset_root / "images" / split
            label_dir = dataset_root / "labels" / split
            img_paths = sorted(img_dir.glob("*.*"))

            print(f"[{split}] {len(img_paths)} images")

            ds = upd.datasets.create(
                name=f"COCO8 {split.capitalize()}",
                modality="coco8-image",
                metadata={
                    "Source": "ultralytics COCO8",
                    "Split": split,
                    "Num-Classes": 80,
                },
            )

            ann_batch = []
            for img_path in img_paths:
                label_path = label_dir / (img_path.stem + ".txt")
                jpeg = image_to_jpeg(Image.open(img_path))

                media = upd.medias.create(
                    blob_data=jpeg,
                    media_type="image/jpeg",
                    metadata={
                        "Original-Filename": img_path.name,
                        "Sha256": hashlib.sha256(jpeg).hexdigest(),
                    },
                )
                entry = upd.entries.create(dataset_id=ds.id, media_url=media.local_url)

                for cls_id, cx, cy, w, h in read_labels(label_path):
                    ann_batch.append({
                        "entry_id":   entry.id,
                        "shape_type": "coco8-image:bounding-box",
                        "shape_args": {"cx": cx, "cy": cy, "w": w, "h": h},
                        "category":   COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else str(cls_id),
                        "properties": {
                            "class_id": cls_id,
                        },
                        "metadata": {"QC-Status": "Passed"},
                    })

            upd.annotations.bulk_create(ann_batch)

        for ds in upd.datasets.all():
            n_entries = upd.entries.filter(upd.entries.dataset_id == ds.id).count().execute()
            n_ann     = upd.annotations.count_for_dataset(ds.id)
            print(f"  {ds.name}: {n_entries} entries, {n_ann} annotations")

    print(f"\nWritten: {output}")


if __name__ == "__main__":
    args = parse_args()
    run(args.output)
