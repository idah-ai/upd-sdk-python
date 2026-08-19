"""
infer_coco8_yolo.py
===================
Run a pre-trained YOLOv8 model on every image stored in a COCO8 UPD file,
then write the predicted bounding boxes back into the same UPD file as a
new dataset of annotations.

What this script demonstrates
------------------------------
- Reading raw image BLOBs out of a UPD file
- Running inference with a pre-trained Ultralytics YOLOv8 model
- Writing model predictions back into UPD as annotations
  (shape_type "yolov8:bounding-box", with confidence and class name)
- Querying the stored predictions with the ibis query API

This is the "round-trip" pattern: UPD is not just an archive you read once —
it is a living dataset you can enrich with model outputs, human corrections,
or any downstream processing, and then query across all of it uniformly.

Why use a pre-trained model here?
----------------------------------
COCO8 contains only 8 images.  That is intentional — it is a smoke-test
fixture, not a training set.  Training a detection model on 8 images would
produce nonsense.  Using a model that was already trained on the full COCO
dataset (80 classes, 118k images) and simply running it on these 8 images is
the honest and useful thing to do: you see real predictions, and you can scale
to a larger UPD file without changing a single line of this script.

Prerequisites
-------------
    pip install upd ultralytics pillow

Generate the UPD file first:
    python coco8_to_upd.py --output coco8.upd

Then run this script:
    python examples/infer_coco8_yolo.py --upd coco8.upd
    python examples/infer_coco8_yolo.py --upd coco8.upd --conf 0.4 --model yolov8s.pt
"""

from __future__ import annotations

import argparse
import io
import json

import numpy as np
from PIL import Image
from ultralytics import YOLO

from upd import UPD


# Shape type used for all annotations created by this script.
# The vendor prefix "yolov8" makes it easy to distinguish these
# machine-generated predictions from the original ground-truth boxes
# (which use "coco8-image:bounding-box").
PRED_SHAPE_TYPE = "yolov8:bounding-box"


# ---------------------------------------------------------------------------
# Step 1 — Run inference on every image in a UPD dataset
# ---------------------------------------------------------------------------

def infer_dataset(
    upd: UPD,
    model: YOLO,
    dataset_name: str,
    conf_threshold: float,
) -> tuple[int, int]:
    """
    Run *model* on every entry in *dataset_name* and write predictions back.

    For each detected object we create one annotation with:
        shape_type : "yolov8:bounding-box"
        shape_args : {"cx": …, "cy": …, "w": …, "h": …}   ← normalised [0,1]
        category   : class_name
        properties : {"class_id": …, "confidence": …}
        metadata   : {"model": …, "QC-Status": "Predicted"}

    Coordinates are stored normalised (0–1 relative to image size) so they
    are resolution-independent — the same format used by the ground-truth
    annotations written by coco8_to_upd.py.

    Returns
    -------
    (n_images, n_predictions) — counts for the progress summary
    """
    # Find the dataset row by name using the ibis filter API.
    t  = upd.datasets.table
    df = upd.datasets.filter(t.name == dataset_name).execute()
    if df.empty:
        available = upd.datasets.select("name").execute()["name"].tolist()
        raise ValueError(f"Dataset {dataset_name!r} not found. Available: {available}")
    ds_id   = df.iloc[0]["id"]
    ds_name = df.iloc[0]["name"]

    model_name = model.model_name if hasattr(model, "model_name") else "yolov8"
    n_images, n_preds = 0, 0

    for entry in upd.entries.iter_for_dataset(ds_id):
        media = upd.medias.get(entry.local_media_id)
        if media is None or media.blob_data is None:
            continue

        # Decode the stored JPEG bytes into a PIL image for YOLO.
        pil_img = Image.open(io.BytesIO(media.blob_data)).convert("RGB")
        img_w, img_h = pil_img.size

        # Run YOLO inference.  verbose=False suppresses per-image console output.
        results = model.predict(pil_img, conf=conf_threshold, verbose=False)
        boxes   = results[0].boxes   # Ultralytics Boxes object

        # Collect all detections for this entry in one bulk_create call.
        ann_batch = []
        for i in range(len(boxes)):
            # xyxy → cx, cy, w, h (all normalised to [0, 1])
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            w  = (x2 - x1) / img_w
            h  = (y2 - y1) / img_h

            class_id   = int(boxes.cls[i].item())
            confidence = float(boxes.conf[i].item())
            class_name = model.names[class_id]

            ann_batch.append({
                "entry_id":   entry.id,
                "shape_type": PRED_SHAPE_TYPE,
                "shape_args": {"cx": cx, "cy": cy, "w": w, "h": h},
                "category":   class_name,
                "properties": {
                    "class_id":   class_id,
                    "confidence": round(confidence, 4),
                },
                "metadata": {
                    "Model":      model_name,
                    "QC-Status":  "Predicted",
                },
            })

        if ann_batch:
            upd.annotations.bulk_create(ann_batch)
            n_preds += len(ann_batch)

        n_images += 1
        print(f"  {ds_name} — {entry.id[:8]}… : {len(ann_batch)} prediction(s)")

    return n_images, n_preds


# ---------------------------------------------------------------------------
# Step 2 — Query the stored predictions to verify the round-trip
# ---------------------------------------------------------------------------

def print_prediction_summary(upd: UPD) -> None:
    """
    Use the ibis query API to summarise what was written back to the file.

    This is the point of the round-trip: once predictions are stored as
    annotations, you can query them the same way you query ground truth —
    filter by shape_type, group by class, join with entries, etc.
    """
    print("\n── Stored predictions (ibis query) ────────────────────────────")

    # Count predictions per class, sorted by frequency.
    t  = upd.annotations.table
    df = (
        t.filter(t.shape_type == PRED_SHAPE_TYPE)
         .group_by("shape_type")   # grouping key — illustrates the API
         .aggregate(n=t.id.count())
         .execute()
    )
    total = int(df["n"].sum()) if not df.empty else 0
    print(f"Total predictions stored : {total}")

    # Show all predictions with their confidence scores.
    # properties is stored as a JSON string — we parse it in Python.
    all_preds = upd.annotations.filter(
        upd.annotations.shape_type == PRED_SHAPE_TYPE
    ).execute()

    if all_preds.empty:
        print("No predictions found.")
        return

    # Parse the JSON properties column and extract class names + confidence.
    rows = []
    for _, row in all_preds.iterrows():
        props = json.loads(row["properties"])
        rows.append((row["category"], props["confidence"]))

    rows.sort(key=lambda x: -x[1])
    print(f"\n{'Class':<20} {'Confidence':>10}")
    print("-" * 32)
    for class_name, conf in rows:
        print(f"{class_name:<20} {conf:>10.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    # Load model — downloads weights on first run (~6 MB for yolov8n).
    print(f"Loading model {args.model} …")
    model = YOLO(args.model)
    print(f"Model loaded ({len(model.names)} classes)\n")

    with UPD.open(args.upd) as upd:
        total_images = 0
        total_preds  = 0

        for split in ("COCO8 Train", "COCO8 Val"):
            print(f"── Running inference on: {split} ──")
            n_img, n_pred = infer_dataset(upd, model, split, args.conf)
            total_images += n_img
            total_preds  += n_pred

        print(f"\nDone. {total_images} images → {total_preds} predictions stored in {args.upd}")

        # Demonstrate that the written predictions are immediately queryable.
        print_prediction_summary(upd)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run YOLOv8 inference on COCO8 UPD and store predictions back."
    )
    p.add_argument("--upd",   required=True, help="Path to coco8.upd")
    p.add_argument("--model", default="yolov8n.pt",
                   help="Ultralytics model name or path (default: yolov8n.pt)")
    p.add_argument("--conf",  type=float, default=0.25,
                   help="Confidence threshold for detections (default: 0.25)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())