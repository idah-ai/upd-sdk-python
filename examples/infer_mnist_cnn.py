"""
infer_mnist_cnn.py
==================
Run a trained MNIST CNN on every image stored in a UPD file and write the
predicted digit labels back as annotations.

This is the inference counterpart to train_mnist_cnn.py.  It mirrors the
round-trip pattern used in infer_coco8_yolo.py:

    mnist.upd
      read images  →  trained CNN  →  predicted digit + confidence score
                                              │
                                              ▼
                                   write back to mnist.upd as new annotations
                                              │
                                              ▼
                                   query predictions with ibis

After this script runs, mnist.upd contains both the original ground-truth
labels (shape_type "mnist:classification", written by mnist_to_upd.py) and
the model's predictions (shape_type "mnist-cnn:classification").  You can
then query both with the same ibis API to measure accuracy, find mistakes,
or inspect low-confidence predictions.

Prerequisites
-------------
    pip install upd torch pillow

Train the model first:
    python examples/train_mnist_cnn.py --upd mnist.upd --output mnist_cnn.pt

Then run inference:
    python examples/infer_mnist_cnn.py --upd mnist.upd --model mnist_cnn.pt
    python examples/infer_mnist_cnn.py --upd mnist.upd --model mnist_cnn.pt --split "MNIST Test"
"""

from __future__ import annotations

import argparse
import io

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from upd import UPD

# Shape type written by this script.  The vendor prefix "mnist-cnn" keeps
# these predictions distinct from the ground-truth "mnist:classification"
# annotations written by mnist_to_upd.py.
PRED_SHAPE_TYPE = "mnist-cnn:classification"


# ---------------------------------------------------------------------------
# Model definition — must match train_mnist_cnn.py exactly
# ---------------------------------------------------------------------------

class MnistCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1   = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2   = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool    = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.4)
        self.fc1     = nn.Linear(64 * 7 * 7, 128)
        self.fc2     = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Step 1 — Load the trained model
# ---------------------------------------------------------------------------

def load_model(model_path: str, device: torch.device) -> MnistCNN:
    """
    Restore a model from weights saved by train_mnist_cnn.py.

    torch.load loads the state dict (the learned weights).
    We then push the whole model to the target device (CPU / GPU / MPS)
    and switch it to eval mode, which disables Dropout — during inference
    we want deterministic outputs, not random neuron dropout.
    """
    model = MnistCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()   # Dropout off, BatchNorm uses running stats (if any)
    return model


# ---------------------------------------------------------------------------
# Step 2 — Run inference on every image in a UPD dataset
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_dataset(
    upd: UPD,
    model: MnistCNN,
    device: torch.device,
    dataset_name: str,
) -> tuple[int, int]:
    """
    Run *model* on every entry in *dataset_name* and write predictions back.

    For each image we create one annotation with:
        shape_type : "mnist-cnn:classification"
        shape_args : {}                               (no geometry for classification)
        annotation : {"label": int, "confidence": float, "scores": [float × 10]}
        metadata   : {"model": "mnist_cnn", "QC-Status": "Predicted"}

    The ``scores`` list contains the softmax probability for each digit (0–9),
    useful for inspecting uncertain predictions.

    Returns
    -------
    (n_images, n_correct) — for accuracy reporting
    """
    # Find the dataset by name.
    t  = upd.datasets.table
    df = upd.datasets.filter(t.name == dataset_name).execute()
    if df.empty:
        available = upd.datasets.select("name").execute()["name"].tolist()
        raise ValueError(f"Dataset {dataset_name!r} not found. Available: {available}")
    ds_id   = df.iloc[0]["id"]
    ds_name = df.iloc[0]["name"]

    n_images, n_correct = 0, 0

    for entry in upd.entries.iter_for_dataset(ds_id):
        media = upd.medias.get(entry.local_media_id)
        if media is None or media.blob_data is None:
            continue

        # Decode PNG → float32 tensor (1, 1, 28, 28) — batch dimension required.
        img = Image.open(io.BytesIO(media.blob_data)).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
        x   = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,28,28)

        # Forward pass — @torch.no_grad() means no gradient memory is allocated.
        logits     = model(x)                                  # (1, 10)
        probs      = F.softmax(logits, dim=1).squeeze(0)       # (10,)
        pred_label = int(probs.argmax().item())
        confidence = float(probs[pred_label].item())
        scores     = [round(float(p), 4) for p in probs.tolist()]

        # Check against ground truth (for accuracy tracking, not stored).
        gt_anns = upd.annotations.for_entry(entry.id)
        gt_anns = [a for a in gt_anns if a.shape_type == "mnist:classification"]
        if gt_anns:
            gt_label = int(gt_anns[0].annotation["label"])
            if pred_label == gt_label:
                n_correct += 1

        # Write the prediction back as a new annotation.
        upd.annotations.create(
            entry_id=entry.id,
            shape_type=PRED_SHAPE_TYPE,
            shape_args={},
            annotation={
                "label":      pred_label,
                "confidence": round(confidence, 4),
                "scores":     scores,
            },
            metadata={
                "Model":     "mnist_cnn",
                "QC-Status": "Predicted",
            },
        )

        n_images += 1

    print(f"  {ds_name}: {n_images} images, accuracy {n_correct}/{n_images} "
          f"({n_correct / n_images * 100:.2f}%)")
    return n_images, n_correct


# ---------------------------------------------------------------------------
# Step 3 — Query the stored predictions to verify the round-trip
# ---------------------------------------------------------------------------

def print_prediction_summary(upd: UPD) -> None:
    """
    Use the ibis query API to show mistakes and low-confidence predictions.

    Once predictions are stored as annotations they are queryable exactly
    like ground-truth data — filter, join, group_by all work the same way.
    """
    print("\n── Stored predictions (ibis query) ────────────────────────────")

    t = upd.annotations.table

    # Total prediction count.
    n_preds = upd.annotations.filter(
        t.shape_type == PRED_SHAPE_TYPE
    ).count().execute()
    print(f"Total predictions stored : {n_preds}")

    # Join predictions with ground truth to find mistakes.
    # We read both annotation sets and compare in Python — a join across
    # two filtered views of the same table is cleaner this way.
    pred_df = upd.annotations.filter(t.shape_type == PRED_SHAPE_TYPE).execute()
    gt_df   = upd.annotations.filter(t.shape_type == "mnist:classification").execute()

    if pred_df.empty or gt_df.empty:
        return

    import json

    # Build a lookup: entry_id → ground-truth label.
    gt_by_entry = {
        row["entry_id"]: json.loads(row["annotation"])["label"]
        for _, row in gt_df.iterrows()
    }

    mistakes, low_conf = [], []
    for _, row in pred_df.iterrows():
        ann        = json.loads(row["annotation"])
        pred_label = ann["label"]
        confidence = ann["confidence"]
        gt_label   = gt_by_entry.get(row["entry_id"])

        if gt_label is not None and pred_label != gt_label:
            mistakes.append((row["entry_id"][:8], gt_label, pred_label, confidence))

        if confidence < 0.90:
            low_conf.append((row["entry_id"][:8], pred_label, confidence))

    if mistakes:
        print(f"\nMistakes ({len(mistakes)}):")
        print(f"  {'Entry':>10}  {'GT':>4}  {'Pred':>6}  {'Conf':>8}")
        print("  " + "-" * 34)
        for entry_id, gt, pred, conf in mistakes[:10]:   # show at most 10
            print(f"  {entry_id}…  {gt:>4}  {pred:>6}  {conf:>8.4f}")
    else:
        print("\nNo mistakes — perfect accuracy on this split.")

    if low_conf:
        print(f"\nLow-confidence predictions (<0.90) — ({len(low_conf)}):")
        print(f"  {'Entry':>10}  {'Pred':>6}  {'Conf':>8}")
        print("  " + "-" * 28)
        for entry_id, pred, conf in sorted(low_conf, key=lambda r: r[2])[:10]:
            print(f"  {entry_id}…  {pred:>6}  {conf:>8.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps"  if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"PyTorch {torch.__version__}  |  device: {device}")

    print(f"Loading model from {args.model} …")
    model = load_model(args.model, device)
    print("Model loaded.\n")

    with UPD.open(args.upd) as upd:
        total_images, total_correct = infer_dataset(
            upd, model, device, args.split
        )
        print_prediction_summary(upd)

    print(f"\nOverall accuracy: {total_correct}/{total_images} "
          f"({total_correct / total_images * 100:.2f}%)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a trained MNIST CNN on a UPD file and store predictions."
    )
    p.add_argument("--upd",   required=True, help="Path to mnist.upd")
    p.add_argument("--model", required=True, help="Path to mnist_cnn.pt")
    p.add_argument("--split", default="MNIST Test",
                   help='Dataset name to run inference on (default: "MNIST Test")')
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())