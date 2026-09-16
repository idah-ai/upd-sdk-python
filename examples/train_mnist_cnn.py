"""
train_mnist_cnn.py
==================
Train a small convolutional neural network (CNN) on MNIST digits,
loading all images and labels directly from a UPD file.

What this script demonstrates
------------------------------
- Opening a UPD file and streaming entries with iter_for_dataset()
- Decoding raw image BLOBs into tensors
- Building a PyTorch DataLoader pipeline
- Defining and training a simple CNN with PyTorch
- Evaluating and saving the trained model

Prerequisites
-------------
    pip install upd torch torchvision pillow

Generate the UPD file first:
    python mnist_to_upd.py --output mnist.upd

Then run this script:
    python examples/train_mnist_cnn.py --upd mnist.upd
    python examples/train_mnist_cnn.py --upd mnist.upd --epochs 10
"""

from __future__ import annotations

import argparse
import io

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from upd import UPD


# ---------------------------------------------------------------------------
# Step 1 — Read data out of the UPD file into a PyTorch Dataset
# ---------------------------------------------------------------------------

class MNISTFromUPD(Dataset):
    """
    A PyTorch Dataset that reads images and labels from a UPD file.

    PyTorch's Dataset contract requires two methods:
        __len__  — total number of samples
        __getitem__ — return the (image_tensor, label) pair at index i

    We load everything into memory upfront here because MNIST fits
    comfortably (~200 MB).  For larger datasets, __getitem__ could
    read from the UPD file lazily instead.
    """

    def __init__(self, upd_path: str, dataset_name: str) -> None:
        self.images: list[torch.Tensor] = []
        self.labels: list[int] = []

        with UPD.open(upd_path, read_only=True) as upd:
            # Find the dataset row by name using the ibis filter API.
            t  = upd.datasets.table
            df = upd.datasets.filter(t.name == dataset_name).execute()
            if df.empty:
                available = upd.datasets.select("name").execute()["name"].tolist()
                raise ValueError(
                    f"Dataset {dataset_name!r} not found. Available: {available}"
                )
            ds_id = df.iloc[0]["id"]

            # Stream entries one at a time — memory usage stays constant
            # regardless of dataset size.
            for entry in upd.entries.iter_for_dataset(ds_id):
                media = upd.medias.get(entry.local_media_id)
                if media is None or media.blob_data is None:
                    continue

                # Decode PNG bytes → float32 tensor of shape (1, 28, 28).
                # Values are normalised to [0, 1].
                img = Image.open(io.BytesIO(media.blob_data)).convert("L")
                arr = np.array(img, dtype=np.float32) / 255.0   # (28, 28)
                tensor = torch.from_numpy(arr).unsqueeze(0)      # (1, 28, 28)
                self.images.append(tensor)

                # Each MNIST entry has exactly one annotation: the digit label.
                anns = upd.annotations.for_entry(entry.id)
                self.labels.append(int(anns[0].category))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.images[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Step 2 — Define the model
# ---------------------------------------------------------------------------

class MnistCNN(nn.Module):
    """
    A small CNN well-suited to 28x28 grayscale images.

    Layer-by-layer explanation
    --------------------------
    Conv2d(1->32, 3x3)   detect low-level features (edges, corners)
    MaxPool2d(2)         halve spatial size, keep strongest activations
    Conv2d(32->64, 3x3)  detect higher-level combinations of features
    MaxPool2d(2)         halve again  ->  feature map is now (64, 7, 7)
    Flatten              unroll to a 1-D vector of length 64x7x7 = 3136
    Linear(3136->128)    learn which combinations predict each digit
    Dropout(0.4)         randomly zero 40% of neurons to prevent overfitting
    Linear(128->10)      one raw score (logit) per digit class (0-9)

    Note: no softmax here — nn.CrossEntropyLoss applies it internally,
    which is numerically more stable.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1   = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2   = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool    = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.4)
        self.fc1     = nn.Linear(64 * 7 * 7, 128)
        self.fc2     = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # (B, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))   # (B, 64,  7,  7)
        x = x.flatten(start_dim=1)             # (B, 3136)
        x = F.relu(self.fc1(x))                # (B, 128)
        x = self.dropout(x)
        return self.fc2(x)                     # (B, 10)


# ---------------------------------------------------------------------------
# Step 3 — Training and evaluation helpers
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one full pass over the training set. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        total      += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate on a DataLoader without computing gradients."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        total      += len(labels)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Step 4 — Train, evaluate, save
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps"  if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"PyTorch {torch.__version__}  |  device: {device}\n")

    # -- Load data from UPD ---------------------------------------------------
    print("Loading MNIST Train from UPD ...")
    train_dataset = MNISTFromUPD(args.upd, "MNIST Train")
    print(f"  {len(train_dataset)} training images")

    print("Loading MNIST Test from UPD ...")
    test_dataset = MNISTFromUPD(args.upd, "MNIST Test")
    print(f"  {len(test_dataset)} test images\n")

    # -- DataLoaders ----------------------------------------------------------
    # shuffle=True for training so the model never sees batches in the same
    # order across epochs — important for generalisation.
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # -- Model, loss, optimiser -----------------------------------------------
    model     = MnistCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ReduceLROnPlateau: halve the learning rate if val_loss stops improving
    # for 2 consecutive epochs.  Helps squeeze out the last bit of accuracy.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=2
    )

    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}\n")

    # -- Training loop --------------------------------------------------------
    best_val_loss     = float("inf")
    best_val_acc      = 0.0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = evaluate(model, test_loader, criterion, device)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:>2}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        # Early stopping: save whenever validation loss improves;
        # stop if it has not improved for 5 consecutive epochs.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), args.output)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 5:
                print("\nEarly stopping triggered — best weights already saved.")
                break

    # -- Final result ---------------------------------------------------------
    print(f"\nBest val accuracy : {best_val_acc * 100:.2f}%")
    print(f"Best val loss     : {best_val_loss:.4f}")
    print(f"Model saved -> {args.output}")
    print(f"Load it later with:  model.load_state_dict(torch.load('{args.output}'))")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a CNN on MNIST loaded from a UPD file."
    )
    p.add_argument("--upd",        required=True,  help="Path to mnist.upd")
    p.add_argument("--epochs",     type=int,   default=5)
    p.add_argument("--batch-size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--output",     default="mnist_cnn.pt",
                   help="Where to save the best model weights")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
