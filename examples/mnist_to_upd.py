"""
MNIST → UPD

torchvision handles the download and exposes each item as a PIL image

Prerequisites
-------------
    pip install upd torchvision

Usage
-----
    python examples/mnist_to_upd.py --output mnist.upd
    python examples/mnist_to_upd.py --output mnist.upd --limit 500
"""

from __future__ import annotations

import argparse
import hashlib
import io


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output",   default="mnist.upd")
    p.add_argument("--limit",    type=int, default=None, help="Max images per split")
    p.add_argument("--data-dir", default="./mnist_data")
    return p.parse_args()


def image_to_png(img) -> bytes:
    """Convert a PIL image to PNG bytes."""
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def run(output: str, limit: int | None, data_dir: str) -> None:
    import torchvision.datasets as tvd
    from upd import UPD

    splits = {
        "train": tvd.MNIST(data_dir, train=True,  download=True),
        "test":  tvd.MNIST(data_dir, train=False, download=True),
    }

    with UPD.open(output) as upd:
        upd.metadata.set("Schema-Built-By", "mnist_to_upd v1.0")

        for split_name, dataset in splits.items():
            n = min(limit, len(dataset)) if limit else len(dataset)
            print(f"[{split_name}] {n} images")

            ds = upd.datasets.create(
                name=f"MNIST {split_name.capitalize()}",
                modality="mnist-image",
                metadata={
                    "Source": "torchvision MNIST",
                    "Split": split_name,
                    "Num-Classes": 10,
                    "Image-Shape": [28, 28, 1],
                },
            )

            for i in range(n):
                img, label = dataset[i]
                png = image_to_png(img)

                media = upd.medias.create(
                    blob_data=png,
                    media_type="image/png",
                    metadata={"sha256": hashlib.sha256(png).hexdigest()},
                )
                entry = upd.entries.create(
                    dataset_id=ds.id,
                    media_url=media.local_url,
                )
                upd.annotations.create(
                    entry_id=entry.id,
                    shape_type="mnist:classification",
                    shape_args={},
                    annotation={"label": int(label), "num_classes": 10},
                    metadata={"Created-By": "MNIST ground-truth", "QC-Status": "Passed"},
                )

                if (i + 1) % 1000 == 0 or (i + 1) == n:
                    print(f"  {i + 1}/{n}", end="\r", flush=True)
            print()

        for ds in upd.datasets.all():
            n_entries = upd.entries.filter(upd.entries.dataset_id == ds.id).count().execute()
            n_ann     = upd.annotations.count_for_dataset(ds.id)
            print(f"  {ds.name}: {n_entries} entries, {n_ann} annotations total")

    print(f"\nWritten: {output}")


if __name__ == "__main__":
    args = parse_args()
    run(args.output, args.limit, args.data_dir)
