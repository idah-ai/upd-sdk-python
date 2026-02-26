# upd

**Python library for Universal Portable Dataset (UPD) files.**

`upd` provides a clean, ORM-like API for creating, reading, updating, and
deleting data in [UPD files](./RFC.md) — portable DuckDB databases designed
to store and share AI/ML datasets with full provenance tracking.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [API Reference](#api-reference)
  - [UPD](#UPD)
  - [MetadataRepository](#metadatarepository)
  - [DatasetRepository](#datasetrepository)
  - [MediaRepository](#mediarepository)
  - [EntryRepository](#entryrepository)
  - [AnnotationRepository](#annotationrepository)
- [Querying](#querying)
- [UPD Schema Overview](#upd-schema-overview)
- [Running Tests](#running-tests)
- [Examples](#examples)
  - [MNIST → UPD](#mnist--upd)
  - [training from MNIST UPD](#training-from-MNIST-UPD)
  - [MNIST inference with the trained CNN (inference + round-trip)](#mnist-inference-with-the-trained-cnn-inference--round-trip)
  - [COCO8 → UPD](#coco8--upd)
  - [COCO8 object detection with YOLOv8 (inference + round-trip)
    ](#coco8-object-detection-with-yolov8-inference--round-trip)

---

## Installation

```bash
pip install .
```

> **Requirements:** Python ≥ 3.14, `duckdb >= 1.4.4`, `ibis-framework[duckdb] >= 12.0`

---

## Quick Start

```python
from upd import UPD

with UPD.open("my_dataset.upd") as upd:

    # ── 1. Create a dataset ───────────────────────────────────────────────
    ds = upd.datasets.create(
        name="My Image Dataset",
        modality="image",
        created_by="alice@example.com",
    )

    # ── 2. Embed a media file ─────────────────────────────────────────────
    media = upd.medias.create_from_file("photo.jpg")
    # — or from raw bytes —
    media = upd.medias.create(blob_data=image_bytes, media_type="image/png")

    # ── 3. Create an entry linking dataset → media ────────────────────────
    entry = upd.entries.create(
        dataset_id=ds.id,
        media_url=media.local_url,   # "local:<uuid>"
    )

    # ── 4. Annotate ───────────────────────────────────────────────────────
    upd.annotations.create(
        entry_id=entry.id,
        shape_type="bounding-box",
        shape_args={"x": 10, "y": 20, "width": 100, "height": 80},
        annotation={"class": "cat", "confidence": 0.95},
        qc_status="Passed",
    )

    # ── 5. Query ──────────────────────────────────────────────────────────
    t  = upd.annotations.table
    df = upd.annotations.filter(t.shape_type == "bounding-box").execute()
    print(df)
```

---

## Architecture

```
upd/
├── _connection.py   # UPD — main facade
├── _schema.py       # SQL DDL constants (CREATE TABLE …)
├── _utils.py        # UUIDv7, JSON helpers, validators
├── _base.py         # _BaseRepository — transparent query delegation
├── metadata.py      # MetadataRepository  ← global key/value config
├── dataset.py       # DatasetRepository   ← logical data groupings
├── media.py         # MediaRepository     ← binary blobs / references
├── entry.py         # EntryRepository     ← data point ↔ media links
└── annotation.py    # AnnotationRepository ← shapes + labels
```

Each repository inherits from `_BaseRepository`, which delegates any
unrecognised attribute to the underlying table expression. This means query
operations (`filter`, `select`, `join`, `order_by`, `group_by`, `limit`,
`aggregate`, `execute`, …) work **directly on the repository object** — the
repository is the query entry point.

```
UPD
├── .metadata    → MetadataRepository
├── .datasets    → DatasetRepository
├── .medias      → MediaRepository
├── .entries     → EntryRepository
└── .annotations → AnnotationRepository
```

---

## API Reference

### UPD

```python
from upd import UPD

# Open / create (read-write)
upd = UPD.open("path/to/file.upd")

# Read-only
upd = UPD.open("path/to/file.upd", read_only=True)

# In-memory (ephemeral)
upd = UPD.open(":memory:")

# Context manager (auto-close)
with UPD.open("file.upd") as upd:
    ...

# Direct SQL for advanced / Flavor-specific queries
upd.execute("SELECT * FROM my_flavor_table WHERE dataset_id = ?", ["ds-001"])
upd.raw_connection.execute("SELECT …").df()
```

---

### MetadataRepository `upd.metadata`

Stores file-level key/value configuration (RFC §3.2).

| Method                          | Description                                |
| ------------------------------- | ------------------------------------------ |
| `get(key)`                      | Return decoded value for _key_, or `None`. |
| `set(key, value)`               | Upsert a JSON-serialisable value.          |
| `delete(key)`                   | Remove a key. Returns `True` if found.     |
| `all()`                         | Return all entries as `dict[str, Any]`.    |
| `table`                         | Underlying table expression.               |
| `filter(*predicates)`           | Shorthand: `table.filter(*predicates)`.    |
| `select(…)` / `order_by(…)` / … | Delegated directly to the table.           |

```python
upd.metadata.set("Authored-By", "alice@example.com")
print(upd.metadata.get("Schema-Version"))   # "0.2"

t  = upd.metadata.table
df = upd.metadata.filter(t.key.startswith("Schema")).execute()
df = upd.metadata.select("key").execute()
```

---

### DatasetRepository `upd.datasets`

Logical groupings of data (RFC §3.3).

| Method                                  | Signature                                       | Description                             |
| --------------------------------------- | ----------------------------------------------- | --------------------------------------- |
| `create`                                | `(name, modality, *, id, metadata, created_by)` | Insert. Returns `Dataset`.              |
| `get`                                   | `(id)`                                          | Fetch by PK. Returns `Dataset \| None`. |
| `all`                                   | `()`                                            | `list[Dataset]`.                        |
| `update`                                | `(id, *, name, modality, metadata)`             | Partial update.                         |
| `delete`                                | `(id)`                                          | Delete (fails if entries exist).        |
| `table`                                 | property                                        | Underlying table expression.            |
| `filter(*predicates)` / `select(…)` / … | Delegated to table.                             |

```python
ds = upd.datasets.create(name="COCO Train", modality="image")

t  = upd.datasets.table
df = upd.datasets.filter(t.modality == "image").execute()
df = upd.datasets.order_by("name").select("id", "name").execute()
```

**`Dataset` dataclass**

```python
@dataclass
class Dataset:
    id: str
    name: str
    modality: str
    metadata: dict[str, Any]
```

---

### MediaRepository `upd.medias`

Binary blobs or external media references (RFC §3.4).

| Method                                  | Signature                                       | Description                              |
| --------------------------------------- | ----------------------------------------------- | ---------------------------------------- |
| `create`                                | `(*, id, key, blob_data, media_type, metadata)` | Insert a media row.                      |
| `create_from_file`                      | `(path, *, id, key, media_type, metadata)`      | Embed a file as BLOB.                    |
| `get`                                   | `(id, key="")`                                  | Fetch by composite PK (includes blob).   |
| `get_all_keys`                          | `(id)`                                          | All rows for a composite group.          |
| `all`                                   | `()`                                            | All rows, including blobs.               |
| `update`                                | `(id, key, *, …)`                               | Partial update.                          |
| `delete`                                | `(id, key="")`                                  | Delete by composite PK.                  |
| `delete_all_keys`                       | `(id)`                                          | Delete an entire composite group.        |
| `table`                                 | property                                        | Table expression (`blob_data` excluded). |
| `filter(*predicates)` / `select(…)` / … | Delegated to table.                             |

```python
m = upd.medias.create_from_file("photo.jpg")
print(m.local_url)   # "local:<uuid>"

# Composite media (map tiles, multi-band imagery, …)
upd.medias.create(id="map-001", key="tile-00.png", blob_data=t0, media_type="image/png")
upd.medias.create(id="map-001", key="tile-01.png", blob_data=t1, media_type="image/png")
tiles = upd.medias.get_all_keys("map-001")

# Query (blob_data not included in expressions)
t  = upd.medias.table
df = upd.medias.filter(t.media_type == "image/png").select("id", "key").execute()
```

**`Media` dataclass**

```python
@dataclass
class Media:
    id: str
    key: str                  # "" for single-file media
    blob_data: bytes | None
    media_type: str | None
    metadata: dict[str, Any]
    local_url: str            # property → "local:<id>"
```

---

### EntryRepository `upd.entries`

Individual data points linking a dataset to media (RFC §3.5).

| Method                            | Signature                                              | Description                          |
| --------------------------------- | ------------------------------------------------------ | ------------------------------------ |
| `create`                          | `(dataset_id, media_url, *, id, metadata, created_by)` | Insert.                              |
| `bulk_create`                     | `(entries: list[dict])`                                | Insert many in one pass.             |
| `get`                             | `(id)`                                                 | Fetch by PK.                         |
| `for_dataset`                     | `(dataset_id)`                                         | All entries in a dataset.            |
| `all`                             | `()`                                                   | All entries.                         |
| `count`                           | `(dataset_id=None)`                                    | Count, optionally per dataset.       |
| `update`                          | `(id, *, media_url, metadata)`                         | Partial update.                      |
| `delete`                          | `(id)`                                                 | Delete (fails if annotations exist). |
| `table` / `filter` / `select` / … | Query delegation.                                      |

```python
entry = upd.entries.create(dataset_id=ds.id, media_url="local:m-001")

rows = [{"dataset_id": ds.id, "media_url": f"local:{mid}"} for mid in media_ids]
entries = upd.entries.bulk_create(rows)

t  = upd.entries.table
df = upd.entries.filter(t.dataset_id == ds.id).limit(100).execute()
```

**`Entry` dataclass**

```python
@dataclass
class Entry:
    id: str
    dataset_id: str
    media_url: str
    metadata: dict[str, Any]
    is_local: bool           # property
    local_media_id: str | None  # property
```

---

### AnnotationRepository `upd.annotations`

Structured labels for entries (RFC §3.6).

| Method                            | Signature                                              | Description                          |
| --------------------------------- | ------------------------------------------------------ | ------------------------------------ |
| `create`                          | `(entry_id, shape_type, shape_args, annotation, *, …)` | Insert.                              |
| `bulk_create`                     | `(annotations: list[dict])`                            | Insert many.                         |
| `get`                             | `(id)`                                                 | Fetch by PK.                         |
| `for_entry`                       | `(entry_id)`                                           | All annotations for one entry.       |
| `for_dataset`                     | `(dataset_id)`                                         | All annotations across a dataset.    |
| `all`                             | `()`                                                   | All annotations.                     |
| `count`                           | `(entry_id=None)`                                      | Count, optionally per entry.         |
| `update`                          | `(id, *, …, qc_status)`                                | Partial update.                      |
| `delete`                          | `(id)`                                                 | Delete by PK.                        |
| `delete_for_entry`                | `(entry_id)`                                           | Delete all annotations for an entry. |
| `table` / `filter` / `select` / … | Query delegation.                                      |

```python
ann = upd.annotations.create(
    entry_id=entry.id,
    shape_type="bounding-box",
    shape_args={"x": 10, "y": 20, "width": 50, "height": 30},
    annotation={"class": "person", "confidence": 0.97},
    qc_status="Passed",
)

upd.annotations.update(ann.id, qc_status="Flagged")
```

**`Annotation` dataclass**

```python
@dataclass
class Annotation:
    id: str
    entry_id: str
    shape_type: str
    shape_args: dict[str, Any]
    annotation: dict[str, Any]
    metadata: dict[str, Any]
```

---

## Querying

Every repository inherits `_BaseRepository`, which delegates unknown attribute
lookups to the underlying ibis table expression.

```python
with UPD.open("my_dataset.upd") as upd:

    # ── .table gives you the base expression to build predicates from ─────
    t  = upd.datasets.table
    df = t.filter(t.modality == "image").execute()

    # ── .filter(predicate) is a direct shorthand — result is chainable ────
    t  = upd.entries.table
    df = upd.entries.filter(t.dataset_id == ds.id).execute()

    # ── Any operation works directly on the repo ──────────────────────────
    df = upd.annotations.select("id", "shape_type", "annotation").execute()
    df = upd.datasets.order_by("name").limit(10).execute()

    # ── Full expressive chains ────────────────────────────────────────────
    t  = upd.annotations.table
    df = (
        upd.annotations
           .filter(t.shape_type == "bounding-box")
           .select("id", "entry_id", "annotation")
           .order_by("id")
           .limit(100)
           .execute()
    )

    # ── Group-by / aggregate ──────────────────────────────────────────────
    t  = upd.annotations.table
    df = (
        upd.annotations
           .group_by("shape_type")
           .aggregate(n=t.id.count())
           .order_by("n", ascending=False)
           .execute()
    )

    # ── Join across tables ────────────────────────────────────────────────
    e  = upd.entries.table
    a  = upd.annotations.table
    df = (
        e
        .join(a, e.id == a.entry_id)
        .filter(e.dataset_id == ds.id)
        .select(e.id.name("entry_id"), a.shape_type, a.annotation)
        .execute()
    )

    # ── Raw SQL for JSON extraction, Flavor tables, complex sub-queries ───
    df = upd.raw_connection.execute("""
        SELECT id,
               json_extract_string(annotation, '$.class')      AS class_name,
               json_extract_string(annotation, '$.confidence') AS confidence
        FROM   annotations
        WHERE  json_extract_string(annotation, '$.confidence')::DOUBLE > 0.9
    """).df()
```

---

## UPD Schema Overview

```
metadata          ← global key/value config (Schema-Type, Schema-Version, …)
│
datasets          ← logical dataset groupings (name, modality, …)
│   │
│   └── entries   ← one row per data point (media_url → local or external)
│           │
│           └── annotations  ← shapes + labels for each entry
│
medias            ← binary blobs, composite PK (id, key)
```

All primary keys use **UUIDv7** (timestamp-ordered, via `uuid.uuid7()` in
Python 3.14) for full lifecycle traceability. All JSON fields are stored as
`VARCHAR` for maximum DuckDB compatibility.

For the complete schema specification, see [RFC.md](./RFC.md).

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest
pytest --cov=upd --cov-report=term-missing
pytest tests/test_annotation.py -v
```

| Test file                  | Source file         |
| -------------------------- | ------------------- |
| `tests/test_metadata.py`   | `upd/metadata.py`   |
| `tests/test_dataset.py`    | `upd/dataset.py`    |
| `tests/test_media.py`      | `upd/media.py`      |
| `tests/test_entry.py`      | `upd/entry.py`      |
| `tests/test_annotation.py` | `upd/annotation.py` |

---

## Examples

Both examples use a standard library to handle the dataset download and
expose items as Python objects

```bash
pip install torchvision   # MNIST
pip install ultralytics   # COCO8
```

### MNIST → UPD

`torchvision` downloads MNIST and returns each item as a PIL image + integer
label. The example converts those directly into UPD entries.

```bash
python examples/mnist_to_upd.py --output mnist.upd
python examples/mnist_to_upd.py --output mnist.upd --limit 500   # quick test
```

| Entity      | Count (full) | Details                                          |
| ----------- | ------------ | ------------------------------------------------ |
| datasets    | 2            | "MNIST Train", "MNIST Test"                      |
| medias      | 70,000       | PNG blobs, 28×28 grayscale                       |
| entries     | 70,000       | `local:<uuid>` links                             |
| annotations | 70,000       | `shape_type = "mnist:classification"`, label 0–9 |

```python
# Re-open and query
from upd import UPD

with UPD.open("mnist.upd") as upd:
    t  = upd.annotations.table
    df = upd.raw_connection.execute("""
        SELECT json_extract_string(annotation, '$.label') AS digit, count(*) AS n
        FROM   annotations GROUP BY digit ORDER BY digit
    """).df()
    print(df)
```

---

### training from MNIST UPD

**Script:** `train_mnist_cnn.py`
**Dataset:** `mnist.upd` (generated by `mnist_to_upd.py`)
**Framework:** PyTorch
**Concept:** Training a model from scratch

#### What it does

MNIST is the "hello world" of machine learning — 70 000 grayscale images of
handwritten digits (0–9), each 28×28 pixels. The task is to classify each
image into one of 10 digit classes.

The script loads images and labels directly from the UPD file, trains a small
convolutional neural network (CNN), and reports test accuracy.

#### How to run

```bash
# 1. Generate the dataset
python mnist_to_upd.py --output mnist.upd

# 2. Train
python examples/train_mnist_cnn.py --upd mnist.upd

# 3. Optional — tweak
python examples/train_mnist_cnn.py --upd mnist.upd --epochs 10 --batch-size 128
```

Expected output (5 epochs):

```
Epoch 1/5 — loss: 0.2341 — accuracy: 0.9301 — val_accuracy: 0.9782
Epoch 2/5 — loss: 0.0721 — accuracy: 0.9782 — val_accuracy: 0.9851
...
Test accuracy : 98.43%
Model saved → mnist_cnn.pt
```

---

### MNIST inference with the trained CNN (inference + round-trip)

**Script:** `infer_mnist_cnn.py`
**Dataset:** `mnist.upd` (generated by `mnist_to_upd.py`)
**Framework:** PyTorch
**Concept:** Running a trained model on new data + writing predictions back to UPD

#### What it does

Once a model is trained, the typical next step is to run it on new data and
store the results somewhere useful. This script reads every image from the
UPD file, runs the CNN trained by `train_mnist_cnn.py`, and writes each
prediction back into the same file as a new annotation.

After it runs, `mnist.upd` contains two annotation types per entry:

| `shape_type`               | Written by           | Content                                         |
| -------------------------- | -------------------- | ----------------------------------------------- |
| `mnist:classification`     | `mnist_to_upd.py`    | ground-truth digit label                        |
| `mnist-cnn:classification` | `infer_mnist_cnn.py` | predicted label + confidence + per-class scores |

Both are queryable with the same API. The script then demonstrates this
by joining them to report mistakes and low-confidence predictions.

#### How to run

```bash
# 1. Generate the UPD file
python mnist_to_upd.py --output mnist.upd

# 2. Train the model
python examples/train_mnist_cnn.py --upd mnist.upd --output mnist_cnn.pt

# 3. Run inference and write predictions back
python examples/infer_mnist_cnn.py --upd mnist.upd --model mnist_cnn.pt

# 4. Optionally run on the training split too
python examples/infer_mnist_cnn.py --upd mnist.upd --model mnist_cnn.pt --split "MNIST Train"
```

Expected output:

```
PyTorch 2.x.x  |  device: cpu
Loading model from mnist_cnn.pt …
Model loaded.

  MNIST Test: 10000 images, accuracy 9843/10000 (98.43%)

── Stored predictions (ibis query) ────────────────────────────
Total predictions stored : 10000

Mistakes (157):
  Entry       GT    Pred      Conf
  ----------------------------------
  0a3f21b8…    5       3    0.7201
  1c88fa02…    4       9    0.8834
  ...

Low-confidence predictions (<0.90) — (312):
  Entry       Pred      Conf
  ----------------------------
  3d09ab12…     8    0.5123
  ...
```

---

### COCO8 → UPD

`ultralytics` downloads COCO8 on first run and exposes each item as a numpy
image array with class IDs and normalised bounding boxes.

```bash
python examples/coco8_to_upd.py --output coco8.upd
```

| Entity      | Count | Details                                   |
| ----------- | ----- | ----------------------------------------- |
| datasets    | 2     | "COCO8 Train", "COCO8 Val"                |
| medias      | 8     | JPEG blobs                                |
| entries     | 8     | `local:<uuid>` links                      |
| annotations | ~30   | `shape_type = "coco8-image:bounding-box"` |

```python
# Re-open and query
from upd import UPD

with UPD.open("coco8.upd") as upd:
    t  = upd.annotations.table
    df = upd.annotations.group_by("shape_type").aggregate(n=t.id.count()).execute()
    print(df)
```

---

### COCO8 object detection with YOLOv8 (inference + round-trip)

**Script:** `infer_coco8_yolo.py`
**Dataset:** `coco8.upd` (generated by `coco8_to_upd.py`)
**Framework:** Ultralytics YOLOv8
**Concept:** Inference with a pre-trained model + writing predictions back to UPD

#### What it does

Object detection is harder than classification: the model must find _where_
each object is (a bounding box) and _what_ it is (a class label), and there
can be many objects per image.

COCO8 contains only 8 images — it exists as a quick smoke-test, not a
training set. The right approach for a dataset this small is to use a model
already trained on the full COCO dataset (118 000 images, 80 classes) and run
it on these images.

The script demonstrates the UPD **round-trip pattern** using trained yolo model

```
coco8.upd
  read images →  YOLOv8 →  predicted boxes + classes + confidence scores
                                    │
                                    ▼
                         write back to coco8.upd as new annotations
                                    │
                                    ▼
                         query predictions with ibis  (same API as ground truth)
```

After the script runs, `coco8.upd` contains both the original ground-truth
annotations (written by `coco8_to_upd.py`, shape_type
`"coco8-image:bounding-box"`) and the model's predictions (shape_type
`"yolov8:bounding-box"`). You can query, compare, and filter both with the
same API.

The vendor-prefixed `shape_type = "yolov8:bounding-box"` keeps predictions
distinguishable from ground-truth boxes so you can filter on one, the other,
or both.

#### How to run

```bash
# 1. Generate the dataset
python coco8_to_upd.py --output coco8.upd

# 2. Run inference and write predictions back
python examples/infer_coco8_yolo.py --upd coco8.upd

# 3. Optional — stricter confidence filter or larger model
python examples/infer_coco8_yolo.py --upd coco8.upd --conf 0.4 --model yolov8s.pt
```

Expected output:

```
Loading model yolov8n.pt …
Model loaded (80 classes)

── Running inference on: COCO8 Train ──
  COCO8 Train — 000002d3… : 3 prediction(s)
  COCO8 Train — 0000048b… : 5 prediction(s)
  ...

Done. 8 images → 27 predictions stored in coco8.upd

── Stored predictions (ibis query) ────────────────────────────
Total predictions stored : 27

Class                Confidence
--------------------------------
person                    0.9201
dog                       0.8734
car                       0.7102
...
```

---
