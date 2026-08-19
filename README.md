# upd

**Python library for Universal Portable Dataset (UPD) files.**

`upd` provides a clean API for creating, reading, updating, and deleting data
in UPD files — portable DuckDB databases designed to store and share AI/ML
datasets with full provenance tracking.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [API Reference](#api-reference)
  - [UPD](#upd-facade)
  - [MetadataRepository](#metadatarepository)
  - [DatasetRepository](#datasetrepository)
  - [MediaRepository](#mediarepository)
  - [EntryRepository](#entryrepository)
  - [AnnotationRepository](#annotationrepository)
- [Querying with ibis](#querying-with-ibis)
- [On dataclasses and ibis results](#on-dataclasses-and-ibis-results)
- [Schema overview](#schema-overview)
- [Running tests](#running-tests)
- [Dataset conversion examples](#dataset-conversion-examples)
- [ML examples](#ml-examples)

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

    # 1. Create a dataset
    ds = upd.datasets.create(name="My Images", modality="image")

    # 2. Embed a media file
    media = upd.medias.create_from_file("photo.jpg")

    # 3. Link dataset → media via an entry
    entry = upd.entries.create(dataset_id=ds.id, media_url=media.local_url)

    # 4. Annotate
    upd.annotations.create(
        entry_id=entry.id,
        shape_type="bounding-box",
        shape_args={"x": 10, "y": 20, "width": 100, "height": 80},
        category="cat",
        properties={"confidence": 0.95},
        qc_status="Passed",
    )

    # 5. Query with ibis
    t  = upd.annotations.table
    df = upd.annotations.filter(t.shape_type == "bounding-box").execute()

    # 6. Cascade delete
    upd.delete_dataset(ds.id)   # removes annotations → entries → dataset
```

---

## Architecture

```
upd/
├── _connection.py   # UPD — main facade + cascade helpers
├── _schema.py       # SQL DDL (CREATE TABLE …)
├── _utils.py        # UUIDv7, JSON helpers, validators
├── _base.py         # _BaseRepository — ibis query delegation
├── metadata.py      # MetadataRepository  ← global key/value config
├── dataset.py       # DatasetRepository   ← logical groupings
├── media.py         # MediaRepository     ← binary blobs / references
├── entry.py         # EntryRepository     ← data point ↔ media links
└── annotation.py    # AnnotationRepository ← shapes + labels
```

Each repository inherits `_BaseRepository`, which delegates any unrecognised
attribute to the underlying ibis table expression. Query operations —
`filter`, `select`, `join`, `order_by`, `group_by`, `limit`, `aggregate`,
`execute` — work **directly on the repository object**.

---

## API Reference

### UPD facade

```python
# Open / create (read-write)
upd = UPD.open("file.upd")

# Read-only
upd = UPD.open("file.upd", read_only=True)

# In-memory (ephemeral)
upd = UPD.open(":memory:")

# Context manager
with UPD.open("file.upd") as upd:
    ...
```

| Method / property              | Description                                                     |
| ------------------------------ | --------------------------------------------------------------- |
| `UPD.open(path, *, read_only)` | Open or create a UPD file.                                      |
| `close()`                      | Flush and close the connection.                                 |
| `delete_dataset(id)`           | Cascade-delete annotations → entries → dataset. Returns `bool`. |
| `execute(sql, params)`         | Run raw SQL. Returns a DuckDB result.                           |
| `raw_connection`               | Direct `duckdb.DuckDBPyConnection` for advanced use.            |
| `path`                         | Filesystem path of the underlying file.                         |
| `.metadata`                    | `MetadataRepository`                                            |
| `.datasets`                    | `DatasetRepository`                                             |
| `.medias`                      | `MediaRepository`                                               |
| `.entries`                     | `EntryRepository`                                               |
| `.annotations`                 | `AnnotationRepository`                                          |

**Cascade delete**

Because the schema uses `ON DELETE RESTRICT` foreign keys, rows must be
removed in order: annotations first, then entries, then the dataset.
`delete_dataset` handles this automatically:

```python
deleted = upd.delete_dataset(ds.id)   # True if the dataset existed
```

To delete selectively without removing the dataset itself:

```python
n_anns    = upd.annotations.delete_for_dataset(ds.id)
n_entries = upd.entries.delete_for_dataset(ds.id)
```

---

### MetadataRepository — `upd.metadata`

Stores file-level key/value configuration (RFC §3.2).

| Method            | Returns          | Description                       |
| ----------------- | ---------------- | --------------------------------- |
| `get(key)`        | `Any \| None`    | Decoded value, or `None`.         |
| `set(key, value)` | `None`           | Upsert a JSON-serialisable value. |
| `delete(key)`     | `bool`           | Remove a key.                     |
| `all()`           | `dict[str, Any]` | Every key/value as a dict.        |

```python
upd.metadata.set("Authored-By", "alice@example.com")
upd.metadata.get("Schema-Version")   # "0.2"

# ibis query
t  = upd.metadata.table
df = upd.metadata.filter(t.key.startswith("Schema")).execute()
```

---

### DatasetRepository — `upd.datasets`

Logical groupings of data (RFC §3.3).

| Method                                                | Returns           | Description                      |
| ----------------------------------------------------- | ----------------- | -------------------------------- |
| `create(name, modality, *, id, metadata, created_by)` | `Dataset`         | Insert.                          |
| `get(id)`                                             | `Dataset \| None` | Fetch by PK.                     |
| `all()`                                               | `list[Dataset]`   | Every dataset, ordered by id.    |
| `update(id, *, name, modality, metadata)`             | `Dataset \| None` | Partial update.                  |
| `delete(id)`                                          | `bool`            | Delete (fails if entries exist). |

```python
ds = upd.datasets.create(name="COCO Train", modality="image")

t  = upd.datasets.table
df = upd.datasets.filter(t.modality == "image").order_by("name").execute()
```

**`Dataset`**

```python
@dataclass
class Dataset:
    id: str
    name: str
    modality: str
    metadata: dict[str, Any]
```

---

### MediaRepository — `upd.medias`

Binary blobs or external media references (RFC §3.4).

The composite primary key is `(id, key)`. Use `key=""` (default) for
single-file media; use distinct keys for related files (tiles, bands, …)
that share the same `id`.

> `blob_data` is **excluded** from the ibis `table` expression for
> performance. Use `get()` or `get_all_keys()` to retrieve binary content.

| Method                                                     | Returns         | Description                                    |
| ---------------------------------------------------------- | --------------- | ---------------------------------------------- |
| `create(*, id, key, blob_data, media_type, metadata)`      | `Media`         | Insert a row.                                  |
| `create_from_file(path, *, id, key, media_type, metadata)` | `Media`         | Read a file and embed it as BLOB.              |
| `get(id, key="")`                                          | `Media \| None` | Fetch by composite PK (includes blob).         |
| `get_all_keys(id)`                                         | `list[Media]`   | All rows for a composite group.                |
| `all()`                                                    | `list[Media]`   | All rows without blob data (`blob_data=None`). |
| `update(id, key, *, blob_data, media_type, metadata)`      | `Media \| None` | Partial update.                                |
| `delete(id, key="")`                                       | `bool`          | Delete one row.                                |
| `delete_all_keys(id)`                                      | `int`           | Delete a whole composite group. Returns count. |

```python
media = upd.medias.create_from_file("photo.jpg")
media.local_url   # "local:<uuid>"  — use in entries.media_url

# Composite group (e.g. map tiles)
upd.medias.create(id="map-001", key="tile-00.png", blob_data=t0, media_type="image/png")
upd.medias.create(id="map-001", key="tile-01.png", blob_data=t1, media_type="image/png")
tiles = upd.medias.get_all_keys("map-001")    # list[Media] with blob_data populated

# Query (blob_data column not present)
t  = upd.medias.table
df = upd.medias.filter(t.media_type == "image/png").select("id", "key").execute()
```

**`Media`**

```python
@dataclass
class Media:
    id: str
    key: str                    # "" for single-file media
    blob_data: bytes | None
    media_type: str | None
    metadata: dict[str, Any]
    local_url: str              # property → "local:<id>"
```

---

### EntryRepository — `upd.entries`

Individual data points linking a dataset to media (RFC §3.5).

| Method                                                       | Returns           | Description                                       |
| ------------------------------------------------------------ | ----------------- | ------------------------------------------------- |
| `create(dataset_id, media_url, *, id, metadata, created_by)` | `Entry`           | Insert.                                           |
| `bulk_create(entries)`                                       | `list[Entry]`     | Insert many in one transaction.                   |
| `get(id)`                                                    | `Entry \| None`   | Fetch by PK.                                      |
| `for_dataset(dataset_id)`                                    | `list[Entry]`     | All entries in a dataset.                         |
| `iter_for_dataset(dataset_id, *, batch_size)`                | `Iterator[Entry]` | Streaming, batched. Preferred for large datasets. |
| `all()`                                                      | `list[Entry]`     | Every entry.                                      |
| `update(id, *, media_url, metadata)`                         | `Entry \| None`   | Partial update.                                   |
| `delete(id)`                                                 | `bool`            | Delete by PK (fails if annotations exist).        |
| `delete_for_dataset(dataset_id)`                             | `int`             | Delete all entries for a dataset. Returns count.  |

```python
entry = upd.entries.create(dataset_id=ds.id, media_url=media.local_url)

# Bulk insert
rows    = [{"dataset_id": ds.id, "media_url": m.local_url} for m in medias]
entries = upd.entries.bulk_create(rows)

# Streaming iteration (ML training loops)
for entry in upd.entries.iter_for_dataset(ds.id):
    blob = upd.medias.get(entry.local_media_id).blob_data

# ibis query
t  = upd.entries.table
df = upd.entries.filter(t.dataset_id == ds.id).count().execute()
```

**`Entry`**

```python
@dataclass
class Entry:
    id: str
    dataset_id: str
    media_url: str
    metadata: dict[str, Any]
    is_local: bool              # property — True when media_url starts with "local:"
    local_media_id: str | None  # property — the medias.id referenced
```

---

### AnnotationRepository — `upd.annotations`

Structured labels for entries (RFC §3.6).

| Method                                                                                         | Returns                | Description                                             |
| ---------------------------------------------------------------------------------------------- | ---------------------- | ------------------------------------------------------- |
| `create(entry_id, shape_type, shape_args, category, properties, *, id, metadata, created_by, qc_status)` | `Annotation`           | Insert.                                                 |
| `bulk_create(annotations)`                                                                     | `list[Annotation]`     | Insert many in one transaction.                         |
| `get(id)`                                                                                      | `Annotation \| None`   | Fetch by PK.                                            |
| `for_entry(entry_id)`                                                                          | `list[Annotation]`     | All annotations for one entry.                          |
| `for_dataset(dataset_id)`                                                                      | `list[Annotation]`     | All annotations across a dataset.                       |
| `iter_for_dataset(dataset_id, *, batch_size)`                                                  | `Iterator[Annotation]` | Streaming, batched.                                     |
| `all()`                                                                                        | `list[Annotation]`     | Every annotation.                                       |
| `count_for_dataset(dataset_id)`                                                                | `int`                  | Annotation count for a dataset.                         |
| `update(id, *, shape_type, shape_args, category, properties, metadata, qc_status)`                       | `Annotation \| None`   | Partial update.                                         |
| `delete(id)`                                                                                   | `bool`                 | Delete by PK.                                           |
| `delete_for_entry(entry_id)`                                                                   | `int`                  | Delete all annotations for an entry. Returns count.     |
| `delete_for_dataset(dataset_id)`                                                               | `int`                  | Delete all annotations across a dataset. Returns count. |

```python
ann = upd.annotations.create(
    entry_id=entry.id,
    shape_type="bounding-box",
    shape_args={"x": 10, "y": 20, "width": 50, "height": 30},
    category="person",
    properties={"confidence": 0.97},
    qc_status="Passed",
)

upd.annotations.update(ann.id, qc_status="Flagged")

# Scoped count via ibis
n = upd.annotations.filter(
    upd.annotations.entry_id == entry.id
).count().execute()

# All annotations for one entry (typed list)
anns = upd.annotations.for_entry(entry.id)
```

**`Annotation`**

```python
@dataclass
class Annotation:
    id: str
    entry_id: str
    shape_type: str
    shape_args: dict[str, Any]
    category: str
    properties: dict[str, Any]
    metadata: dict[str, Any]
```

---

## Querying with ibis

Every repository delegates unknown attributes to its ibis table expression,
so the full ibis query API is available directly:

```python
with UPD.open("file.upd") as upd:

    # Predicate from .table
    t  = upd.annotations.table
    df = upd.annotations.filter(t.shape_type == "bounding-box").execute()

    # Full chain
    df = (
        upd.annotations
           .filter(t.shape_type == "bounding-box")
           .select("id", "entry_id", "category")
           .order_by("id")
           .limit(100)
           .execute()
    )

    # Group-by / aggregate
    df = (
        upd.annotations
           .group_by("shape_type")
           .aggregate(n=t.id.count())
           .order_by("n", ascending=False)
           .execute()
    )

    # Join — use .table on the right-hand side (ibis type requirement)
    e  = upd.entries.table
    a  = upd.annotations.table
    df = (
        e.join(a, e.id == a.entry_id)
         .filter(e.dataset_id == ds.id)
         .select(e.id.name("entry_id"), a.shape_type, a.category)
         .execute()
    )

    # Semi-join (filter without exposing joined columns)
    e = upd.entries.table
    a = upd.annotations.table
    df = (
        a.semi_join(e.filter(e.dataset_id == ds.id), a.entry_id == e.id)
         .execute()
    )

    # Scoped count
    n = upd.entries.filter(
        upd.entries.dataset_id == ds.id
    ).count().execute()

    # Raw SQL for JSON extraction or Flavor-specific tables
    df = upd.raw_connection.execute("""
        SELECT id,
               category AS class_name,
               CAST(json_extract_string(properties, '$.confidence') AS DOUBLE) AS confidence
        FROM   annotations
        WHERE  CAST(json_extract_string(properties, '$.confidence') AS DOUBLE) > 0.9
    """).df()
```

---

## On dataclasses and ibis results

The library returns **dataclasses** (`Dataset`, `Entry`, `Annotation`, `Media`)
from write operations and typed read helpers, and **pandas DataFrames** from
ibis expressions. These serve different purposes:

| Situation                                                                       | Use                            |
| ------------------------------------------------------------------------------- | ------------------------------ |
| After `create()` / `update()` — need the new row's id or properties immediately | Dataclass                      |
| Iterating entries in an ML loop — `entry.local_media_id`, `entry.is_local`      | Dataclass (`iter_for_dataset`) |
| Any filter, aggregation, join, or subquery                                      | ibis → DataFrame               |
| Passing data into a further ibis expression                                     | ibis → DataFrame               |

The list-returning methods (`for_entry`, `for_dataset`, `all`) are convenience
wrappers for simple lookups. For anything more complex — filters, counts,
joins — use the ibis query API directly.

---

## Schema overview

```
metadata          ← file-level key/value config (Schema-Type, Schema-Version, …)

datasets          ← logical groupings (name, modality)
  └── entries     ← one row per data point (dataset_id FK, media_url)
        └── annotations ← shapes + labels (entry_id FK, shape_type, shape_args, category, properties)

medias            ← binary blobs, composite PK (id, key)
```

All primary keys use **UUIDv7** (`uuid.uuid7()`, Python 3.14 stdlib) for
timestamp-ordered, globally unique identifiers. All JSON fields are stored
as `VARCHAR` for maximum DuckDB compatibility.

Foreign keys use `ON DELETE RESTRICT` — use `upd.delete_dataset()` or the
individual `delete_for_*` methods to remove rows in the correct order.

---

## Running tests

```bash
pip install -e ".[dev]"
pytest
pytest --cov=upd --cov-report=term-missing
pytest tests/test_annotation.py -v   # single module
```

---

## Dataset conversion examples

```bash
pip install "upd[examples]"   # torchvision + ultralytics
```

### MNIST → UPD

```bash
python examples/mnist_to_upd.py --output mnist.upd
python examples/mnist_to_upd.py --output mnist.upd --limit 500  # quick test
```

| Table       | Rows (full) | Detail                                           |
| ----------- | ----------- | ------------------------------------------------ |
| datasets    | 2           | "MNIST Train", "MNIST Test"                      |
| medias      | 70 000      | PNG blobs, 28×28 grayscale                       |
| entries     | 70 000      | `local:<uuid>` links                             |
| annotations | 70 000      | `shape_type = "mnist:classification"`, label 0–9 |

### COCO8 → UPD

```bash
python examples/coco8_to_upd.py --output coco8.upd
```

| Table       | Rows | Detail                                                          |
| ----------- | ---- | --------------------------------------------------------------- |
| datasets    | 2    | "COCO8 Train", "COCO8 Val"                                      |
| medias      | 8    | JPEG blobs                                                      |
| entries     | 8    | `local:<uuid>` links                                            |
| annotations | ~30  | `shape_type = "coco8-image:bounding-box"`, cx/cy/w/h normalised |

---

## ML examples

### Example 1 — Train a CNN on MNIST (PyTorch)

```bash
python examples/train_mnist_cnn.py --upd mnist.upd
python examples/train_mnist_cnn.py --upd mnist.upd --epochs 10 --output mnist_cnn.pt
```

Trains a small convolutional network (~98% test accuracy in 5 epochs).
Demonstrates reading BLOBs and labels from UPD into a `torch.utils.data.Dataset`.

### Example 2 — Run inference and write predictions back (MNIST)

```bash
python examples/infer_mnist_cnn.py --upd mnist.upd --model mnist_cnn.pt
```

Reads images from UPD, runs the trained model, writes predicted labels back
as `shape_type = "mnist-cnn:classification"` annotations alongside the
original ground truth. Shows the **round-trip pattern**: UPD as both source
and destination.

### Example 3 — YOLOv8 inference on COCO8

```bash
python examples/infer_coco8_yolo.py --upd coco8.upd
```

Uses a pre-trained YOLOv8n model (no training needed — COCO8 is too small).
Writes predicted bounding boxes back as `shape_type = "yolov8:bounding-box"`
annotations, then queries them with ibis alongside the original ground-truth boxes.
