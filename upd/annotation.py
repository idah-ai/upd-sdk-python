"""
AnnotationRepository — CRUD operations for the ``annotations`` table.

An *annotation* stores a structured label for an entry, consisting of a
shape (geometry) and semantic payload.  See UPD RFC §3.6.

Because this class inherits from ``_BaseRepository``, every query operation
works directly on the repository object:

    df = upd.annotations.filter(upd.annotations.shape_type == "bbox").execute()

    # Full chain
    df = (
        upd.annotations
           .filter(upd.annotations.entry_id == entry.id)
           .select("id", "shape_type", "annotation")
           .order_by("id")
           .execute()
    )

    # Group-by
    df = (
        upd.annotations
           .group_by("shape_type")
           .aggregate(n=upd.annotations.id.count())
           .execute()
    )

    # Total count
    print(len(upd.annotations))

    # Scoped count using ibis count()
    n = upd.annotations.filter(upd.annotations.entry_id == entry.id).count().execute()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import duckdb

from ._base import _BaseRepository
from ._utils import new_id, utc_now, json_dumps, json_loads, validate_id


@dataclass
class Annotation:
    """
    Represents a row in the ``annotations`` table.

    Attributes
    ----------
    id : str
        UUIDv7 primary key (≤ 64 chars).
    entry_id : str
        Foreign key referencing ``entries.id``.
    shape_type : str
        Vendor-prefixed geometry type (≤ 64 chars).
    shape_args : dict
        JSON geometry parameters (structure is shape_type-specific).
    annotation : dict
        JSON semantic payload (e.g. class label, confidence score).
    metadata : dict
        Lifecycle metadata.  Recommended keys: ``Created-At``,
        ``Created-By``, ``QC-Status``, ``Confidence``.
    """

    id: str
    entry_id: str
    shape_type: str
    shape_args: dict[str, Any]
    annotation: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def _from_row(cls, row: tuple) -> "Annotation":
        id_, entry_id, shape_type, shape_args, annotation, meta = row
        return cls(
            id=id_,
            entry_id=entry_id,
            shape_type=shape_type,
            shape_args=json_loads(shape_args),
            annotation=json_loads(annotation),
            metadata=json_loads(meta),
        )


class AnnotationRepository(_BaseRepository):
    """
    CRUD access to the ``annotations`` table.

    Parameters
    ----------
    conn:
        Raw DuckDB connection (used for writes).
    ibis_conn:
        Query backend connection (used for reads).
    """

    _table_name = "annotations"

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create(
        self,
        entry_id: str,
        shape_type: str,
        shape_args: dict[str, Any],
        annotation: dict[str, Any],
        *,
        id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        created_by: Optional[str] = None,
        qc_status: Optional[str] = None,
    ) -> Annotation:
        """
        Insert a new annotation and return the resulting :class:`Annotation`.

        Parameters
        ----------
        entry_id:
            Foreign key referencing an existing entry.
        shape_type:
            Vendor-prefixed shape identifier (≤ 64 chars).
        shape_args:
            JSON-serialisable dict describing the shape geometry.
        annotation:
            JSON-serialisable dict with semantic labels / scores.
        id:
            Explicit primary key.  A UUIDv7 is generated when omitted.
        metadata:
            Additional lifecycle metadata.
        created_by:
            Shortcut to set ``Created-By`` in metadata.
        qc_status:
            Shortcut to set ``QC-Status`` (e.g. ``"Passed"``, ``"Flagged"``).

        Returns
        -------
        Annotation
        """
        validate_id(entry_id, field="entry_id")
        if len(shape_type) > 64:
            raise ValueError("shape_type must be ≤ 64 characters")
        ann_id = id or new_id()
        validate_id(ann_id)

        now  = utc_now()
        meta = metadata or {}
        meta.setdefault("Created-At", now)
        meta.setdefault("Updated-At", now)
        if created_by:
            meta.setdefault("Created-By", created_by)
        if qc_status:
            meta.setdefault("QC-Status", qc_status)

        self._conn.execute(
            "INSERT INTO annotations "
            "(id, entry_id, shape_type, shape_args, annotation, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [ann_id, entry_id, shape_type,
             json_dumps(shape_args), json_dumps(annotation), json_dumps(meta)],
        )
        self._conn.commit()
        return Annotation(
            id=ann_id, entry_id=entry_id, shape_type=shape_type,
            shape_args=shape_args, annotation=annotation, metadata=meta,
        )

    def bulk_create(self, annotations: list[dict[str, Any]]) -> list[Annotation]:
        """
        Insert multiple annotations in a single transaction.

        Significantly faster than calling ``create()`` in a loop for large
        batches (one commit instead of N).

        Each dict must contain: ``entry_id``, ``shape_type``, ``shape_args``,
        ``annotation``.  Optional keys: ``id``, ``metadata``, ``created_by``,
        ``qc_status``.

        Returns
        -------
        list[Annotation]
        """
        now     = utc_now()
        results = []
        self._conn.execute("BEGIN")
        try:
            for row in annotations:
                validate_id(row["entry_id"], field="entry_id")
                if len(row["shape_type"]) > 64:
                    raise ValueError("shape_type must be ≤ 64 characters")
                ann_id = row.get("id") or new_id()
                validate_id(ann_id)
                meta = dict(row.get("metadata") or {})
                meta.setdefault("Created-At", now)
                meta.setdefault("Updated-At", now)
                if row.get("created_by"):
                    meta.setdefault("Created-By", row["created_by"])
                if row.get("qc_status"):
                    meta.setdefault("QC-Status", row["qc_status"])
                self._conn.execute(
                    "INSERT INTO annotations "
                    "(id, entry_id, shape_type, shape_args, annotation, metadata) "
                    "VALUES (?,?,?,?,?,?)",
                    [ann_id, row["entry_id"], row["shape_type"],
                     json_dumps(row["shape_args"]), json_dumps(row["annotation"]),
                     json_dumps(meta)],
                )
                results.append(Annotation(
                    id=ann_id, entry_id=row["entry_id"], shape_type=row["shape_type"],
                    shape_args=row["shape_args"], annotation=row["annotation"], metadata=meta,
                ))
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return results

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, id: str) -> Optional[Annotation]:
        """Retrieve a single annotation by primary key, or ``None``."""
        row = self._conn.execute(
            "SELECT id, entry_id, shape_type, shape_args, annotation, metadata "
            "FROM annotations WHERE id = ?",
            [id],
        ).fetchone()
        return Annotation._from_row(row) if row else None

    def for_entry(self, entry_id: str) -> list[Annotation]:
        """
        Return all annotations for *entry_id* as a list.

        For large batches consider :meth:`iter_for_entry`.
        """
        rows = self._conn.execute(
            "SELECT id, entry_id, shape_type, shape_args, annotation, metadata "
            "FROM annotations WHERE entry_id = ? ORDER BY id",
            [entry_id],
        ).fetchall()
        return [Annotation._from_row(r) for r in rows]

    def iter_for_entry(self, entry_id: str):
        """Yield annotations for *entry_id* one at a time."""
        rows = self._conn.execute(
            "SELECT id, entry_id, shape_type, shape_args, annotation, metadata "
            "FROM annotations WHERE entry_id = ? ORDER BY id",
            [entry_id],
        ).fetchall()
        for row in rows:
            yield Annotation._from_row(row)

    def for_dataset(self, dataset_id: str) -> list[Annotation]:
        """Return all annotations across every entry in *dataset_id*."""
        rows = self._conn.execute(
            "SELECT a.id, a.entry_id, a.shape_type, a.shape_args, "
            "       a.annotation, a.metadata "
            "FROM annotations a "
            "JOIN entries e ON a.entry_id = e.id "
            "WHERE e.dataset_id = ? ORDER BY a.id",
            [dataset_id],
        ).fetchall()
        return [Annotation._from_row(r) for r in rows]

    def iter_for_dataset(self, dataset_id: str, *, batch_size: int = 1000):
        """
        Yield annotations for *dataset_id* one at a time, fetching in batches.

        Preferred over :meth:`for_dataset` when iterating large datasets::

            for ann in upd.annotations.iter_for_dataset(ds.id):
                process(ann.annotation)
        """
        offset = 0
        while True:
            rows = self._conn.execute(
                "SELECT a.id, a.entry_id, a.shape_type, a.shape_args, "
                "       a.annotation, a.metadata "
                "FROM annotations a "
                "JOIN entries e ON a.entry_id = e.id "
                "WHERE e.dataset_id = ? ORDER BY a.id "
                "LIMIT ? OFFSET ?",
                [dataset_id, batch_size, offset],
            ).fetchall()
            if not rows:
                break
            for row in rows:
                yield Annotation._from_row(row)
            offset += batch_size

    def all(self) -> list[Annotation]:
        """Return every annotation in the file as a list."""
        rows = self._conn.execute(
            "SELECT id, entry_id, shape_type, shape_args, annotation, metadata "
            "FROM annotations ORDER BY id"
        ).fetchall()
        return [Annotation._from_row(r) for r in rows]

    def count_for_dataset(self, dataset_id: str) -> int:
        """
        Return the number of annotations across all entries in *dataset_id*.

        Uses a SQL join — prefer this over ibis ``isin(subquery)`` patterns
        which ibis does not support.

            n = upd.annotations.count_for_dataset(ds.id)
        """
        return self._conn.execute(
            "SELECT count(*) FROM annotations a "
            "JOIN entries e ON a.entry_id = e.id "
            "WHERE e.dataset_id = ?",
            [dataset_id],
        ).fetchone()[0]

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        id: str,
        *,
        shape_type: Optional[str] = None,
        shape_args: Optional[dict[str, Any]] = None,
        annotation: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
        qc_status: Optional[str] = None,
    ) -> Optional[Annotation]:
        """
        Update an existing annotation.  ``Updated-At`` is refreshed automatically.

        Returns
        -------
        Annotation | None
        """
        existing = self.get(id)
        if existing is None:
            return None

        new_st   = shape_type if shape_type is not None else existing.shape_type
        new_sa   = shape_args if shape_args is not None else existing.shape_args
        new_ann  = annotation if annotation is not None else existing.annotation
        new_meta = existing.metadata.copy()
        if metadata is not None:
            new_meta.update(metadata)
        if qc_status is not None:
            new_meta["QC-Status"] = qc_status
        new_meta["Updated-At"] = utc_now()

        self._conn.execute(
            "UPDATE annotations "
            "SET shape_type=?, shape_args=?, annotation=?, metadata=? WHERE id=?",
            [new_st, json_dumps(new_sa), json_dumps(new_ann), json_dumps(new_meta), id],
        )
        self._conn.commit()
        return Annotation(
            id=id, entry_id=existing.entry_id, shape_type=new_st,
            shape_args=new_sa, annotation=new_ann, metadata=new_meta,
        )

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, id: str) -> bool:
        """Delete a single annotation by primary key."""
        result = self._conn.execute(
            "DELETE FROM annotations WHERE id = ? RETURNING id", [id]
        ).fetchone()
        self._conn.commit()
        return result is not None

    def delete_for_entry(self, entry_id: str) -> int:
        """Delete all annotations for *entry_id*.  Returns the count removed."""
        rows = self._conn.execute(
            "DELETE FROM annotations WHERE entry_id = ? RETURNING id", [entry_id]
        ).fetchall()
        self._conn.commit()
        return len(rows)
