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
           .select("id", "shape_type", "category")
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
    category : str
        Single classification string for the annotation (e.g. a class label).
    properties : dict
        Open JSON bag for any additional semantic payload.
    metadata : dict
        Lifecycle metadata.  Recommended keys: ``Created-At``,
        ``Created-By``, ``QC-Status``, ``Confidence``.
    """

    id: str
    entry_id: str
    shape_type: str
    shape_args: dict[str, Any]
    category: str
    properties: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def _from_record(cls, r: dict) -> "Annotation":
        return cls(
            id=r["id"],
            entry_id=r["entry_id"],
            shape_type=r["shape_type"],
            shape_args=json_loads(r["shape_args"]),
            category=r["category"],
            properties=json_loads(r["properties"]),
            metadata=json_loads(r["metadata"]),
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
        category: str,
        properties: Optional[dict[str, Any]] = None,
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
        category:
            Single classification string for the annotation (e.g. a class label).
        properties:
            Optional JSON-serialisable dict with any additional semantic payload.
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
        if not category or not isinstance(category, str):
            raise ValueError("category must be a non-empty string")
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

        props = properties or {}
        self.insert([{
            "id":         ann_id,
            "entry_id":   entry_id,
            "shape_type": shape_type,
            "shape_args": json_dumps(shape_args),
            "category":   category,
            "properties": json_dumps(props),
            "metadata":   json_dumps(meta),
        }])
        return Annotation(
            id=ann_id, entry_id=entry_id, shape_type=shape_type,
            shape_args=shape_args, category=category, properties=props, metadata=meta,
        )

    def bulk_create(self, annotations: list[dict[str, Any]]) -> list[Annotation]:
        """
        Insert multiple annotations in a single transaction.

        Significantly faster than calling ``create()`` in a loop for large
        batches (one commit instead of N).

        Each dict must contain: ``entry_id``, ``shape_type``, ``shape_args``,
        ``category``.  Optional keys: ``properties``, ``id``, ``metadata``,
        ``created_by``, ``qc_status``.

        Returns
        -------
        list[Annotation]

        Note
        ----
        Uses raw SQL to maintain explicit transaction control
        (BEGIN / COMMIT / ROLLBACK), which ibis does not expose.
        """
        now     = utc_now()
        results = []
        self._conn.execute("BEGIN")
        try:
            for row in annotations:
                validate_id(row["entry_id"], field="entry_id")
                if len(row["shape_type"]) > 64:
                    raise ValueError("shape_type must be ≤ 64 characters")
                if not row.get("category") or not isinstance(row.get("category"), str):
                    raise ValueError("category must be a non-empty string")
                ann_id = row.get("id") or new_id()
                validate_id(ann_id)
                meta = dict(row.get("metadata") or {})
                meta.setdefault("Created-At", now)
                meta.setdefault("Updated-At", now)
                if row.get("created_by"):
                    meta.setdefault("Created-By", row["created_by"])
                if row.get("qc_status"):
                    meta.setdefault("QC-Status", row["qc_status"])
                props = row.get("properties") or {}
                self._conn.execute(
                    "INSERT INTO annotations "
                    "(id, entry_id, shape_type, shape_args, category, properties, metadata) "
                    "VALUES (?,?,?,?,?,?,?)",
                    [ann_id, row["entry_id"], row["shape_type"],
                     json_dumps(row["shape_args"]), row["category"],
                     json_dumps(props), json_dumps(meta)],
                )
                results.append(Annotation(
                    id=ann_id, entry_id=row["entry_id"], shape_type=row["shape_type"],
                    shape_args=row["shape_args"], category=row["category"],
                    properties=props, metadata=meta,
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
        t  = self.table
        df = t.filter(t.id == id).execute()
        if df.empty:
            return None
        return Annotation._from_record(df.iloc[0].to_dict())

    def for_entry(self, entry_id: str) -> list[Annotation]:
        """
        Return all annotations for *entry_id* as a list.

        For large batches consider :meth:`iter_for_entry`.
        """
        t  = self.table
        records = (
            t.filter(t.entry_id == entry_id)
             .order_by("id")
             .execute()
             .to_dict("records")
        )
        return [Annotation._from_record(r) for r in records]

    def for_dataset(self, dataset_id: str) -> list[Annotation]:
        """
        Return all annotations across every entry in *dataset_id*.

        Uses an ibis ``semi_join`` to filter by dataset without exposing
        ambiguous join columns in the result.
        """
        a = self.table
        e = self._ibis.table("entries")
        records = (
            a.semi_join(e.filter(e.dataset_id == dataset_id), a.entry_id == e.id)
             .order_by("id")
             .execute()
             .to_dict("records")
        )
        return [Annotation._from_record(r) for r in records]

    def iter_for_dataset(self, dataset_id: str, *, batch_size: int = 1000):
        """
        Yield annotations for *dataset_id* one at a time, fetching in batches.

        Preferred over :meth:`for_dataset` when iterating large datasets::

            for ann in upd.annotations.iter_for_dataset(ds.id):
                process(ann.category)

        Note
        ----
        Uses raw SQL with explicit LIMIT/OFFSET for memory-bounded pagination,
        which ibis does not expose at the cursor level.
        """
        offset = 0
        while True:
            rows = self._conn.execute(
                "SELECT a.id, a.entry_id, a.shape_type, a.shape_args, "
                "       a.category, a.properties, a.metadata "
                "FROM annotations a "
                "JOIN entries e ON a.entry_id = e.id "
                "WHERE e.dataset_id = ? ORDER BY a.id "
                "LIMIT ? OFFSET ?",
                [dataset_id, batch_size, offset],
            ).fetchall()
            if not rows:
                break
            for id_, entry_id, shape_type, shape_args, category, properties, meta in rows:
                yield Annotation(
                    id=id_, entry_id=entry_id, shape_type=shape_type,
                    shape_args=json_loads(shape_args), category=category,
                    properties=json_loads(properties), metadata=json_loads(meta),
                )
            offset += batch_size

    def all(self) -> list[Annotation]:
        """Return every annotation in the file as a list."""
        records = self.table.order_by("id").execute().to_dict("records")
        return [Annotation._from_record(r) for r in records]

    def count_for_dataset(self, dataset_id: str) -> int:
        """
        Return the number of annotations across all entries in *dataset_id*.
            n = upd.annotations.count_for_dataset(ds.id)
        """
        a = self.table
        e = self._ibis.table("entries")
        return int(
            a.semi_join(e.filter(e.dataset_id == dataset_id), a.entry_id == e.id)
             .count()
             .execute()
        )

    # ------------------------------------------------------------------
    # Update  (raw SQL — ibis has no UPDATE support)
    # ------------------------------------------------------------------

    def update(
        self,
        id: str,
        *,
        shape_type: Optional[str] = None,
        shape_args: Optional[dict[str, Any]] = None,
        category: Optional[str] = None,
        properties: Optional[dict[str, Any]] = None,
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

        new_st    = shape_type if shape_type is not None else existing.shape_type
        new_sa    = shape_args if shape_args is not None else existing.shape_args
        new_cat   = category if category is not None else existing.category
        new_props = properties if properties is not None else existing.properties
        new_meta  = existing.metadata.copy()
        if metadata is not None:
            new_meta.update(metadata)
        if qc_status is not None:
            new_meta["QC-Status"] = qc_status
        new_meta["Updated-At"] = utc_now()

        self._conn.execute(
            "UPDATE annotations "
            "SET shape_type=?, shape_args=?, category=?, properties=?, metadata=? "
            "WHERE id=?",
            [new_st, json_dumps(new_sa), new_cat, json_dumps(new_props),
             json_dumps(new_meta), id],
        )
        self._conn.commit()
        return Annotation(
            id=id, entry_id=existing.entry_id, shape_type=new_st,
            shape_args=new_sa, category=new_cat, properties=new_props, metadata=new_meta,
        )

    # ------------------------------------------------------------------
    # Delete  (raw SQL — RETURNING clause not available via ibis)
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

    def delete_for_dataset(self, dataset_id: str) -> int:
        """
        Delete all annotations across every entry in *dataset_id*.

        Uses a subquery to resolve the entry → dataset relationship in one
        SQL statement.  Returns the count of annotations removed.

        Typically called before :meth:`EntryRepository.delete_for_dataset`
        to satisfy the ``ON DELETE RESTRICT`` foreign key on ``entry_id``.
        Use :meth:`UPD.delete_dataset` to handle the full cascade automatically.
        """
        rows = self._conn.execute(
            "DELETE FROM annotations "
            "WHERE entry_id IN (SELECT id FROM entries WHERE dataset_id = ?) "
            "RETURNING id",
            [dataset_id],
        ).fetchall()
        self._conn.commit()
        return len(rows)
