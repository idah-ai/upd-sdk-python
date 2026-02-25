"""
DatasetRepository — CRUD operations for the ``datasets`` table.

A *dataset* is a logical grouping of data points sharing the same modality
(e.g. images, text, audio).  See UPD RFC §3.3.

Because this class inherits from ``_BaseRepository``, every query operation
works directly on the repository object:

    # Predicates are built from .table
    t  = upd.datasets.table
    df = upd.datasets.filter(t.modality == "image").execute()

    # Full chain from the repo
    df = (
        upd.datasets
           .filter(t.modality.startswith("vanilla"))
           .select("id", "name", "modality")
           .order_by("name")
           .execute()
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import duckdb

from ._base import _BaseRepository
from ._utils import new_id, utc_now, json_dumps, json_loads, validate_id, validate_name


@dataclass
class Dataset:
    """
    Represents a row in the ``datasets`` table.

    Attributes
    ----------
    id : str
        UUIDv7 primary key (≤ 64 chars).
    name : str
        Human-readable dataset name (≤ 64 chars).
    modality : str
        Vendor-prefixed modality string, e.g. ``"image"`` (≤ 64 chars).
    metadata : dict
        Arbitrary JSON metadata.  Recommended keys: ``Created-At``,
        ``Updated-At``, ``Created-By``, ``Dataset-Specification-URL``.
    """

    id: str
    name: str
    modality: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def _from_record(cls, r: dict) -> "Dataset":
        return cls(
            id=r["id"],
            name=r["name"],
            modality=r["modality"],
            metadata=json_loads(r["metadata"]),
        )


class DatasetRepository(_BaseRepository):
    """
    CRUD access to the ``datasets`` table.

    Parameters
    ----------
    conn:
        Raw DuckDB connection (used for writes).
    ibis_conn:
        Query backend connection (used for reads).
    """

    _table_name = "datasets"

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create(
        self,
        name: str,
        modality: str,
        *,
        id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        created_by: Optional[str] = None,
    ) -> Dataset:
        """
        Insert a new dataset and return the resulting :class:`Dataset`.

        Parameters
        ----------
        name:
            Human-readable name (≤ 64 chars).
        modality:
            Vendor-prefixed modality string (≤ 64 chars).
        id:
            Explicit primary key.  A UUIDv7 is generated when omitted.
        metadata:
            Additional JSON metadata.  ``Created-At`` and ``Updated-At``
            are populated automatically when absent.
        created_by:
            Shortcut to set ``Created-By`` in metadata.

        Returns
        -------
        Dataset
        """
        validate_name(name)
        validate_name(modality, field="modality")
        ds_id = id or new_id()
        validate_id(ds_id)

        now  = utc_now()
        meta = metadata or {}
        meta.setdefault("Created-At", now)
        meta.setdefault("Updated-At", now)
        if created_by:
            meta.setdefault("Created-By", created_by)

        self.insert([{
            "id":       ds_id,
            "name":     name,
            "modality": modality,
            "metadata": json_dumps(meta),
        }])
        return Dataset(id=ds_id, name=name, modality=modality, metadata=meta)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, id: str) -> Optional[Dataset]:
        """Retrieve a dataset by primary key, or ``None``."""
        t  = self.table
        df = t.filter(t.id == id).execute()
        if df.empty:
            return None
        return Dataset._from_record(df.iloc[0].to_dict())

    def all(self) -> list[Dataset]:
        """Return all datasets, ordered by ``id``."""
        records = self.table.order_by("id").execute().to_dict("records")
        return [Dataset._from_record(r) for r in records]

    # ------------------------------------------------------------------
    # Update  (raw SQL — ibis has no UPDATE support)
    # ------------------------------------------------------------------

    def update(
        self,
        id: str,
        *,
        name: Optional[str] = None,
        modality: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[Dataset]:
        """
        Update one or more fields.  ``Updated-At`` is refreshed automatically.

        Returns
        -------
        Dataset | None
            The updated dataset, or ``None`` if *id* was not found.
        """
        existing = self.get(id)
        if existing is None:
            return None

        new_name     = name     if name     is not None else existing.name
        new_modality = modality if modality is not None else existing.modality
        new_meta     = existing.metadata.copy()
        if metadata is not None:
            new_meta.update(metadata)
        new_meta["Updated-At"] = utc_now()

        validate_name(new_name)
        validate_name(new_modality, field="modality")

        self._conn.execute(
            "UPDATE datasets SET name=?, modality=?, metadata=? WHERE id=?",
            [new_name, new_modality, json_dumps(new_meta), id],
        )
        self._conn.commit()
        return Dataset(id=id, name=new_name, modality=new_modality, metadata=new_meta)

    # ------------------------------------------------------------------
    # Delete  (raw SQL — RETURNING clause not available via ibis)
    # ------------------------------------------------------------------

    def delete(self, id: str) -> bool:
        """
        Delete by primary key.

        .. note::
            Raises a database error if any entries reference this dataset
            (``ON DELETE RESTRICT``).

        Returns
        -------
        bool
            ``True`` if a row was deleted.
        """
        result = self._conn.execute(
            "DELETE FROM datasets WHERE id = ? RETURNING id", [id]
        ).fetchone()
        self._conn.commit()
        return result is not None
