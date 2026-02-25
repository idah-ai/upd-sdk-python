"""
EntryRepository — CRUD operations for the ``entries`` table.

An *entry* represents a single data point within a dataset, linking a dataset
to a media location (either ``local:<id>`` for embedded media or an external
URL).  See UPD RFC §3.5.

Because this class inherits from ``_BaseRepository``, every query operation
works directly on the repository object:

    df = upd.entries.filter(upd.entries.dataset_id == ds.id).execute()

    # Chain any operations
    df = (
        upd.entries
           .filter(upd.entries.dataset_id == ds.id)
           .select("id", "media_url")
           .limit(50)
           .execute()
    )

    # Join — repositories work as join arguments via to_expr()
    df = (
        upd.entries
           .join(upd.annotations, upd.entries.id == upd.annotations.entry_id)
           .execute()
    )

    # Total entry count
    print(len(upd.entries))

    # Scoped count (per dataset) using ibis count()
    n = upd.entries.filter(upd.entries.dataset_id == ds.id).count().execute()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import duckdb

from ._base import _BaseRepository
from ._utils import new_id, utc_now, json_dumps, json_loads, validate_id


@dataclass
class Entry:
    """
    Represents a row in the ``entries`` table.

    Attributes
    ----------
    id : str
        UUIDv7 primary key (≤ 64 chars).
    dataset_id : str
        Foreign key referencing ``datasets.id``.
    media_url : str
        ``local:<media_id>`` for embedded media, or any external URL.
    metadata : dict
        Arbitrary JSON metadata.  Recommended keys: ``Created-At``,
        ``Updated-At``, ``Created-By``.
    """

    id: str
    dataset_id: str
    media_url: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def _from_record(cls, r: dict) -> "Entry":
        return cls(
            id=r["id"],
            dataset_id=r["dataset_id"],
            media_url=r["media_url"],
            metadata=json_loads(r["metadata"]),
        )

    @property
    def is_local(self) -> bool:
        """``True`` when the entry references locally-embedded media."""
        return self.media_url.startswith("local:")

    @property
    def local_media_id(self) -> Optional[str]:
        """The ``medias.id`` referenced when :attr:`is_local` is ``True``."""
        return self.media_url[len("local:"):] if self.is_local else None


class EntryRepository(_BaseRepository):
    """
    CRUD access to the ``entries`` table.

    Parameters
    ----------
    conn:
        Raw DuckDB connection (used for writes).
    ibis_conn:
        Query backend connection (used for reads).
    """

    _table_name = "entries"

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create(
        self,
        dataset_id: str,
        media_url: str,
        *,
        id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        created_by: Optional[str] = None,
    ) -> Entry:
        """
        Insert a new entry and return the resulting :class:`Entry`.

        Parameters
        ----------
        dataset_id:
            Foreign key referencing an existing dataset.
        media_url:
            ``"local:<media_id>"`` for embedded media, or any URL
            (``https://``, ``s3://``, ``file:///``, …).
        id:
            Explicit primary key.  A UUIDv7 is generated when omitted.
        metadata:
            Additional JSON metadata.
        created_by:
            Shortcut to set ``Created-By`` in metadata.

        Returns
        -------
        Entry
        """
        validate_id(dataset_id, field="dataset_id")
        entry_id = id or new_id()
        validate_id(entry_id)

        now  = utc_now()
        meta = metadata or {}
        meta.setdefault("Created-At", now)
        meta.setdefault("Updated-At", now)
        if created_by:
            meta.setdefault("Created-By", created_by)

        self.insert([{
            "id":         entry_id,
            "dataset_id": dataset_id,
            "media_url":  media_url,
            "metadata":   json_dumps(meta),
        }])
        return Entry(id=entry_id, dataset_id=dataset_id, media_url=media_url, metadata=meta)

    def bulk_create(self, entries: list[dict[str, Any]]) -> list[Entry]:
        """
        Insert multiple entries in a single transaction.

        Significantly faster than calling ``create()`` in a loop for large
        batches (one commit instead of N).

        Each dict must contain ``dataset_id`` and ``media_url``; optional
        keys: ``id``, ``metadata``, ``created_by``.

        Returns
        -------
        list[Entry]

        Note
        ----
        Uses raw SQL to maintain explicit transaction control
        (BEGIN / COMMIT / ROLLBACK), which ibis does not expose.
        """
        now      = utc_now()
        results  = []
        self._conn.execute("BEGIN")
        try:
            for row in entries:
                entry_id = row.get("id") or new_id()
                validate_id(entry_id)
                validate_id(row["dataset_id"], field="dataset_id")
                meta = dict(row.get("metadata") or {})
                meta.setdefault("Created-At", now)
                meta.setdefault("Updated-At", now)
                if row.get("created_by"):
                    meta.setdefault("Created-By", row["created_by"])
                self._conn.execute(
                    "INSERT INTO entries (id, dataset_id, media_url, metadata) VALUES (?,?,?,?)",
                    [entry_id, row["dataset_id"], row["media_url"], json_dumps(meta)],
                )
                results.append(Entry(id=entry_id, dataset_id=row["dataset_id"],
                                     media_url=row["media_url"], metadata=meta))
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return results

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, id: str) -> Optional[Entry]:
        """Retrieve a single entry by primary key, or ``None``."""
        t  = self.table
        df = t.filter(t.id == id).execute()
        if df.empty:
            return None
        return Entry._from_record(df.iloc[0].to_dict())

    def for_dataset(self, dataset_id: str) -> list[Entry]:
        """
        Return all entries in *dataset_id* as a list.

        For large datasets consider :meth:`iter_for_dataset` to avoid
        loading everything into memory at once.
        """
        t  = self.table
        records = (
            t.filter(t.dataset_id == dataset_id)
             .order_by("id")
             .execute()
             .to_dict("records")
        )
        return [Entry._from_record(r) for r in records]

    def iter_for_dataset(self, dataset_id: str, *, batch_size: int = 1000):
        """
        Yield entries for *dataset_id* one at a time, fetching in batches.

        Preferred over :meth:`for_dataset` when iterating large datasets in
        ML training loops — keeps memory usage constant regardless of size::

            for entry in upd.entries.iter_for_dataset(ds.id):
                blob = upd.medias.get(entry.local_media_id).blob_data
                # feed to model …

        Note
        ----
        Uses raw SQL with explicit LIMIT/OFFSET for memory-bounded pagination,
        which ibis does not expose at the cursor level.
        """
        offset = 0
        while True:
            rows = self._conn.execute(
                "SELECT id, dataset_id, media_url, metadata "
                "FROM entries WHERE dataset_id = ? ORDER BY id "
                "LIMIT ? OFFSET ?",
                [dataset_id, batch_size, offset],
            ).fetchall()
            if not rows:
                break
            for row in rows:
                id_, ds_id, media_url, meta = row
                yield Entry(id=id_, dataset_id=ds_id, media_url=media_url,
                            metadata=json_loads(meta))
            offset += batch_size

    def all(self) -> list[Entry]:
        """Return every entry in the file as a list."""
        records = self.table.order_by("id").execute().to_dict("records")
        return [Entry._from_record(r) for r in records]

    # ------------------------------------------------------------------
    # Update  (raw SQL — ibis has no UPDATE support)
    # ------------------------------------------------------------------

    def update(
        self,
        id: str,
        *,
        media_url: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[Entry]:
        """
        Update an existing entry.  ``Updated-At`` is refreshed automatically.

        Returns
        -------
        Entry | None
        """
        existing = self.get(id)
        if existing is None:
            return None

        new_url  = media_url if media_url is not None else existing.media_url
        new_meta = existing.metadata.copy()
        if metadata is not None:
            new_meta.update(metadata)
        new_meta["Updated-At"] = utc_now()

        self._conn.execute(
            "UPDATE entries SET media_url=?, metadata=? WHERE id=?",
            [new_url, json_dumps(new_meta), id],
        )
        self._conn.commit()
        return Entry(id=id, dataset_id=existing.dataset_id, media_url=new_url, metadata=new_meta)

    # ------------------------------------------------------------------
    # Delete  (raw SQL — RETURNING clause not available via ibis)
    # ------------------------------------------------------------------

    def delete(self, id: str) -> bool:
        """
        Delete by primary key.

        .. note::
            Raises a database error if any annotations reference this entry
            (``ON DELETE RESTRICT``).
        """
        result = self._conn.execute(
            "DELETE FROM entries WHERE id = ? RETURNING id", [id]
        ).fetchone()
        self._conn.commit()
        return result is not None
