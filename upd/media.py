"""
MediaRepository — CRUD operations for the ``medias`` table.

The ``medias`` table stores binary blobs (or external references) for media
items.  A media item may be a single file (``key = ''``) or multiple related
files sharing the same ``id`` but differing in ``key`` (e.g. map tiles,
multi-band imagery).  See UPD RFC §3.4.

Because this class inherits from ``_BaseRepository``, every query operation
works directly on the repository object.  The ``blob_data`` column is
**excluded** from the base expression for performance — use :meth:`get` or
:meth:`get_all_keys` to retrieve actual binary content.

    t  = upd.medias.table          # blob_data column not present
    df = upd.medias.filter(t.media_type == "image/png").execute()

    # Full chain
    df = (
        upd.medias
           .filter(t.media_type == "image/png")
           .select("id", "key", "metadata")
           .execute()
    )
"""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass, field
from typing import Any, Optional

import duckdb

from ._base import _BaseRepository
from ._utils import new_id, json_dumps, json_loads, validate_id


@dataclass
class Media:
    """
    Represents a row in the ``medias`` table.

    Attributes
    ----------
    id : str
        Shared identifier for a media item (≤ 64 chars).
    key : str
        File discriminator within a composite group.  Must be ``''``
        for single-file media (≤ 256 chars).
    blob_data : bytes | None
        Raw binary content, or ``None`` for external / confidential media.
    media_type : str | None
        MIME type (e.g. ``"image/jpeg"``).
    metadata : dict
        Arbitrary JSON metadata (e.g. a SHA-256 checksum).
    """

    id: str
    key: str = ""
    blob_data: Optional[bytes] = None
    media_type: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def _from_row(cls, row: tuple) -> "Media":
        id_, key, blob, mtype, meta = row
        return cls(
            id=id_,
            key=key,
            blob_data=bytes(blob) if blob is not None else None,
            media_type=mtype,
            metadata=json_loads(meta),
        )

    @property
    def local_url(self) -> str:
        """``local:<id>`` URL for use in ``entries.media_url``."""
        return f"local:{self.id}"


class MediaRepository(_BaseRepository):
    """
    CRUD access to the ``medias`` table.

    The composite primary key is ``(id, key)``.  Use ``key=''`` (default)
    for single-file media items.

    Parameters
    ----------
    conn:
        Raw DuckDB connection (used for writes).
    ibis_conn:
        Query backend connection (used for reads).
    """

    _table_name = "medias"

    @property
    def table(self):
        """
        Return the ``medias`` table with ``blob_data`` excluded.

        Use :meth:`get` or :meth:`get_all_keys` to retrieve binary content.
        """
        return self._ibis.table("medias").drop("blob_data")

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create(
        self,
        *,
        id: Optional[str] = None,
        key: str = "",
        blob_data: Optional[bytes] = None,
        media_type: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Media:
        """
        Insert a new media row and return the resulting :class:`Media`.

        Parameters
        ----------
        id:
            Shared media identifier (≤ 64 chars).  A UUIDv7 is generated
            when omitted.
        key:
            File key within a composite group.  Use ``''`` (default) for
            single-file media.
        blob_data:
            Raw bytes to store as a BLOB.  Pass ``None`` for external media.
        media_type:
            MIME type (e.g. ``"image/png"``).
        metadata:
            Optional JSON-serialisable dict.

        Returns
        -------
        Media
        """
        media_id = id or new_id()
        validate_id(media_id)
        if len(key) > 256:
            raise ValueError(f"media key must be ≤ 256 characters (got {len(key)})")
        if media_type and len(media_type) > 64:
            raise ValueError("media_type must be ≤ 64 characters")

        meta = metadata or {}
        self._conn.execute(
            "INSERT INTO medias (id, key, blob_data, media_type, metadata) "
            "VALUES (?, ?, ?, ?, ?)",
            [media_id, key, blob_data, media_type, json_dumps(meta)],
        )
        self._conn.commit()
        return Media(id=media_id, key=key, blob_data=blob_data, media_type=media_type, metadata=meta)

    def create_from_file(
        self,
        path: str,
        *,
        id: Optional[str] = None,
        key: str = "",
        media_type: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Media:
        """
        Read a file from disk and store it as an embedded BLOB.

        Parameters
        ----------
        path:
            Filesystem path to the media file.
        id, key, media_type, metadata:
            See :meth:`create`.

        Returns
        -------
        Media
        """
        with open(path, "rb") as fh:
            blob = fh.read()

        if media_type is None:
            guessed, _ = mimetypes.guess_type(path)
            media_type = guessed or "application/octet-stream"

        return self.create(id=id, key=key, blob_data=blob, media_type=media_type, metadata=metadata)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, id: str, key: str = "") -> Optional[Media]:
        """Retrieve a single media row by ``(id, key)``, or ``None``."""
        row = self._conn.execute(
            "SELECT id, key, blob_data, media_type, metadata "
            "FROM medias WHERE id = ? AND key = ?",
            [id, key],
        ).fetchone()
        return Media._from_row(row) if row else None

    def get_all_keys(self, id: str) -> list[Media]:
        """Return all rows sharing *id* (a composite media group)."""
        rows = self._conn.execute(
            "SELECT id, key, blob_data, media_type, metadata "
            "FROM medias WHERE id = ? ORDER BY key",
            [id],
        ).fetchall()
        return [Media._from_row(r) for r in rows]

    def all(self) -> list[Media]:
        """
        Return all media rows **without** blob data.

        ``blob_data`` is ``None`` on every returned object — use :meth:`get`
        or :meth:`get_all_keys` to retrieve binary content for specific items.
        Loading blobs for an entire collection at once can exhaust memory.
        """
        rows = self._conn.execute(
            "SELECT id, key, NULL, media_type, metadata FROM medias ORDER BY id, key"
        ).fetchall()
        return [Media._from_row(r) for r in rows]

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        id: str,
        key: str = "",
        *,
        blob_data: Optional[bytes] = None,
        media_type: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[Media]:
        """Update an existing media row.  Only non-``None`` arguments are changed."""
        existing = self.get(id, key)
        if existing is None:
            return None

        new_blob  = blob_data  if blob_data  is not None else existing.blob_data
        new_mtype = media_type if media_type is not None else existing.media_type
        new_meta  = existing.metadata.copy()
        if metadata is not None:
            new_meta.update(metadata)

        self._conn.execute(
            "UPDATE medias SET blob_data=?, media_type=?, metadata=? WHERE id=? AND key=?",
            [new_blob, new_mtype, json_dumps(new_meta), id, key],
        )
        self._conn.commit()
        return Media(id=id, key=key, blob_data=new_blob, media_type=new_mtype, metadata=new_meta)

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, id: str, key: str = "") -> bool:
        """Delete a single media row by ``(id, key)``."""
        result = self._conn.execute(
            "DELETE FROM medias WHERE id = ? AND key = ? RETURNING id", [id, key]
        ).fetchone()
        self._conn.commit()
        return result is not None

    def delete_all_keys(self, id: str) -> int:
        """Delete all rows sharing *id*.  Returns the number of rows removed."""
        rows = self._conn.execute(
            "DELETE FROM medias WHERE id = ? RETURNING id", [id]
        ).fetchall()
        self._conn.commit()
        return len(rows)
