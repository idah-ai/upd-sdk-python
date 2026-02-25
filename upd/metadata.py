"""
MetadataRepository — CRUD operations for the ``metadata`` table.

The ``metadata`` table holds file-level key/value configuration pairs as
defined in UPD RFC §3.2.  Values are stored as JSON-encoded VARCHAR strings.

Because this class inherits from ``_BaseRepository``, every query operation
works directly on the repository object — no intermediate call required:

    upd.metadata.set("Authored-By", "alice@example.com")

    # Read back
    value = upd.metadata.get("Schema-Version")   # "0.2"

    # Query — predicates built from .table, result is chainable
    t  = upd.metadata.table
    df = upd.metadata.filter(t.key.startswith("Schema")).execute()

    # Or go straight from the table expression
    df = upd.metadata.table.filter(t.key == "Schema-Type").execute()

    # Any other operation works too
    df = upd.metadata.select("key").execute()
"""

from __future__ import annotations

from typing import Any, Optional

import duckdb

from ._base import _BaseRepository
from ._utils import json_dumps, json_loads, validate_id


class MetadataRepository(_BaseRepository):
    """
    Read/write access to the global ``metadata`` table.

    Parameters
    ----------
    conn:
        Raw DuckDB connection (used for writes).
    ibis_conn:
        Query backend connection (used for reads).
    """

    _table_name = "metadata"

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve the decoded value for *key*, or ``None`` if absent.

        Parameters
        ----------
        key:
            The metadata key (≤ 64 characters).

        Returns
        -------
        Any
            Decoded Python object, or ``None``.
        """
        t  = self.table
        df = t.filter(t.key == key).execute()
        if df.empty:
            return None
        return json_loads(df.iloc[0]["value"])

    def all(self) -> dict[str, Any]:
        """
        Return all metadata as a ``{key: decoded_value}`` dictionary.

        Returns
        -------
        dict[str, Any]
        """
        records = self.table.execute().to_dict("records")
        return {r["key"]: json_loads(r["value"]) for r in records}

    # ------------------------------------------------------------------
    # Write  (upsert / delete — no ibis equivalent; raw SQL retained)
    # ------------------------------------------------------------------

    def set(self, key: str, value: Any) -> None:
        """
        Upsert a metadata key/value pair.

        Parameters
        ----------
        key:
            Metadata key (≤ 64 characters).
        value:
            Any JSON-serialisable Python object.
        """
        validate_id(key, field="metadata key")
        # ON CONFLICT upsert has no ibis equivalent — raw SQL required.
        self._conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?, ?) "
            "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            [key, json_dumps(value)],
        )
        self._conn.commit()

    def delete(self, key: str) -> bool:
        """
        Remove the entry for *key*.

        Returns
        -------
        bool
            ``True`` if a row was deleted.
        """
        # RETURNING clause not available via ibis — raw SQL required.
        result = self._conn.execute(
            "DELETE FROM metadata WHERE key = ? RETURNING key", [key]
        ).fetchone()
        self._conn.commit()
        return result is not None
