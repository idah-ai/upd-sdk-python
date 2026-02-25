"""
_BaseRepository — shared base class for all UPD repository objects.

Design
------
Every repository inherits from this class.  Two mechanisms make query
operations work directly on the repository:

1. ``__getattr__`` delegation — any attribute not defined on the repository
   is transparently looked up on ``self.table``.  ``filter``, ``select``,
   ``order_by``, ``group_by``, ``limit``, ``aggregate``, ``execute``, …
   all work directly on the repository.

2. ``__getitem__`` — ``repo["col"]`` returns a column expression, so
   predicates can be built without a separate ``t = repo.table`` line.

3. ``__len__`` — ``len(repo)`` returns total row count as a Python int.
   For scoped counts use the chainable ``count()`` from the query API.

4. ``.table`` property — returns the raw ibis Table expression.
   **Required when passing a repository as the right-hand side of a join**,
   since ibis enforces ``isinstance(right, Table)`` at that point.

Examples
--------
    # Filter with [] — no .table needed
    df = upd.entries.filter(upd.entries.dataset_id == ds.id).execute()

    # Chain freely
    df = (
        upd.annotations
           .filter(upd.annotations.shape_type == "image:bounding-box")
           .select("id", "entry_id", "annotation")
           .order_by("id")
           .execute()
    )

    # Join — .table required on the right-hand side
    df = (
        upd.entries
           .join(upd.annotations.table, upd.entries.id == upd.annotations.entry_id)
           .execute()
    )

    # Total row count
    n = len(upd.entries)

    # Scoped count
    n = upd.entries.filter(upd.entries.dataset_id == ds.id).count().execute()
"""

from __future__ import annotations

import duckdb


class _BaseRepository:
    _table_name: str

    def __init__(self, conn: duckdb.DuckDBPyConnection, ibis_conn) -> None:
        self._conn  = conn
        self._ibis  = ibis_conn

    @property
    def table(self):
        """
        Raw ibis Table expression for this repository.

        Use this when passing the repository as the **right-hand side of a
        join** (ibis enforces a strict Table type-check there).  For all
        other query operations, use the repository directly.

            df = (
                upd.entries
                   .join(upd.annotations.table,
                         upd.entries.id == upd.annotations.entry_id)
                   .execute()
            )
        """
        return self._ibis.table(self._table_name)

    def __getitem__(self, column: str):
        """Column expression by name — use this to build predicates.

            upd.entries.filter(upd.entries.dataset_id == ds.id)
        """
        return self.table[column]

    def __len__(self) -> int:
        """Total row count as a Python int.  Use ``.count().execute()`` for scoped counts."""
        return self.table.count().execute()

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")
        return getattr(self.table, name)

    def insert(self, records: list[dict]) -> None:
        """
        Insert one or more rows via ibis and commit.

        Parameters
        ----------
        records:
            A list of dicts whose keys match the target table's columns.
            All JSON/blob fields must already be serialised before calling
            (e.g. dicts serialised via ``json_dumps``).
        """
        self._ibis.insert(self._table_name, records)
        self._conn.commit()
