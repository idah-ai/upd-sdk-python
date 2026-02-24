"""
UPD — central facade for opening and managing UPD files.

Usage
-----
    from upd import UPD

    with UPD.open("my_dataset.upd") as upd:
        ds    = upd.datasets.create(name="MNIST", modality="mnist-image")
        media = upd.medias.create_from_file("img.png")
        entry = upd.entries.create(dataset_id=ds.id, media_url=media.local_url)

        # Repositories are first-class query objects
        df = upd.entries.filter(upd.entries.dataset_id == ds.id).execute()

        # Joins without .table — to_expr() handles coercion
        df = (
            upd.entries
               .join(upd.annotations, upd.entries.id == upd.annotations.entry_id)
               .execute()
        )
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    import duckdb
    import ibis
    import ibis.backends.duckdb  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "upd requires 'duckdb' and 'ibis-framework[duckdb]'.\n"
        "Install with:  pip install duckdb ibis-framework[duckdb]"
    ) from exc

from ._schema import INITIALIZATION_STATEMENTS, DEFAULT_METADATA
from .metadata import MetadataRepository
from .dataset import DatasetRepository
from .media import MediaRepository
from .entry import EntryRepository
from .annotation import AnnotationRepository


class UPD:
    """
    A handle to an open UPD (Universal Portable Dataset) file.

    Parameters
    ----------
    path:
        Filesystem path to the ``.upd`` / ``.duckdb`` file.
        Pass ``":memory:"`` for an ephemeral in-memory database.
    read_only:
        Open in read-only mode (default: ``False``).

    Attributes
    ----------
    metadata    : MetadataRepository
    datasets    : DatasetRepository
    medias      : MediaRepository
    entries     : EntryRepository
    annotations : AnnotationRepository
    """

    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self._path      = str(path)
        self._read_only = read_only
        self._conn      = duckdb.connect(self._path, read_only=read_only)
        self._qb        = ibis.duckdb.from_connection(self._conn)

        if not read_only:
            self._initialize_schema()

        self.metadata    = MetadataRepository(self._conn, self._qb)
        self.datasets    = DatasetRepository(self._conn, self._qb)
        self.medias      = MediaRepository(self._conn, self._qb)
        self.entries     = EntryRepository(self._conn, self._qb)
        self.annotations = AnnotationRepository(self._conn, self._qb)

    # ------------------------------------------------------------------
    # Context-manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "UPD":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def open(cls, path: str | Path, *, read_only: bool = False) -> "UPD":
        """
        Open (or create) a UPD file.

        Parameters
        ----------
        path:
            File path.  Use ``":memory:"`` for an ephemeral database.
        read_only:
            If ``True``, open for reading only.
        """
        return cls(path, read_only=read_only)

    def close(self) -> None:
        """Flush and close the underlying database connection."""
        try:
            self._conn.close()
        except Exception:
            pass

    @property
    def path(self) -> str:
        """Filesystem path of the underlying DuckDB file."""
        return self._path

    @property
    def raw_connection(self) -> duckdb.DuckDBPyConnection:
        """
        Direct access to the raw DuckDB connection.

        Use for advanced SQL, JSON extraction, or Flavor-specific queries.
        """
        return self._conn

    def execute(self, sql: str, params: Optional[list] = None):
        """Execute arbitrary SQL against the UPD file."""
        return self._conn.execute(sql, params or [])

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _initialize_schema(self) -> None:
        for stmt in INITIALIZATION_STATEMENTS:
            self._conn.execute(stmt)
        for key, value in DEFAULT_METADATA.items():
            self._conn.execute(
                "INSERT INTO metadata (key, value) VALUES (?, ?) "
                "ON CONFLICT (key) DO NOTHING",
                [key, value],
            )
        self._conn.commit()

    def __repr__(self) -> str:
        mode = "r" if self._read_only else "r/w"
        return f"UPD({self._path!r}, mode={mode!r})"
