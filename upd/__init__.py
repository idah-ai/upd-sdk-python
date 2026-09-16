"""
upd — Python library for Universal Portable Dataset (UPD) files.

Quick Start
-----------
    from upd import UPD

    with UPD.open("my_dataset.upd") as upd:
        ds = upd.datasets.create(name="My Images", modality="image")

        media = upd.medias.create_from_file("photo.jpg")
        entry = upd.entries.create(dataset_id=ds.id, media_url=media.local_url)

        upd.annotations.create(
            entry_id=entry.id,
            shape_type="bounding-box",
            shape_args={"x": 10, "y": 20, "width": 100, "height": 80},
            category="cat",
            properties={"confidence": 0.95},
        )

        # Repositories are first-class query objects
        df = upd.entries.filter(upd.entries.dataset_id == ds.id).execute()

        # Row count
        print(len(upd.entries))
"""

from ._connection import UPD
from ._base import _BaseRepository
from .metadata import MetadataRepository
from .dataset import DatasetRepository, Dataset
from .media import MediaRepository, Media
from .entry import EntryRepository, Entry
from .annotation import AnnotationRepository, Annotation

__all__ = [
    "UPD",
    "_BaseRepository",
    "MetadataRepository",
    "DatasetRepository",
    "MediaRepository",
    "EntryRepository",
    "AnnotationRepository",
    "Dataset",
    "Media",
    "Entry",
    "Annotation",
]

__version__ = "0.1.0"
