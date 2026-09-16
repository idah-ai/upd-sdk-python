"""
Tests for upd.DatasetRepository (upd/dataset.py).

Run with:  pytest tests/test_dataset.py -v
"""

import pytest
from upd import UPD, Dataset


@pytest.fixture
def upd():
    with UPD.open(":memory:") as f:
        yield f


class TestDatasetCreate:
    def test_create_returns_dataset(self, upd):
        ds = upd.datasets.create(name="Test DS", modality="image")
        assert isinstance(ds, Dataset)
        assert ds.name == "Test DS"
        assert ds.modality == "image"
        assert len(ds.id) == 36   # UUID format

    def test_create_seeds_timestamps(self, upd):
        ds = upd.datasets.create(name="DS", modality="text")
        assert "Created-At" in ds.metadata
        assert "Updated-At" in ds.metadata

    def test_create_with_explicit_id(self, upd):
        ds = upd.datasets.create(name="DS", modality="image", id="my-custom-id")
        assert ds.id == "my-custom-id"

    def test_create_with_created_by(self, upd):
        ds = upd.datasets.create(name="DS", modality="m", created_by="alice@example.com")
        assert ds.metadata["Created-By"] == "alice@example.com"

    def test_create_name_too_long_raises(self, upd):
        with pytest.raises(ValueError, match="≤ 64"):
            upd.datasets.create(name="n" * 65, modality="image")

    def test_create_modality_too_long_raises(self, upd):
        with pytest.raises(ValueError, match="≤ 64"):
            upd.datasets.create(name="DS", modality="m" * 65)

    def test_duplicate_id_raises(self, upd):
        upd.datasets.create(name="DS1", modality="m", id="same-id")
        with pytest.raises(Exception):
            upd.datasets.create(name="DS2", modality="m", id="same-id")


class TestDatasetGet:
    def test_get_existing(self, upd):
        ds = upd.datasets.create(name="DS", modality="image")
        found = upd.datasets.get(ds.id)
        assert found is not None
        assert found.id == ds.id
        assert found.name == "DS"

    def test_get_missing_returns_none(self, upd):
        assert upd.datasets.get("nonexistent-id") is None


class TestDatasetAll:
    def test_all_returns_list(self, upd):
        upd.datasets.create(name="DS1", modality="m")
        upd.datasets.create(name="DS2", modality="m")
        result = upd.datasets.all()
        assert len(result) == 2
        assert all(isinstance(d, Dataset) for d in result)

    def test_all_empty_when_no_datasets(self, upd):
        assert upd.datasets.all() == []


class TestDatasetQuery:
    def test_table_has_execute(self, upd):
        assert hasattr(upd.datasets.table, "execute")

    def test_filter_predicate_direct(self, upd):
        """filter(predicate)"""
        upd.datasets.create(name="ImageDS", modality="image")
        upd.datasets.create(name="TextDS",  modality="text")
        t  = upd.datasets.table
        df = upd.datasets.filter(t.modality == "image").execute()
        assert len(df) == 1
        assert df.iloc[0]["name"] == "ImageDS"

    def test_table_chain_directly(self, upd):
        upd.datasets.create(name="ImageDS", modality="image")
        t  = upd.datasets.table
        df = t.filter(t.modality == "image").execute()
        assert len(df) == 1

    def test_select_on_repo(self, upd):
        upd.datasets.create(name="DS", modality="m")
        df = upd.datasets.select("name", "modality").execute()
        assert set(df.columns) == {"name", "modality"}

    def test_filter_select_chain(self, upd):
        upd.datasets.create(name="ImageDS", modality="image")
        t  = upd.datasets.table
        df = (
            upd.datasets
               .filter(t.modality == "image")
               .select("name")
               .execute()
        )
        assert list(df.columns) == ["name"]
        assert df.iloc[0]["name"] == "ImageDS"

    def test_order_by_on_repo(self, upd):
        upd.datasets.create(name="ZZZ", modality="m")
        upd.datasets.create(name="AAA", modality="m")
        df = upd.datasets.order_by("name").execute()
        assert df.iloc[0]["name"] == "AAA"

    def test_limit_on_repo(self, upd):
        for i in range(5):
            upd.datasets.create(name=f"DS{i}", modality="m")
        df = upd.datasets.limit(3).execute()
        assert len(df) == 3


class TestDatasetUpdate:
    def test_update_name(self, upd):
        ds = upd.datasets.create(name="Old Name", modality="m")
        updated = upd.datasets.update(ds.id, name="New Name")
        assert updated is not None
        assert updated.name == "New Name"

    def test_update_missing_returns_none(self, upd):
        assert upd.datasets.update("nonexistent-id", name="X") is None


class TestDatasetDelete:
    def test_delete_existing(self, upd):
        ds = upd.datasets.create(name="DS", modality="m")
        assert upd.datasets.delete(ds.id) is True
        assert upd.datasets.get(ds.id) is None

    def test_delete_missing_returns_false(self, upd):
        assert upd.datasets.delete("nonexistent-id") is False

    def test_delete_with_entries_raises(self, upd):
        ds = upd.datasets.create(name="DS", modality="m")
        upd.entries.create(dataset_id=ds.id, media_url="https://example.com/x")
        with pytest.raises(Exception):   # ON DELETE RESTRICT
            upd.datasets.delete(ds.id)

class TestDatasetCascadeDelete:
    """Tests for UPD.delete_dataset() — the facade-level cascade helper."""

    def _populate(self, upd, ds):
        """Create 2 entries with 2 annotations each under *ds*."""
        entries = [
            upd.entries.create(dataset_id=ds.id, media_url=f"https://example.com/{i}")
            for i in range(2)
        ]
        for e in entries:
            upd.annotations.create(entry_id=e.id, shape_type="t", shape_args={}, category="c")
            upd.annotations.create(entry_id=e.id, shape_type="t", shape_args={}, category="c")
        return entries

    def test_delete_dataset_returns_true(self, upd):
        ds = upd.datasets.create(name="DS", modality="m")
        self._populate(upd, ds)
        assert upd.delete_dataset(ds.id) is True

    def test_delete_dataset_removes_dataset_row(self, upd):
        ds = upd.datasets.create(name="DS", modality="m")
        self._populate(upd, ds)
        upd.delete_dataset(ds.id)
        assert upd.datasets.get(ds.id) is None

    def test_delete_dataset_removes_all_entries(self, upd):
        ds = upd.datasets.create(name="DS", modality="m")
        self._populate(upd, ds)
        upd.delete_dataset(ds.id)
        assert upd.entries.for_dataset(ds.id) == []

    def test_delete_dataset_removes_all_annotations(self, upd):
        ds = upd.datasets.create(name="DS", modality="m")
        self._populate(upd, ds)
        upd.delete_dataset(ds.id)
        assert upd.annotations.count_for_dataset(ds.id) == 0

    def test_delete_dataset_returns_false_when_not_found(self, upd):
        assert upd.delete_dataset("nonexistent") is False

    def test_delete_dataset_isolates_other_datasets(self, upd):
        """Rows belonging to other datasets must be untouched."""
        ds1 = upd.datasets.create(name="DS1", modality="m")
        ds2 = upd.datasets.create(name="DS2", modality="m")
        self._populate(upd, ds1)
        entries2 = self._populate(upd, ds2)

        upd.delete_dataset(ds1.id)

        assert upd.datasets.get(ds2.id) is not None
        assert len(upd.entries.for_dataset(ds2.id)) == 2
        assert upd.annotations.count_for_dataset(ds2.id) == 4
