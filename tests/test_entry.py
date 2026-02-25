"""
Tests for upd.EntryRepository (upd/entry.py).

Run with:  pytest tests/test_entry.py -v
"""

import pytest
from upd import UPD, Entry


@pytest.fixture
def upd():
    with UPD.open(":memory:") as f:
        yield f


@pytest.fixture
def ds(upd):
    return upd.datasets.create(name="Test DS", modality="image")


@pytest.fixture
def media(upd):
    return upd.medias.create(blob_data=b"fake-img", media_type="image/jpeg")


class TestEntryCreate:
    def test_create_with_local_url(self, upd, ds, media):
        e = upd.entries.create(dataset_id=ds.id, media_url=f"local:{media.id}")
        assert isinstance(e, Entry)
        assert e.is_local is True
        assert e.local_media_id == media.id

    def test_create_with_external_url(self, upd, ds):
        e = upd.entries.create(dataset_id=ds.id, media_url="https://example.com/img.jpg")
        assert e.is_local is False
        assert e.local_media_id is None

    def test_create_seeds_timestamps(self, upd, ds):
        e = upd.entries.create(dataset_id=ds.id, media_url="https://example.com/x")
        assert "Created-At" in e.metadata
        assert "Updated-At" in e.metadata

    def test_create_with_created_by(self, upd, ds):
        e = upd.entries.create(dataset_id=ds.id, media_url="s3://bucket/img",
                               created_by="bob@example.com")
        assert e.metadata["Created-By"] == "bob@example.com"

    def test_create_invalid_dataset_id_raises(self, upd):
        with pytest.raises(Exception):   # FK constraint
            upd.entries.create(dataset_id="nonexistent-ds", media_url="https://x.com")


class TestEntryBulkCreate:
    def test_bulk_create(self, upd, ds):
        rows = [{"dataset_id": ds.id, "media_url": f"https://example.com/{i}"} for i in range(5)]
        entries = upd.entries.bulk_create(rows)
        assert len(entries) == 5
        assert all(isinstance(e, Entry) for e in entries)


class TestEntryGet:
    def test_get_existing(self, upd, ds):
        e = upd.entries.create(dataset_id=ds.id, media_url="https://x.com")
        assert upd.entries.get(e.id) is not None

    def test_get_missing_returns_none(self, upd):
        assert upd.entries.get("nonexistent") is None


class TestEntryForDataset:
    def test_for_dataset(self, upd, ds):
        upd.entries.create(dataset_id=ds.id, media_url="https://a.com")
        upd.entries.create(dataset_id=ds.id, media_url="https://b.com")
        assert len(upd.entries.for_dataset(ds.id)) == 2

    def test_for_dataset_isolates(self, upd):
        ds1 = upd.datasets.create(name="DS1", modality="m")
        ds2 = upd.datasets.create(name="DS2", modality="m")
        upd.entries.create(dataset_id=ds1.id, media_url="https://a.com")
        upd.entries.create(dataset_id=ds2.id, media_url="https://b.com")
        assert len(upd.entries.for_dataset(ds1.id)) == 1
        assert len(upd.entries.for_dataset(ds2.id)) == 1


class TestEntryCount:
    def test_len_total(self, upd, ds):
        upd.entries.create(dataset_id=ds.id, media_url="https://a.com")
        upd.entries.create(dataset_id=ds.id, media_url="https://b.com")
        assert len(upd.entries.all()) == 2

    def test_scoped_count_via_ibis(self, upd):
        ds1 = upd.datasets.create(name="DS1", modality="m")
        ds2 = upd.datasets.create(name="DS2", modality="m")
        upd.entries.create(dataset_id=ds1.id, media_url="https://a.com")
        upd.entries.create(dataset_id=ds1.id, media_url="https://b.com")
        upd.entries.create(dataset_id=ds2.id, media_url="https://c.com")
        n1 = upd.entries.filter(upd.entries.dataset_id == ds1.id).count().execute()
        n2 = upd.entries.filter(upd.entries.dataset_id == ds2.id).count().execute()
        assert n1 == 2
        assert n2 == 1


class TestEntryQuery:
    def test_table_has_execute(self, upd):
        assert hasattr(upd.entries.table, "execute")

    def test_filter_predicate_direct(self, upd, ds):
        """filter(predicate)"""
        upd.entries.create(dataset_id=ds.id, media_url="https://a.com")
        df = upd.entries.filter(upd.entries.dataset_id == ds.id).execute()
        assert len(df) == 1

    def test_table_chain_directly(self, upd, ds):
        upd.entries.create(dataset_id=ds.id, media_url="https://a.com")
        df = upd.entries.table.filter(upd.entries.dataset_id == ds.id).execute()
        assert len(df) == 1

    def test_select_on_repo(self, upd, ds):
        upd.entries.create(dataset_id=ds.id, media_url="https://a.com")
        df = upd.entries.select("id", "media_url").execute()
        assert set(df.columns) == {"id", "media_url"}

    def test_filter_select_chain(self, upd, ds):
        upd.entries.create(dataset_id=ds.id, media_url="https://a.com")
        df = (
            upd.entries
               .filter(upd.entries.dataset_id == ds.id)
               .select("id", "media_url")
               .execute()
        )
        assert "id" in df.columns
        assert "media_url" in df.columns

    def test_limit_on_repo(self, upd, ds):
        for i in range(5):
            upd.entries.create(dataset_id=ds.id, media_url=f"https://example.com/{i}")
        df = upd.entries.limit(2).execute()
        assert len(df) == 2


class TestEntryUpdate:
    def test_update_media_url(self, upd, ds):
        e = upd.entries.create(dataset_id=ds.id, media_url="https://old.com")
        updated = upd.entries.update(e.id, media_url="https://new.com")
        assert updated.media_url == "https://new.com"

    def test_update_missing_returns_none(self, upd):
        assert upd.entries.update("nonexistent") is None


class TestEntryDelete:
    def test_delete_existing(self, upd, ds):
        e = upd.entries.create(dataset_id=ds.id, media_url="https://x.com")
        assert upd.entries.delete(e.id) is True
        assert upd.entries.get(e.id) is None

    def test_delete_with_annotations_raises(self, upd, ds):
        e = upd.entries.create(dataset_id=ds.id, media_url="https://x.com")
        upd.annotations.create(entry_id=e.id, shape_type="t", shape_args={}, annotation={})
        with pytest.raises(Exception):   # ON DELETE RESTRICT
            upd.entries.delete(e.id)

    def test_delete_missing_returns_false(self, upd):
        assert upd.entries.delete("nonexistent") is False
