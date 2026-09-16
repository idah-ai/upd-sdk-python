"""
Tests for upd.AnnotationRepository (upd/annotation.py).

Run with:  pytest tests/test_annotation.py -v
"""

import pytest
from upd import UPD, Annotation


@pytest.fixture
def upd():
    with UPD.open(":memory:") as f:
        yield f


@pytest.fixture
def ds(upd):
    return upd.datasets.create(name="Test DS", modality="image")


@pytest.fixture
def entry(upd, ds):
    return upd.entries.create(dataset_id=ds.id, media_url="https://example.com/img.jpg")


BBOX       = {"x": 10, "y": 20, "width": 100, "height": 80}
CATEGORY   = "cat"
PROPERTIES = {"confidence": 0.95}


class TestAnnotationCreate:
    def test_create_returns_annotation(self, upd, entry):
        a = upd.annotations.create(
            entry_id=entry.id,
            shape_type="image:bounding-box",
            shape_args=BBOX,
            category=CATEGORY,
            properties=PROPERTIES,
        )
        assert isinstance(a, Annotation)
        assert a.shape_type == "image:bounding-box"
        assert a.shape_args == BBOX
        assert a.category == CATEGORY
        assert a.properties == PROPERTIES

    def test_create_seeds_timestamps(self, upd, entry):
        a = upd.annotations.create(entry_id=entry.id, shape_type="t", shape_args={}, category="c")
        assert "Created-At" in a.metadata

    def test_create_with_qc_status(self, upd, entry):
        a = upd.annotations.create(
            entry_id=entry.id, shape_type="t", shape_args={}, category="c", qc_status="Passed"
        )
        assert a.metadata["QC-Status"] == "Passed"

    def test_create_shape_type_too_long_raises(self, upd, entry):
        with pytest.raises(ValueError, match="≤ 64"):
            upd.annotations.create(entry_id=entry.id, shape_type="s" * 65,
                                   shape_args={}, category="c")

    def test_create_invalid_entry_id_raises(self, upd):
        with pytest.raises(Exception):   # FK constraint
            upd.annotations.create(entry_id="nonexistent", shape_type="t",
                                   shape_args={}, category="c")

    def test_create_missing_category_raises(self, upd, entry):
        # category is a required argument — omitting it fails fast in Python
        # rather than surfacing as a DuckDB NOT NULL violation.
        with pytest.raises(TypeError):
            upd.annotations.create(entry_id=entry.id, shape_type="t", shape_args={})

    def test_create_blank_category_raises(self, upd, entry):
        with pytest.raises(ValueError, match="category"):
            upd.annotations.create(entry_id=entry.id, shape_type="t", shape_args={}, category="")


class TestAnnotationBulkCreate:
    def test_bulk_create(self, upd, entry):
        rows = [
            {"entry_id": entry.id, "shape_type": "image:bounding-box",
             "shape_args": {"x": i}, "category": "c", "properties": {"id": i}}
            for i in range(5)
        ]
        anns = upd.annotations.bulk_create(rows)
        assert len(anns) == 5

    def test_bulk_create_missing_category_raises(self, upd, entry):
        rows = [
            {"entry_id": entry.id, "shape_type": "t", "shape_args": {}, "category": "c"},
            {"entry_id": entry.id, "shape_type": "t", "shape_args": {}},
        ]
        with pytest.raises(ValueError, match="category"):
            upd.annotations.bulk_create(rows)


class TestAnnotationGet:
    def test_get_existing(self, upd, entry):
        a = upd.annotations.create(entry_id=entry.id, shape_type="t", shape_args={}, category="c")
        assert upd.annotations.get(a.id) is not None

    def test_get_missing_returns_none(self, upd):
        assert upd.annotations.get("nonexistent") is None


class TestAnnotationForEntry:
    def test_for_entry(self, upd, entry):
        upd.annotations.create(entry_id=entry.id, shape_type="t", shape_args={}, category="c", properties={"n": 1})
        upd.annotations.create(entry_id=entry.id, shape_type="t", shape_args={}, category="c", properties={"n": 2})
        assert len(upd.annotations.for_entry(entry.id)) == 2

    def test_for_entry_empty(self, upd, entry):
        assert upd.annotations.for_entry(entry.id) == []


class TestAnnotationForDataset:
    def test_for_dataset_spans_entries(self, upd, ds):
        e1 = upd.entries.create(dataset_id=ds.id, media_url="https://a.com")
        e2 = upd.entries.create(dataset_id=ds.id, media_url="https://b.com")
        upd.annotations.create(entry_id=e1.id, shape_type="t", shape_args={}, category="c")
        upd.annotations.create(entry_id=e2.id, shape_type="t", shape_args={}, category="c")
        assert len(upd.annotations.for_dataset(ds.id)) == 2


class TestAnnotationCount:
    def test_len_total(self, upd, entry):
        upd.annotations.create(entry_id=entry.id, shape_type="t", shape_args={}, category="c")
        upd.annotations.create(entry_id=entry.id, shape_type="t", shape_args={}, category="c")
        assert len(upd.annotations.all()) == 2

    def test_scoped_count_via_ibis(self, upd, ds):
        e1 = upd.entries.create(dataset_id=ds.id, media_url="https://a.com")
        e2 = upd.entries.create(dataset_id=ds.id, media_url="https://b.com")
        upd.annotations.create(entry_id=e1.id, shape_type="t", shape_args={}, category="c")
        upd.annotations.create(entry_id=e1.id, shape_type="t", shape_args={}, category="c")
        upd.annotations.create(entry_id=e2.id, shape_type="t", shape_args={}, category="c")
        n1 = upd.annotations.filter(upd.annotations.entry_id == e1.id).count().execute()
        n2 = upd.annotations.filter(upd.annotations.entry_id == e2.id).count().execute()
        assert n1 == 2
        assert n2 == 1


class TestAnnotationQuery:
    def test_table_has_execute(self, upd):
        assert hasattr(upd.annotations.table, "execute")

    def test_filter_with_bracket_notation(self, upd, entry):
        """Predicates built with repo["col"] — no .table needed."""
        upd.annotations.create(entry_id=entry.id, shape_type="image:bounding-box",
                               shape_args={}, category="c")
        upd.annotations.create(entry_id=entry.id, shape_type="image:polygon",
                               shape_args={}, category="c")
        df = upd.annotations.filter(upd.annotations["shape_type"] == "image:bounding-box").execute()
        assert len(df) == 1


    def test_filter_with_table_notation(self, upd, entry):
        """.table still works as an explicit expression handle."""
        upd.annotations.create(entry_id=entry.id, shape_type="image:bounding-box",
                               shape_args={}, category="c")
        t  = upd.annotations.table
        df = t.filter(t.shape_type == "image:bounding-box").execute()
        assert len(df) == 1

    def test_select_on_repo(self, upd, entry):
        upd.annotations.create(entry_id=entry.id, shape_type="t", shape_args={}, category="c")
        df = upd.annotations.select("id", "shape_type").execute()
        assert set(df.columns) == {"id", "shape_type"}

    def test_filter_select_chain(self, upd, entry):
        upd.annotations.create(entry_id=entry.id, shape_type="image:bounding-box",
                               shape_args={}, category="c")
        df = (
            upd.annotations
               .filter(upd.annotations.shape_type == "image:bounding-box")
               .select("id", "entry_id")
               .execute()
        )
        assert set(df.columns) == {"id", "entry_id"}

    def test_order_by_on_repo(self, upd, entry):
        upd.annotations.create(entry_id=entry.id, shape_type="zzz", shape_args={}, category="c")
        upd.annotations.create(entry_id=entry.id, shape_type="aaa", shape_args={}, category="c")
        df = upd.annotations.order_by("shape_type").execute()
        assert df.iloc[0]["shape_type"] == "aaa"

    def test_limit_on_repo(self, upd, entry):
        for _ in range(5):
            upd.annotations.create(entry_id=entry.id, shape_type="t", shape_args={}, category="c")
        df = upd.annotations.limit(3).execute()
        assert len(df) == 3

    def test_group_by_aggregation(self, upd, entry):
        upd.annotations.create(entry_id=entry.id, shape_type="bbox", shape_args={}, category="c")
        upd.annotations.create(entry_id=entry.id, shape_type="bbox", shape_args={}, category="c")
        upd.annotations.create(entry_id=entry.id, shape_type="poly", shape_args={}, category="c")
        df = (
            upd.annotations
               .group_by("shape_type")
               .aggregate(n=upd.annotations.id.count())
               .execute()
        )
        assert len(df) == 2
        counts = dict(zip(df["shape_type"], df["n"]))
        assert counts["bbox"] == 2
        assert counts["poly"] == 1

    def test_join_requires_table_on_rhs(self, upd, ds, entry):
        """.table is required on the right-hand side of a join (ibis type check)."""
        upd.annotations.create(entry_id=entry.id, shape_type="bbox", shape_args={}, category="c")
        df = (
            upd.entries
               .join(upd.annotations.table, upd.entries.id == upd.annotations.entry_id)
               .execute()
        )
        assert len(df) >= 1

    def test_ibis_count_scoped(self, upd, entry):
        """ibis count() (chainable scalar) works correctly."""
        upd.annotations.create(entry_id=entry.id, shape_type="bbox", shape_args={}, category="c")
        upd.annotations.create(entry_id=entry.id, shape_type="bbox", shape_args={}, category="c")
        n = upd.annotations.filter(upd.annotations.entry_id == entry.id).count().execute()
        assert n == 2


class TestAnnotationUpdate:
    def test_update_properties(self, upd, entry):
        a = upd.annotations.create(entry_id=entry.id, shape_type="t",
                                   shape_args={}, category="cat", properties={"confidence": 0.9})
        updated = upd.annotations.update(a.id, properties={"confidence": 0.95})
        assert updated.properties == {"confidence": 0.95}
        assert updated.category == "cat"

    def test_update_category(self, upd, entry):
        a = upd.annotations.create(entry_id=entry.id, shape_type="t",
                                   shape_args={}, category="cat")
        updated = upd.annotations.update(a.id, category="dog")
        assert updated.category == "dog"

    def test_update_category_none_keeps_existing(self, upd, entry):
        a = upd.annotations.create(entry_id=entry.id, shape_type="t",
                                   shape_args={}, category="cat")
        updated = upd.annotations.update(a.id, category=None)
        assert updated.category == "cat"

    def test_update_qc_status(self, upd, entry):
        a = upd.annotations.create(entry_id=entry.id, shape_type="t", shape_args={}, category="c")
        updated = upd.annotations.update(a.id, qc_status="Rejected")
        assert updated.metadata["QC-Status"] == "Rejected"

    def test_update_missing_returns_none(self, upd):
        assert upd.annotations.update("nonexistent") is None


class TestAnnotationDelete:
    def test_delete_existing(self, upd, entry):
        a = upd.annotations.create(entry_id=entry.id, shape_type="t", shape_args={}, category="c")
        assert upd.annotations.delete(a.id) is True
        assert upd.annotations.get(a.id) is None

    def test_delete_for_entry(self, upd, entry):
        upd.annotations.create(entry_id=entry.id, shape_type="t", shape_args={}, category="c")
        upd.annotations.create(entry_id=entry.id, shape_type="t", shape_args={}, category="c")
        assert upd.annotations.delete_for_entry(entry.id) == 2

    def test_delete_missing_returns_false(self, upd):
        assert upd.annotations.delete("nonexistent") is False

class TestAnnotationCountForDataset:
    def test_count_for_dataset(self, upd, ds):
        e1 = upd.entries.create(dataset_id=ds.id, media_url="https://a.com")
        e2 = upd.entries.create(dataset_id=ds.id, media_url="https://b.com")
        upd.annotations.create(entry_id=e1.id, shape_type="t", shape_args={}, category="c")
        upd.annotations.create(entry_id=e1.id, shape_type="t", shape_args={}, category="c")
        upd.annotations.create(entry_id=e2.id, shape_type="t", shape_args={}, category="c")
        assert upd.annotations.count_for_dataset(ds.id) == 3

    def test_count_for_dataset_empty(self, upd, ds):
        assert upd.annotations.count_for_dataset(ds.id) == 0

    def test_count_for_dataset_isolates(self, upd):
        ds1 = upd.datasets.create(name="DS1", modality="m")
        ds2 = upd.datasets.create(name="DS2", modality="m")
        e1  = upd.entries.create(dataset_id=ds1.id, media_url="https://a.com")
        e2  = upd.entries.create(dataset_id=ds2.id, media_url="https://b.com")
        upd.annotations.create(entry_id=e1.id, shape_type="t", shape_args={}, category="c")
        upd.annotations.create(entry_id=e2.id, shape_type="t", shape_args={}, category="c")
        upd.annotations.create(entry_id=e2.id, shape_type="t", shape_args={}, category="c")
        assert upd.annotations.count_for_dataset(ds1.id) == 1
        assert upd.annotations.count_for_dataset(ds2.id) == 2


class TestAnnotationDeleteForDataset:
    def test_delete_for_dataset_removes_all(self, upd, ds):
        e1 = upd.entries.create(dataset_id=ds.id, media_url="https://a.com")
        e2 = upd.entries.create(dataset_id=ds.id, media_url="https://b.com")
        upd.annotations.create(entry_id=e1.id, shape_type="t", shape_args={}, category="c")
        upd.annotations.create(entry_id=e1.id, shape_type="t", shape_args={}, category="c")
        upd.annotations.create(entry_id=e2.id, shape_type="t", shape_args={}, category="c")
        assert upd.annotations.delete_for_dataset(ds.id) == 3
        assert upd.annotations.count_for_dataset(ds.id) == 0

    def test_delete_for_dataset_returns_zero_when_empty(self, upd, ds):
        assert upd.annotations.delete_for_dataset(ds.id) == 0

    def test_delete_for_dataset_isolates(self, upd):
        """Annotations from other datasets must not be affected."""
        ds1 = upd.datasets.create(name="DS1", modality="m")
        ds2 = upd.datasets.create(name="DS2", modality="m")
        e1  = upd.entries.create(dataset_id=ds1.id, media_url="https://a.com")
        e2  = upd.entries.create(dataset_id=ds2.id, media_url="https://b.com")
        upd.annotations.create(entry_id=e1.id, shape_type="t", shape_args={}, category="c")
        upd.annotations.create(entry_id=e2.id, shape_type="t", shape_args={}, category="c")
        upd.annotations.delete_for_dataset(ds1.id)
        assert upd.annotations.count_for_dataset(ds1.id) == 0
        assert upd.annotations.count_for_dataset(ds2.id) == 1
