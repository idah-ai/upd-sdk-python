"""
Tests for upd.MetadataRepository (upd/metadata.py).

Run with:  pytest tests/test_metadata.py -v
"""

import pytest
from upd import UPD


@pytest.fixture
def upd():
    with UPD.open(":memory:") as f:
        yield f


class TestMetadataGet:
    def test_required_keys_are_seeded(self, upd):
        assert upd.metadata.get("Schema-Type")    == "Universal Portable Dataset"
        assert upd.metadata.get("Schema-Version") == "1.0"
        assert upd.metadata.get("Schema-Flavor")  == "Vanilla"

    def test_get_missing_key_returns_none(self, upd):
        assert upd.metadata.get("nonexistent-key") is None

    def test_get_returns_decoded_number(self, upd):
        upd.metadata.set("My-Number", 42)
        assert upd.metadata.get("My-Number") == 42

    def test_get_returns_decoded_dict(self, upd):
        upd.metadata.set("My-Object", {"a": 1, "b": [1, 2, 3]})
        assert upd.metadata.get("My-Object") == {"a": 1, "b": [1, 2, 3]}


class TestMetadataSet:
    def test_set_new_key(self, upd):
        upd.metadata.set("Authored-By", "alice@example.com")
        assert upd.metadata.get("Authored-By") == "alice@example.com"

    def test_set_updates_existing_key(self, upd):
        upd.metadata.set("Schema-Built-By", "tool v1")
        upd.metadata.set("Schema-Built-By", "tool v2")
        assert upd.metadata.get("Schema-Built-By") == "tool v2"

    def test_set_key_too_long_raises(self, upd):
        with pytest.raises(ValueError, match="≤ 64"):
            upd.metadata.set("k" * 65, "value")

    def test_set_various_value_types(self, upd):
        upd.metadata.set("Bool-Val",  True)
        upd.metadata.set("List-Val",  [1, 2, 3])
        upd.metadata.set("None-Val",  None)
        assert upd.metadata.get("Bool-Val") is True
        assert upd.metadata.get("List-Val") == [1, 2, 3]
        assert upd.metadata.get("None-Val") is None


class TestMetadataAll:
    def test_all_returns_dict(self, upd):
        result = upd.metadata.all()
        assert isinstance(result, dict)
        assert "Schema-Type" in result

    def test_all_includes_custom_keys(self, upd):
        upd.metadata.set("Custom-Key", "hello")
        assert upd.metadata.all()["Custom-Key"] == "hello"


class TestMetadataDelete:
    def test_delete_existing_key(self, upd):
        upd.metadata.set("Temp-Key", "temp")
        assert upd.metadata.delete("Temp-Key") is True
        assert upd.metadata.get("Temp-Key") is None

    def test_delete_nonexistent_returns_false(self, upd):
        assert upd.metadata.delete("does-not-exist") is False


class TestMetadataQuery:
    def test_table_is_a_query_expression(self, upd):
        # table property returns something with an execute() method
        t = upd.metadata.table
        assert hasattr(t, "execute")
        assert hasattr(t, "filter")

    def test_filter_with_predicate_on_table(self, upd):
        """filter(predicate) works directly — no double-call needed."""
        upd.metadata.set("Schema-Built-By", "pytest")
        t  = upd.metadata.table
        df = upd.metadata.filter(t.key.startswith("Schema")).execute()
        assert len(df) >= 3

    def test_table_filter_chaining(self, upd):
        """Start from .table and chain freely."""
        t  = upd.metadata.table
        df = t.filter(t.key == "Schema-Type").execute()
        assert len(df) == 1
        assert df.iloc[0]["key"] == "Schema-Type"

    def test_select_directly_on_repo(self, upd):
        """select() is delegated transparently — no .table needed."""
        df = upd.metadata.select("key").execute()
        assert "key" in df.columns
        assert "value" not in df.columns

    def test_filter_then_select_chain(self, upd):
        t  = upd.metadata.table
        df = upd.metadata.filter(t.key == "Schema-Type").select("value").execute()
        assert list(df.columns) == ["value"]
