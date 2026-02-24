"""
Tests for upd.MediaRepository (upd/media.py).

Run with:  pytest tests/test_media.py -v
"""

import pytest
from upd import UPD, Media


@pytest.fixture
def upd():
    with UPD.open(":memory:") as f:
        yield f


SAMPLE_BYTES = b"\x89PNG\r\n\x1a\nfake-png-content"


class TestMediaCreate:
    def test_create_single_file(self, upd):
        m = upd.medias.create(blob_data=SAMPLE_BYTES, media_type="image/png")
        assert isinstance(m, Media)
        assert m.key == ""
        assert m.blob_data == SAMPLE_BYTES
        assert m.media_type == "image/png"

    def test_create_with_explicit_id(self, upd):
        m = upd.medias.create(id="m-001", blob_data=b"data", media_type="image/jpeg")
        assert m.id == "m-001"

    def test_create_null_blob_allowed(self, upd):
        m = upd.medias.create(media_type="image/jpeg")
        assert m.blob_data is None

    def test_create_composite_keys(self, upd):
        m1 = upd.medias.create(id="map-001", key="00,00.png", blob_data=b"t1", media_type="image/png")
        m2 = upd.medias.create(id="map-001", key="00,01.png", blob_data=b"t2", media_type="image/png")
        assert m1.id == m2.id == "map-001"
        assert m1.key != m2.key

    def test_create_id_too_long_raises(self, upd):
        with pytest.raises(ValueError, match="≤ 64"):
            upd.medias.create(id="i" * 65, blob_data=b"x")

    def test_create_key_too_long_raises(self, upd):
        with pytest.raises(ValueError, match="≤ 256"):
            upd.medias.create(key="k" * 257, blob_data=b"x")

    def test_local_url_property(self, upd):
        m = upd.medias.create(id="m-001", blob_data=b"data")
        assert m.local_url == "local:m-001"


class TestMediaCreateFromFile:
    def test_create_from_file(self, upd, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(SAMPLE_BYTES)
        m = upd.medias.create_from_file(str(img))
        assert m.blob_data == SAMPLE_BYTES
        assert m.media_type == "image/png"

    def test_create_from_file_explicit_mime(self, upd, tmp_path):
        f = tmp_path / "data.bin"
        f.write_bytes(b"\x00\x01\x02")
        m = upd.medias.create_from_file(str(f), media_type="application/octet-stream")
        assert m.media_type == "application/octet-stream"


class TestMediaGet:
    def test_get_existing(self, upd):
        upd.medias.create(id="m-001", blob_data=SAMPLE_BYTES, media_type="image/png")
        found = upd.medias.get("m-001")
        assert found is not None
        assert found.blob_data == SAMPLE_BYTES

    def test_get_missing_returns_none(self, upd):
        assert upd.medias.get("nonexistent") is None

    def test_get_composite_key(self, upd):
        upd.medias.create(id="map-001", key="tile-a", blob_data=b"A", media_type="image/png")
        upd.medias.create(id="map-001", key="tile-b", blob_data=b"B", media_type="image/png")
        m = upd.medias.get("map-001", "tile-b")
        assert m is not None
        assert m.blob_data == b"B"

    def test_get_all_keys(self, upd):
        upd.medias.create(id="map-001", key="tile-a", blob_data=b"A", media_type="image/png")
        upd.medias.create(id="map-001", key="tile-b", blob_data=b"B", media_type="image/png")
        medias = upd.medias.get_all_keys("map-001")
        assert len(medias) == 2
        assert {m.key for m in medias} == {"tile-a", "tile-b"}


class TestMediaQuery:
    def test_table_excludes_blob_data(self, upd):
        """blob_data is excluded from query expressions for performance."""
        t = upd.medias.table
        assert "blob_data" not in t.columns

    def test_filter_predicate_direct(self, upd):
        """filter(predicate) — no double-call."""
        upd.medias.create(blob_data=b"img",   media_type="image/png")
        upd.medias.create(blob_data=b"audio", media_type="audio/wav")
        t  = upd.medias.table
        df = upd.medias.filter(t.media_type == "image/png").execute()
        assert len(df) == 1

    def test_table_chain_directly(self, upd):
        upd.medias.create(blob_data=b"img", media_type="image/png")
        t  = upd.medias.table
        df = t.filter(t.media_type == "image/png").execute()
        assert len(df) == 1

    def test_select_on_repo(self, upd):
        upd.medias.create(blob_data=b"x", media_type="image/png")
        df = upd.medias.select("id", "media_type").execute()
        assert set(df.columns) == {"id", "media_type"}

    def test_filter_select_chain(self, upd):
        upd.medias.create(id="m-001", blob_data=b"x", media_type="image/png")
        t  = upd.medias.table
        df = (
            upd.medias
               .filter(t.media_type == "image/png")
               .select("id")
               .execute()
        )
        assert list(df.columns) == ["id"]
        assert df.iloc[0]["id"] == "m-001"


class TestMediaUpdate:
    def test_update_media_type(self, upd):
        upd.medias.create(id="m-001", blob_data=b"data", media_type="image/jpeg")
        updated = upd.medias.update("m-001", media_type="image/png")
        assert updated.media_type == "image/png"

    def test_update_missing_returns_none(self, upd):
        assert upd.medias.update("nonexistent") is None


class TestMediaDelete:
    def test_delete_single_file(self, upd):
        upd.medias.create(id="m-001", blob_data=b"x")
        assert upd.medias.delete("m-001") is True
        assert upd.medias.get("m-001") is None

    def test_delete_one_key_of_composite(self, upd):
        upd.medias.create(id="map-001", key="tile-a", blob_data=b"A")
        upd.medias.create(id="map-001", key="tile-b", blob_data=b"B")
        upd.medias.delete("map-001", "tile-a")
        assert upd.medias.get("map-001", "tile-a") is None
        assert upd.medias.get("map-001", "tile-b") is not None

    def test_delete_all_keys(self, upd):
        upd.medias.create(id="map-001", key="tile-a", blob_data=b"A")
        upd.medias.create(id="map-001", key="tile-b", blob_data=b"B")
        count = upd.medias.delete_all_keys("map-001")
        assert count == 2
        assert upd.medias.get_all_keys("map-001") == []

    def test_delete_missing_returns_false(self, upd):
        assert upd.medias.delete("nonexistent") is False
