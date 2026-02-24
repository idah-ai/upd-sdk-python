"""
UPD Core Schema SQL definitions.

Contains the exact CREATE TABLE statements required by the UPD RFC 0.2 Beta.
These statements are used both for database initialization and for schema
hashing during digital signature operations.
"""

# ---------------------------------------------------------------------------
# Core DDL statements — must match the RFC verbatim for correct schema hashes
# ---------------------------------------------------------------------------

DDL_METADATA = """
CREATE TABLE IF NOT EXISTS metadata (
    key   VARCHAR NOT NULL PRIMARY KEY CHECK(length(key) <= 64),
    value VARCHAR NOT NULL
)
""".strip()

DDL_DATASETS = """
CREATE TABLE IF NOT EXISTS datasets (
    id       VARCHAR NOT NULL PRIMARY KEY CHECK(length(id) <= 64),
    name     VARCHAR NOT NULL             CHECK(length(name) <= 64),
    modality VARCHAR NOT NULL             CHECK(length(modality) <= 64),
    metadata VARCHAR DEFAULT '{}'
)
""".strip()

DDL_MEDIAS = """
CREATE TABLE IF NOT EXISTS medias (
    id         VARCHAR NOT NULL CHECK(length(id) <= 64),
    key        VARCHAR NOT NULL CHECK(length(key) <= 256),
    blob_data  BLOB,
    media_type VARCHAR          CHECK(length(media_type) <= 64),
    metadata   VARCHAR DEFAULT '{}',
    PRIMARY KEY (id, key)
)
""".strip()

DDL_ENTRIES = """
CREATE TABLE IF NOT EXISTS entries (
    id         VARCHAR NOT NULL PRIMARY KEY CHECK(length(id) <= 64),
    dataset_id VARCHAR NOT NULL REFERENCES datasets(id) ON DELETE RESTRICT,
    media_url  VARCHAR NOT NULL,
    metadata   VARCHAR DEFAULT '{}'
)
""".strip()

DDL_IDX_ENTRIES = (
    "CREATE INDEX IF NOT EXISTS idx_entries_dataset_id ON entries (dataset_id)"
)

DDL_ANNOTATIONS = """
CREATE TABLE IF NOT EXISTS annotations (
    id         VARCHAR NOT NULL PRIMARY KEY CHECK(length(id) <= 64),
    entry_id   VARCHAR NOT NULL REFERENCES entries(id) ON DELETE RESTRICT,
    shape_type VARCHAR NOT NULL             CHECK(length(shape_type) <= 64),
    shape_args VARCHAR NOT NULL,
    annotation VARCHAR NOT NULL,
    metadata   VARCHAR DEFAULT '{}'
)
""".strip()

DDL_IDX_ANNOTATIONS_ENTRY  = (
    "CREATE INDEX IF NOT EXISTS idx_annotations_entry_id  ON annotations (entry_id)"
)
DDL_IDX_ANNOTATIONS_SHAPE  = (
    "CREATE INDEX IF NOT EXISTS idx_annotations_shape_type ON annotations (shape_type)"
)

# Ordered list used to initialize an empty UPD file
INITIALIZATION_STATEMENTS: list[str] = [
    DDL_METADATA,
    DDL_DATASETS,
    DDL_MEDIAS,
    DDL_ENTRIES,
    DDL_IDX_ENTRIES,
    DDL_ANNOTATIONS,
    DDL_IDX_ANNOTATIONS_ENTRY,
    DDL_IDX_ANNOTATIONS_SHAPE,
]

# Required global metadata keys (RFC §3.2.1)
REQUIRED_METADATA: dict[str, str] = {
    "Schema-Type":    '"Universal Portable Dataset"',
    "Schema-Version": '"0.2"',
}

DEFAULT_METADATA: dict[str, str] = {
    **REQUIRED_METADATA,
    "Schema-Flavor": '"Vanilla"',
}