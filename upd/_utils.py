"""
Internal utilities for the UPD library.

Provides:
- UUIDv7 generation (RFC 9562) via ``uuid.uuid7()`` (Python 3.14+ stdlib).
- ISO-8601 UTC timestamp helper
- JSON encode / decode wrappers
- Field-length validators
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid7
from typing import Any


# ---------------------------------------------------------------------------
# UUIDv7 — timestamp-ordered UUID (RFC 9562)
# ---------------------------------------------------------------------------

def new_id() -> str:
    """Return a new UUIDv7 string for use as a UPD primary key."""
    return str(uuid7())


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------

def utc_now() -> str:
    """Return current UTC time as an ISO-8601 string, e.g. ``2025-11-01T10:00:00Z``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def json_dumps(obj: Any) -> str:
    """Serialize *obj* to a compact JSON string."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def json_loads(text: str | None) -> Any:
    """Deserialize a JSON string; returns ``{}`` for ``None`` / empty input."""
    if not text:
        return {}
    return json.loads(text)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_id(value: str, field: str = "id") -> None:
    """Raise ``ValueError`` if *value* exceeds the RFC 64-character limit."""
    if len(value) > 64:
        raise ValueError(f"{field!r} must be ≤ 64 characters (got {len(value)})")


def validate_name(value: str, field: str = "name") -> None:
    """Raise ``ValueError`` if *value* exceeds the RFC 64-character limit."""
    if len(value) > 64:
        raise ValueError(f"{field!r} must be ≤ 64 characters (got {len(value)})")