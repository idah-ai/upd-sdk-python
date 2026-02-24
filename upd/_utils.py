"""
Internal utilities for the UPD library.

Provides:
- UUIDv7 generation (RFC 9562).
  Python 3.14 ships ``uuid.uuid7()`` in the standard library.
  The implementation falls back to the ``uuid6`` third-party package, then
  to a pure-Python implementation, for older runtimes.
- ISO-8601 UTC timestamp helper
- JSON encode / decode wrappers
- Field-length validators
"""

from __future__ import annotations

import json
import os
import struct
import time
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# UUIDv7 — timestamp-ordered UUID (RFC 9562)
# ---------------------------------------------------------------------------

def _uuid7_pure() -> str:
    """Pure-Python UUIDv7 fallback (no external dependencies)."""
    ms     = int(time.time() * 1000)
    rand_a = int.from_bytes(os.urandom(2), "big") & 0x0FFF
    rand_b = int.from_bytes(os.urandom(8), "big") & 0x3FFFFFFFFFFFFFFF
    hi     = (ms << 16) | (0x7 << 12) | rand_a
    lo     = (0b10 << 62) | rand_b
    raw    = struct.pack(">QQ", hi, lo)
    h      = raw.hex()
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


try:
    # Python 3.14+: uuid7 is in the standard library
    from uuid import uuid7 as _stdlib_uuid7  # type: ignore[attr-defined]

    def new_id() -> str:
        """Return a new UUIDv7 string for use as a UPD primary key."""
        return str(_stdlib_uuid7())

except ImportError:
    try:
        # Older Python: fall back to the uuid6 package
        from uuid6 import uuid7 as _pkg_uuid7  # type: ignore[import]

        def new_id() -> str:  # type: ignore[misc]
            """Return a new UUIDv7 string for use as a UPD primary key."""
            return str(_pkg_uuid7())

    except ImportError:
        # No uuid6 package — pure Python implementation
        def new_id() -> str:  # type: ignore[misc]
            """Return a new UUIDv7 string for use as a UPD primary key."""
            return _uuid7_pure()


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
