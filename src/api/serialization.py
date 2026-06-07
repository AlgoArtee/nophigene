"""JSON and atomic-file helpers for the local API."""

from __future__ import annotations

import json
import math
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def utc_now() -> str:
    """Return an RFC 3339 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def to_jsonable(value: Any) -> Any:
    """Convert pandas, NumPy-like, path, and date values to strict JSON values."""
    if isinstance(value, pd.DataFrame):
        return [to_jsonable(record) for record in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return to_jsonable(value.to_dict())
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(value, "item"):
        try:
            return to_jsonable(value.item())
        except (TypeError, ValueError):
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON through a same-directory temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(to_jsonable(payload), indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    """Read JSON, returning a caller-provided default when absent."""
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))
