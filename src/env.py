"""Small dotenv loader for local NophiGene processes."""

from __future__ import annotations

import os
from pathlib import Path


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_dotenv(path: str | Path | None = None, *, override: bool = False) -> Path | None:
    """Load simple KEY=VALUE pairs from a local .env file."""
    env_path = Path(path) if path else Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key.startswith("#"):
            continue
        if not override and key in os.environ:
            continue
        os.environ[key] = _strip_quotes(value.strip())
    return env_path
