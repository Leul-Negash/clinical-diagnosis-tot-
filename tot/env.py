"""Small .env loader so we don't pull in python-dotenv. Reads KEY=VALUE lines
into os.environ; anything already set in the environment is left as-is."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | os.PathLike | None = None, *, override: bool = False) -> dict[str, str]:
    """Load `.env` into os.environ. Returns the keys that were applied.

    Searches the given path, else `.env` next to the project root (the parent of
    this package). Silently does nothing if no file is found.
    """
    if path is None:
        path = Path(__file__).resolve().parent.parent / ".env"
    path = Path(path)
    if not path.is_file():
        return {}

    applied: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        # Drop optional `export ` prefix and surrounding quotes.
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied
