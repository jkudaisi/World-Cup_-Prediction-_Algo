"""API-Football response cache helpers."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from src.config.pipeline_config import DATA_CACHE, DATA_RAW


def cache_path(endpoint: str, key: str) -> Path:
    safe = key.replace("/", "_").replace(":", "_")
    return DATA_CACHE / endpoint / f"{safe}.json"


def read_cache(endpoint: str, key: str) -> dict[str, Any] | None:
    p = cache_path(endpoint, key)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def write_cache(endpoint: str, key: str, payload: dict[str, Any]) -> Path:
    p = cache_path(endpoint, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def clear_cache() -> list[str]:
    """Remove stale API cache; returns archived paths."""
    removed: list[str] = []
    if DATA_CACHE.exists():
        for f in DATA_CACHE.rglob("*"):
            if f.is_file():
                removed.append(str(f))
        shutil.rmtree(DATA_CACHE, ignore_errors=True)
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    return removed


def ensure_raw_dirs() -> None:
    for sub in ("fixtures", "events", "statistics", "players", "lineups", "injuries"):
        (DATA_RAW / sub).mkdir(parents=True, exist_ok=True)
