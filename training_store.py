"""Persistent training state and World Cup completed-match dataset."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
TRAINING_STATE_PATH = ROOT / "training_state.json"
WC_MATCHES_PATH = DATA_DIR / "world_cup_completed_matches.json"
BASE_CACHE_PATH = DATA_DIR / "base_training_cache.json"

DEFAULT_TRAINING_STATE: dict[str, Any] = {
    "last_trained_at": None,
    "last_trained_fixture_date": None,
    "trained_fixture_ids": [],
    "model_versions": {},
    "training_rows_count": 0,
    "new_matches_added": 0,
    "last_incremental_run_status": "never",
    "errors": [],
    "dataset_checksum": None,
    "total_world_cup_matches_used": 0,
}


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return deepcopy(default)
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Failed to load %s: %s — using default", path, exc)
        return deepcopy(default)


def load_training_state() -> dict[str, Any]:
    state = load_json(TRAINING_STATE_PATH, DEFAULT_TRAINING_STATE)
    for key, val in DEFAULT_TRAINING_STATE.items():
        state.setdefault(key, val if not isinstance(val, list) else [])
    state["trained_fixture_ids"] = [int(x) for x in state.get("trained_fixture_ids", [])]
    return state


def save_training_state(state: dict[str, Any]) -> None:
    atomic_write_json(TRAINING_STATE_PATH, state)
    log.info("Saved training state (%s fixture ids trained)", len(state.get("trained_fixture_ids", [])))


def load_wc_matches() -> list[dict[str, Any]]:
    raw = load_json(WC_MATCHES_PATH, {"matches": []})
    if isinstance(raw, list):
        return raw
    return raw.get("matches", [])


def save_wc_matches(matches: list[dict[str, Any]]) -> None:
    atomic_write_json(WC_MATCHES_PATH, {"matches": matches, "count": len(matches)})


def append_wc_matches(new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Append rows deduplicated by fixture_id. Returns full match list."""
    existing = load_wc_matches()
    by_id = {int(m["fixture_id"]): m for m in existing if m.get("fixture_id") is not None}
    added = 0
    for row in new_rows:
        fid = int(row["fixture_id"])
        if fid not in by_id:
            by_id[fid] = row
            added += 1
        else:
            by_id[fid] = {**by_id[fid], **row}
    merged = sorted(by_id.values(), key=lambda m: (m.get("date", ""), m.get("fixture_id", 0)))
    save_wc_matches(merged)
    log.info("WC dataset: %s total rows (%s new)", len(merged), added)
    return merged


def dataset_checksum(matches: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        [{"id": m.get("fixture_id"), "gh": m.get("goals_h"), "ga": m.get("goals_a")} for m in matches],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_base_cache() -> dict[str, Any] | None:
    if not BASE_CACHE_PATH.exists():
        return None
    return load_json(BASE_CACHE_PATH, {})


def save_base_cache(cache: dict[str, Any]) -> None:
    atomic_write_json(BASE_CACHE_PATH, cache)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
