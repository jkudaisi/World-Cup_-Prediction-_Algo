"""Persist timestamped live match snapshots to disk."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from training_store import atomic_write_json, load_json

ROOT = Path(__file__).parent
SNAPSHOT_DIR = ROOT / "data" / "live_snapshots"


def _snapshot_path(fixture_id: int) -> Path:
    return SNAPSHOT_DIR / f"{fixture_id}.json"


def load_snapshots(fixture_id: int) -> list[dict[str, Any]]:
    return load_json(_snapshot_path(fixture_id), [])


def snapshot_fingerprint(snapshot: dict[str, Any]) -> str:
    """Hash minute + score + core stats to detect duplicate snapshots."""
    payload = {
        "minute": snapshot.get("minute"),
        "status": snapshot.get("status"),
        "score": snapshot.get("score"),
        "stats": snapshot.get("stats"),
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def append_snapshot(fixture_id: int, snapshot: dict[str, Any]) -> bool:
    """Append snapshot if not a duplicate. Returns True if appended."""
    snapshots = load_snapshots(fixture_id)
    fp = snapshot_fingerprint(snapshot)
    if snapshots and snapshots[-1].get("_fingerprint") == fp:
        return False
    snapshot = dict(snapshot)
    snapshot["_fingerprint"] = fp
    snapshots.append(snapshot)
    atomic_write_json(_snapshot_path(fixture_id), snapshots)
    return True


def get_latest_snapshot(fixture_id: int) -> dict[str, Any] | None:
    snapshots = load_snapshots(fixture_id)
    return snapshots[-1] if snapshots else None
