"""Persist timestamped live match snapshots to disk."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from training_store import atomic_write_json

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
SNAPSHOT_DIR = ROOT / "data" / "live_snapshots"


def _snapshot_path(fixture_id: int) -> Path:
    return SNAPSHOT_DIR / f"{fixture_id}.json"


def load_snapshot_file(path: Path) -> list[dict[str, Any]]:
    """Load a snapshot JSON file (list of polling snapshots)."""
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8-sig", errors="replace") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Failed to load snapshot file %s: %s", path, exc)
        return []
    if not isinstance(data, list):
        log.warning("Snapshot file %s is not a list — skipping", path)
        return []
    return data


def load_snapshots(fixture_id: int) -> list[dict[str, Any]]:
    return load_snapshot_file(_snapshot_path(fixture_id))


def is_snapshot_completed(snapshot: dict[str, Any]) -> bool:
    """True when polling captured a finished match (FT or 2H at/after 90')."""
    status = (snapshot.get("status") or "").upper()
    try:
        minute = int(snapshot.get("minute") or 0)
    except (TypeError, ValueError):
        minute = 0
    return status == "FT" or (status == "2H" and minute >= 90)


def get_final_completed_snapshot(
    snapshots: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the last completed snapshot in a polling history, if any."""
    for snap in reversed(snapshots):
        if is_snapshot_completed(snap):
            return snap
    return None


def iter_snapshot_files() -> list[Path]:
    if not SNAPSHOT_DIR.exists():
        return []
    return sorted(SNAPSHOT_DIR.glob("*.json"))


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
