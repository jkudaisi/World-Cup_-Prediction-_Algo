"""Provider manifests and resume state for API-Football backfill."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data.providers.api_football.paths import MANIFEST_ROOT
from training_store import atomic_write_json, utc_now_iso

PROVIDER = "api_football"


def _manifest_path(name: str) -> Path:
    return MANIFEST_ROOT / name


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def build_backfill_manifest(
    *,
    run_id: str | None = None,
    status: str = "running",
    date_from: str | None = None,
    date_to: str | None = None,
    teams_requested: int = 0,
    teams_resolved: int = 0,
    fixtures_discovered: int = 0,
    fixtures_deduplicated: int = 0,
    endpoint_requests: dict[str, int] | None = None,
    cache: dict[str, int] | None = None,
    missing_data_summary: dict[str, Any] | None = None,
    failed_requests_count: int = 0,
    notes: list[str] | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
) -> dict[str, Any]:
    return {
        "provider": PROVIDER,
        "run_id": run_id or new_run_id(),
        "status": status,
        "started_at": started_at or utc_now_iso(),
        "finished_at": finished_at,
        "date_range": {"from": date_from, "to": date_to},
        "teams_requested": teams_requested,
        "teams_resolved": teams_resolved,
        "fixtures_discovered": fixtures_discovered,
        "fixtures_deduplicated": fixtures_deduplicated,
        "endpoint_requests": endpoint_requests or {},
        "cache": cache or {"hits": 0, "misses": 0},
        "missing_data_summary": missing_data_summary or {},
        "failed_requests_count": failed_requests_count,
        "notes": notes or [],
    }


def write_manifest(name: str, data: dict[str, Any]) -> Path:
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    path = _manifest_path(name)
    atomic_write_json(path, data)
    return path


def load_resume_state() -> dict[str, Any]:
    default = {
        "last_completed_team_id": None,
        "completed_team_ids": [],
        "completed_fixture_ids": [],
        "failed_fixture_ids": [],
        "pending_fixture_ids": [],
        "last_successful_checkpoint_at": None,
    }
    return load_json(_manifest_path("resume_state.json"), default)


def save_resume_state(state: dict[str, Any]) -> Path:
    return write_manifest("resume_state.json", state)


def append_failed_request(entry: dict[str, Any]) -> Path:
    path = _manifest_path("failed_requests.json")
    items = load_json(path, [])
    if not isinstance(items, list):
        items = items.get("requests", []) if isinstance(items, dict) else []
    items.append({**entry, "logged_at": utc_now_iso()})
    return write_manifest("failed_requests.json", items)
