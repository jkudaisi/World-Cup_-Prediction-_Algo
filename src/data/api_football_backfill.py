"""Persist API-Football responses — bridges legacy callers to provider raw store."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.config.pipeline_config import DATA_RAW as LEGACY_RAW
from src.data.providers.api_football.paths import RAW_ROOT as PROVIDER_RAW
from src.data.providers.api_football.raw_store import APIFootballRawStore
from training_store import atomic_write_json, utc_now_iso

log = logging.getLogger(__name__)

RAW_SUBDIRS = ("fixtures", "events", "statistics", "players", "lineups", "injuries")
_store = APIFootballRawStore()


def ensure_raw_dirs() -> None:
    _store._ensure_dirs()
    for sub in RAW_SUBDIRS:
        (LEGACY_RAW / sub).mkdir(parents=True, exist_ok=True)


def _legacy_path(kind: str, fixture_id: int) -> Path:
    return LEGACY_RAW / kind / f"{fixture_id}.json"


def load_raw(kind: str, fixture_id: int) -> dict[str, Any] | None:
    """Load raw payload — prefers provider layout, falls back to legacy flat files."""
    loaded = _store.load_endpoint(kind, str(fixture_id))
    if loaded:
        # Normalize wrapper → legacy shape for existing feature_store code
        if "api_response" in loaded:
            return {
                "fixture_id": fixture_id,
                "fetched_at": loaded.get("fetched_at"),
                "data": loaded.get("api_response"),
                "success": loaded.get("success"),
                "error": loaded.get("error"),
            }
        return loaded
    p = _legacy_path(kind, fixture_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def persist_fixture_bundle(
    fixture_id: int,
    *,
    fixture: dict[str, Any] | None = None,
    stats: dict[str, Any] | None = None,
    events: list[dict[str, Any]] | None = None,
    lineups: dict[str, Any] | None = None,
    players: dict[str, Any] | None = None,
    injuries: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Write available API payloads; missing layers are skipped, not fatal."""
    ensure_raw_dirs()
    written: list[str] = []
    mapping = [
        ("fixtures_by_fixture", str(fixture_id), "/fixtures", {"fixture": fixture_id}, fixture),
        ("events", str(fixture_id), "/fixtures/events", {"fixture": fixture_id}, events),
        ("statistics", str(fixture_id), "/fixtures/statistics", {"fixture": fixture_id}, stats),
        ("players", str(fixture_id), "/fixtures/players", {"fixture": fixture_id}, players),
        ("lineups", str(fixture_id), "/fixtures/lineups", {"fixture": fixture_id}, lineups),
        ("injuries", str(fixture_id), "/injuries", {"fixture": fixture_id}, injuries),
    ]
    for kind, key, endpoint, params, data in mapping:
        if data is None:
            continue
        ok = data is not None and (not isinstance(data, list) or len(data) > 0 or kind == "events")
        if isinstance(data, dict) and not data and kind != "fixtures_by_fixture":
            ok = False
        path = _store.save_endpoint_response(
            kind if kind != "fixtures_by_fixture" else "fixtures_by_fixture",
            key,
            endpoint=endpoint,
            params=params,
            api_response=data,
            success=ok,
            error=None if ok else "endpoint returned no data",
        )
        written.append(str(path))
        # Legacy mirror for transitional readers
        legacy_kind = "fixtures" if kind == "fixtures_by_fixture" else kind
        legacy_payload = {"fixture_id": fixture_id, "fetched_at": utc_now_iso(), "data": data}
        lp = _legacy_path(legacy_kind, fixture_id)
        atomic_write_json(lp, legacy_payload)
        written.append(str(lp))
    return written


def fetch_and_persist_fixture(
    fixture_id: int,
    *,
    home_team_id: int | None = None,
    away_team_id: int | None = None,
    include_lineups: bool = True,
    include_players: bool = True,
    include_injuries: bool = True,
) -> list[str]:
    from src.data.providers.api_football.client import APIFootballClient

    client = APIFootballClient()
    stats = client.get_fixture_statistics(fixture_id)
    events = client.get_fixture_events(fixture_id)
    lineups = client.get_fixture_lineups(fixture_id) if include_lineups else None
    players = client.get_fixture_players(fixture_id) if include_players else None
    injuries = client.get_injuries(fixture=fixture_id) if include_injuries else None
    return persist_fixture_bundle(
        fixture_id,
        stats=stats,
        events=events,
        lineups=lineups,
        players=players,
        injuries=injuries,
    )
