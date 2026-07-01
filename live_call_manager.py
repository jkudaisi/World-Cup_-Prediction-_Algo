"""Smart API call scheduling for live match updates."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from apifootball_client import (
    APIFootballError,
    calls_remaining,
    get_all_live_fixtures,
    get_fixture_events,
    get_fixture_injuries,
    get_fixture_lineups,
    get_fixture_players,
    get_fixture_stats,
)

log = logging.getLogger(__name__)

LINEUP_INTERVAL = 180          # 2–5 min
PLAYERS_INTERVAL = 180
RARE_INTERVAL = 600            # 10 min for rarely-changing data
RESERVE_CALLS = 5


@dataclass
class FixtureCallState:
    fixture_id: int
    last_refresh: float = 0.0
    last_lineups: float = 0.0
    last_players: float = 0.0
    last_rare: float = 0.0
    cached_lineups: dict = field(default_factory=dict)
    cached_players: dict = field(default_factory=dict)
    cached_injuries: list = field(default_factory=list)
    injuries_loaded: bool = False


_call_states: dict[int, FixtureCallState] = {}


def _state(fixture_id: int) -> FixtureCallState:
    if fixture_id not in _call_states:
        _call_states[fixture_id] = FixtureCallState(fixture_id=fixture_id)
    return _call_states[fixture_id]


def _due(last_ts: float, interval: float) -> bool:
    return (time.monotonic() - last_ts) >= interval


def live_poll_interval_seconds() -> int:
    """Live scheduler + per-fixture stats/events refresh interval (from .env)."""
    from trading_config import get_config
    return max(1, int(get_config().live_poll_interval_seconds))


# Backward-compatible alias for imports that expect a module constant.
REFRESH_INTERVAL = 20


def should_refresh(fixture_id: int) -> bool:
    st = _state(fixture_id)
    return _due(st.last_refresh, live_poll_interval_seconds())


def fetch_live_fixtures_list() -> list[dict]:
    """One API call: all currently live fixtures."""
    if calls_remaining() <= RESERVE_CALLS:
        return []
    try:
        return get_all_live_fixtures()
    except APIFootballError as exc:
        log.warning("live=all fetch failed: %s", exc)
        return []


def fetch_live_bundle(
    fixture_id: int,
    home_team_id: int | None,
    away_team_id: int | None,
    *,
    force: bool = False,
) -> dict[str, Any] | None:
    """Fetch stats + events (+ optional lineups/players) with smart intervals."""
    if calls_remaining() <= RESERVE_CALLS:
        log.warning("Budget too low for fixture %s", fixture_id)
        return None

    st = _state(fixture_id)
    if not force and not _due(st.last_refresh, live_poll_interval_seconds()):
        return None

    calls_used = 0
    stats: dict = {"home": {}, "away": {}}
    events: list = []
    partial = False

    try:
        stats = get_fixture_stats(fixture_id, home_team_id, away_team_id)
        calls_used += 1
    except APIFootballError as exc:
        log.warning("fixture %s stats failed: %s", fixture_id, exc)
        partial = True

    try:
        events = get_fixture_events(fixture_id)
        calls_used += 1
    except APIFootballError as exc:
        log.warning("fixture %s events failed: %s", fixture_id, exc)
        partial = True

    lineups = st.cached_lineups
    if force or _due(st.last_lineups, LINEUP_INTERVAL):
        if calls_remaining() > RESERVE_CALLS:
            try:
                lineups = get_fixture_lineups(fixture_id)
                st.cached_lineups = lineups
                st.last_lineups = time.monotonic()
                calls_used += 1
            except APIFootballError as exc:
                log.warning("fixture %s lineups failed: %s", fixture_id, exc)

    players = st.cached_players
    if force or _due(st.last_players, PLAYERS_INTERVAL):
        if calls_remaining() > RESERVE_CALLS:
            try:
                players = get_fixture_players(fixture_id)
                st.cached_players = players
                st.last_players = time.monotonic()
                calls_used += 1
            except APIFootballError as exc:
                log.warning("fixture %s players failed: %s", fixture_id, exc)

    injuries = st.cached_injuries
    if not st.injuries_loaded:
        try:
            from live_context import load_prematch_injuries
            cached = load_prematch_injuries(fixture_id)
            if cached is not None:
                injuries = cached
                st.cached_injuries = injuries
                st.injuries_loaded = True
        except Exception:
            pass
    if force or _due(st.last_rare, RARE_INTERVAL):
        if calls_remaining() > RESERVE_CALLS:
            try:
                live_inj = get_fixture_injuries(fixture_id)
                if live_inj:
                    injuries = live_inj
                    st.cached_injuries = injuries
                st.injuries_loaded = True
                st.last_rare = time.monotonic()
                calls_used += 1
            except APIFootballError as exc:
                log.warning("fixture %s injuries failed: %s", fixture_id, exc)

    st.last_refresh = time.monotonic()

    result: dict[str, Any] = {
        "stats": stats,
        "events": events,
        "lineups": lineups,
        "players": players,
        "injuries": injuries,
        "api_calls_used": calls_used,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    if partial:
        result["partial"] = True
    return result


def clear_fixture_state(fixture_id: int) -> None:
    _call_states.pop(fixture_id, None)
