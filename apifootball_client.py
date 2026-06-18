"""HTTP client for API-Football (api-sports.io) with daily request budgeting."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import APIFOOTBALL_KEY

log = logging.getLogger(__name__)

BASE_URL = "https://v3.football.api-sports.io"
WC_LEAGUE_ID = 1
WC_SEASON = 2026
DAILY_LIMIT = 7500

_calls_made_today: int = 0
_calls_date: str = ""

_session: requests.Session | None = None

STAT_TYPE_MAP = {
    "Shots on Goal": "shots_on_goal",
    "Shots off Goal": "shots_off_goal",
    "Total Shots": "shots_total",
    "Blocked Shots": "shots_blocked",
    "Corner Kicks": "corner_kicks",
    "Offsides": "offsides",
    "Ball Possession": "ball_possession",
    "Yellow Cards": "yellow_cards",
    "Red Cards": "red_cards",
    "Goalkeeper Saves": "goalkeeper_saves",
    "Total passes": "total_passes",
    "Passes accurate": "passes_accurate",
    "Passes %": "passes_pct",
    "Fouls": "fouls",
    "expected_goals": "expected_goals",
}

STAT_DEFAULTS = {v: None for v in STAT_TYPE_MAP.values()}


class APIFootballError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(f"APIFootball {status}: {message}")


def _session_get() -> requests.Session:
    global _session
    if _session is None:
        retry = Retry(total=2, backoff_factor=1, status_forcelist=[429, 500, 502, 503])
        adapter = HTTPAdapter(max_retries=retry)
        _session = requests.Session()
        _session.mount("https://", adapter)
        _session.mount("http://", adapter)
    return _session


def _bump_counter() -> None:
    global _calls_made_today, _calls_date
    today = date.today().isoformat()
    if _calls_date != today:
        _calls_made_today = 0
        _calls_date = today
    _calls_made_today += 1


def calls_remaining() -> int:
    today = date.today().isoformat()
    if _calls_date != today:
        return DAILY_LIMIT
    return max(0, DAILY_LIMIT - _calls_made_today)


def _headers() -> dict[str, str]:
    return {"x-apisports-key": APIFOOTBALL_KEY}


def _get(path: str, params: dict) -> Any:
    if calls_remaining() <= 0:
        raise APIFootballError(0, "Daily request budget exhausted")

    if not APIFOOTBALL_KEY:
        raise APIFootballError(401, "APIFOOTBALL_KEY not configured")

    session = _session_get()
    resp = session.get(
        BASE_URL + path,
        params=params,
        headers=_headers(),
        timeout=(8, 12),
    )
    if not resp.ok:
        raise APIFootballError(resp.status_code, resp.text[:300])

    data = resp.json()
    errors = data.get("errors") or {}
    if isinstance(errors, dict) and errors:
        raise APIFootballError(400, str(errors))

    _bump_counter()
    return data.get("response", [])


def get_today_fixtures(date_str: str) -> list[dict]:
    result = _get("/fixtures", {"league": WC_LEAGUE_ID, "season": WC_SEASON, "date": date_str})
    return result if isinstance(result, list) else []


def get_all_live_fixtures() -> list[dict]:
    """All in-play fixtures worldwide (`/fixtures?live=all`)."""
    result = _get("/fixtures", {"live": "all"})
    return result if isinstance(result, list) else []


def get_live_fixtures(league_id: int = WC_LEAGUE_ID) -> list[dict]:
    """Legacy: filter live=all to one league client-side."""
    return [f for f in get_all_live_fixtures() if f.get("league", {}).get("id") == league_id]


def _empty_side_stats() -> dict:
    return dict(STAT_DEFAULTS)


def _parse_stat_value(raw: Any, key: str) -> Any:
    if raw is None:
        return None
    if key in ("ball_possession", "passes_pct"):
        if isinstance(raw, str):
            return raw.strip() or None
        return str(raw)
    if key == "expected_goals":
        try:
            return str(raw)
        except (TypeError, ValueError):
            return None
    try:
        if isinstance(raw, str) and raw.endswith("%"):
            return int(raw.replace("%", "").strip())
        return int(raw)
    except (TypeError, ValueError):
        return None


def get_fixture_stats(fixture_id: int, home_team_id: int | None = None, away_team_id: int | None = None) -> dict:
    raw = _get("/fixtures/statistics", {"fixture": fixture_id})
    out = {"home": _empty_side_stats(), "away": _empty_side_stats()}
    if not isinstance(raw, list):
        return out

    for block in raw:
        team_id = (block.get("team") or {}).get("id")
        if home_team_id is not None and team_id == home_team_id:
            side = "home"
        elif away_team_id is not None and team_id == away_team_id:
            side = "away"
        else:
            side = None
        if side is None:
            idx = raw.index(block)
            side = "home" if idx == 0 else "away"
        target = out[side]
        for item in block.get("statistics") or []:
            key = STAT_TYPE_MAP.get(item.get("type"))
            if key:
                target[key] = _parse_stat_value(item.get("value"), key)
    return out


def get_fixture_events(fixture_id: int) -> list[dict]:
    result = _get("/fixtures/events", {"fixture": fixture_id})
    return result if isinstance(result, list) else []


def get_fixture_players(fixture_id: int) -> dict:
    raw = _get("/fixtures/players", {"fixture": fixture_id})
    out: dict[str, Any] = {"home": [], "away": []}
    if isinstance(raw, list):
        for i, block in enumerate(raw[:2]):
            side = "home" if i == 0 else "away"
            out[side] = block.get("players") or []
    return out


def get_fixture_lineups(fixture_id: int) -> dict:
    raw = _get("/fixtures/lineups", {"fixture": fixture_id})
    out: dict[str, Any] = {}
    if isinstance(raw, list):
        for i, block in enumerate(raw[:2]):
            side = "home" if i == 0 else "away"
            out[side] = {
                "formation": block.get("formation"),
                "startXI": block.get("startXI") or [],
                "substitutes": block.get("substitutes") or [],
            }
    return out


def get_fixture_full(fixture_id: int, home_team_id: int | None = None, away_team_id: int | None = None) -> dict:
    partial = False
    stats: dict = {"home": _empty_side_stats(), "away": _empty_side_stats()}
    events: list[dict] = []

    try:
        stats = get_fixture_stats(fixture_id, home_team_id, away_team_id)
    except APIFootballError as exc:
        log.warning("fixture %s stats failed: %s", fixture_id, exc)
        partial = True

    try:
        events = get_fixture_events(fixture_id)
    except APIFootballError as exc:
        log.warning("fixture %s events failed: %s", fixture_id, exc)
        partial = True

    result: dict[str, Any] = {"stats": stats, "events": events}
    if partial:
        result["partial"] = True
    return result
