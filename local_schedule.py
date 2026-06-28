"""Load WC fixtures for the user's local calendar day (fixes UTC date mismatch)."""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, time, timedelta, timezone
from typing import Callable
from zoneinfo import ZoneInfo

from apifootball_client import get_today_fixtures

log = logging.getLogger(__name__)

WC_LEAGUE_ID = 1
LIVE_STATUSES = frozenset({"1H", "HT", "2H", "ET", "P", "LIVE", "BT"})


def get_display_tz():
    """Timezone for 'today' boundaries — DISPLAY_TIMEZONE env or system local."""
    name = (os.getenv("DISPLAY_TIMEZONE") or "").strip()
    if name:
        try:
            return ZoneInfo(name)
        except Exception:
            log.warning("Invalid DISPLAY_TIMEZONE %r — using system local time", name)
    return datetime.now().astimezone().tzinfo


def display_timezone_label() -> str:
    tz = get_display_tz()
    if isinstance(tz, ZoneInfo):
        return tz.key
    offset = tz.utcoffset(datetime.now()) if tz else timedelta(0)
    if offset is None:
        return "UTC"
    total_min = int(offset.total_seconds() // 60)
    sign = "+" if total_min >= 0 else "-"
    total_min = abs(total_min)
    return f"UTC{sign}{total_min // 60:02d}:{total_min % 60:02d}"


def local_today() -> date:
    return datetime.now(get_display_tz()).date()


def local_today_iso() -> str:
    return local_today().isoformat()


def parse_kickoff_utc(fixture: dict) -> datetime:
    raw = fixture["fixture"]["date"]
    if raw.endswith("Z"):
        raw = raw.replace("Z", "+00:00")
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def utc_dates_spanning_local_day(local_day: date, tz) -> list[str]:
    """API-Football date keys (UTC) that can contain kickoffs on local_day."""
    start_local = datetime.combine(local_day, time.min, tzinfo=tz)
    end_local = start_local + timedelta(days=1)
    start_utc = start_local.astimezone(timezone.utc).date()
    end_utc = (end_local - timedelta(microseconds=1)).astimezone(timezone.utc).date()
    dates: list[str] = []
    d = start_utc
    while d <= end_utc:
        dates.append(d.isoformat())
        d += timedelta(days=1)
    return dates


def fixture_belongs_to_local_day(fixture: dict, local_day: date, tz) -> bool:
    status = fixture["fixture"]["status"]["short"]
    if status in LIVE_STATUSES:
        return True
    kickoff_local = parse_kickoff_utc(fixture).astimezone(tz).date()
    return kickoff_local == local_day


def fetch_wc_fixtures_for_local_day(
    local_day: date | None = None,
    *,
    fetch_by_date: Callable[[str], list] | None = None,
) -> list[dict]:
    """
    World Cup fixtures whose kickoff falls on local_day (local timezone).

    Fetches all UTC calendar dates that overlap local_day, then filters by
    local kickoff time. Live matches are always included.
    """
    tz = get_display_tz()
    local_day = local_day or local_today()
    fetch = fetch_by_date or get_today_fixtures
    by_id: dict[int, dict] = {}

    for date_str in utc_dates_spanning_local_day(local_day, tz):
        try:
            batch = fetch(date_str)
        except Exception as exc:
            log.warning("Fixture fetch failed for %s: %s", date_str, exc)
            continue
        for f in batch:
            if f.get("league", {}).get("id") != WC_LEAGUE_ID:
                continue
            if not fixture_belongs_to_local_day(f, local_day, tz):
                continue
            by_id[f["fixture"]["id"]] = f

    return list(by_id.values())


def merge_live_wc_fixtures(fixtures: list[dict]) -> list[dict]:
    """Add any in-play WC fixtures from live=all (catches date-filter misses)."""
    from apifootball_client import get_all_live_fixtures

    by_id = {f["fixture"]["id"]: f for f in fixtures}
    try:
        for f in get_all_live_fixtures():
            if f.get("league", {}).get("id") != WC_LEAGUE_ID:
                continue
            if f["fixture"]["status"]["short"] not in LIVE_STATUSES:
                continue
            by_id[f["fixture"]["id"]] = f
    except Exception as exc:
        log.debug("merge_live_wc_fixtures skipped: %s", exc)
    return list(by_id.values())
