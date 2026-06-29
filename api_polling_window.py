"""API-Football polling windows: T-15min through FT+15min per fixture."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from local_schedule import parse_kickoff_utc

PRE_MATCH_BUFFER = timedelta(minutes=15)
POST_FT_BUFFER = timedelta(minutes=15)
# Regulation + stoppage estimate when exact FT time is unknown.
MATCH_DURATION = timedelta(minutes=103)

LIVE_STATUSES = frozenset({"1H", "HT", "2H", "ET", "P", "LIVE", "BT"})
FINAL_STATUSES = frozenset({"FT", "AET", "PEN"})
SKIP_POLL_STATUSES = frozenset({"PST", "CANC"})


def poll_window_start(kickoff: datetime) -> datetime:
    return kickoff - PRE_MATCH_BUFFER


def poll_window_end(
    kickoff: datetime,
    status: str,
    finalized_at: datetime | None,
) -> datetime:
    if finalized_at is not None:
        return finalized_at + POST_FT_BUFFER
    if status in FINAL_STATUSES:
        return kickoff + MATCH_DURATION + POST_FT_BUFFER
    return kickoff + MATCH_DURATION + POST_FT_BUFFER


def is_fixture_in_poll_window(
    fixture: dict,
    status: str,
    *,
    now: datetime | None = None,
    finalized_at: datetime | None = None,
) -> bool:
    status = (status or fixture["fixture"]["status"]["short"]).upper()
    if status in SKIP_POLL_STATUSES:
        return False
    if status in LIVE_STATUSES:
        return True

    kickoff = parse_kickoff_utc(fixture)
    now = now or datetime.now(timezone.utc)
    start = poll_window_start(kickoff)
    end = poll_window_end(kickoff, status, finalized_at)
    return start <= now <= end


def any_fixture_in_poll_window(
    fixtures: list[dict],
    statuses: dict[int, str],
    finalized_at: dict[int, datetime],
    *,
    now: datetime | None = None,
) -> bool:
    now = now or datetime.now(timezone.utc)
    for f in fixtures:
        fid = f["fixture"]["id"]
        status = statuses.get(fid) or f["fixture"]["status"]["short"]
        if is_fixture_in_poll_window(
            f, status, now=now, finalized_at=finalized_at.get(fid),
        ):
            return True
    return False


def seconds_until_next_poll_window(
    fixtures: list[dict],
    statuses: dict[int, str],
    finalized_at: dict[int, datetime],
    *,
    now: datetime | None = None,
    min_sleep: float = 30,
    max_sleep: float = 1800,
) -> float:
    """Seconds until the next T-15min window (0 if a fixture is in-window now)."""
    now = now or datetime.now(timezone.utc)
    if any_fixture_in_poll_window(fixtures, statuses, finalized_at, now=now):
        return min_sleep

    next_wake: datetime | None = None
    for f in fixtures:
        fid = f["fixture"]["id"]
        status = (statuses.get(fid) or f["fixture"]["status"]["short"]).upper()
        if status in SKIP_POLL_STATUSES:
            continue

        kickoff = parse_kickoff_utc(f)
        fin = finalized_at.get(fid)
        start = poll_window_start(kickoff)
        end = poll_window_end(kickoff, status, fin)

        if start <= now <= end:
            return min_sleep

        if now < start:
            if next_wake is None or start < next_wake:
                next_wake = start

    if next_wake is None:
        return max_sleep
    return max(min_sleep, min((next_wake - now).total_seconds(), max_sleep))
