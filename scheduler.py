"""Smart daily scheduler: morning init + live prediction loop (interval from .env)."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from api_polling_window import (
    POST_FT_BUFFER,
    PRE_MATCH_BUFFER,
    any_fixture_in_poll_window,
    seconds_until_next_poll_window,
)
from apifootball_client import (
    APIFootballError,
    calls_remaining,
)
from incremental_trainer import mark_fixture_for_training, process_pending_training
from live_call_manager import live_poll_interval_seconds
from live_updater import mark_fixture_final, run_live_cycle
from local_schedule import (
    display_timezone_label,
    local_today_iso,
    merge_live_wc_fixtures,
    parse_kickoff_utc,
)
from team_names import resolve_team_name
from training_store import load_training_state

log = logging.getLogger(__name__)

DAILY_BUDGET = 7500
RESERVE_CALLS = 5
SKIP_STATUSES = frozenset({"FT", "AET", "PEN", "PST", "CANC", "TBD"})
FINAL_STATUSES = frozenset({"FT", "AET", "PEN"})
LIVE_STATUSES = frozenset({"1H", "HT", "2H", "ET", "P", "LIVE", "BT"})
UPCOMING_STATUSES = frozenset({"NS", "TBD", "SUSP", "INT"})

schedule: "DaySchedule | None" = None
cached_status: dict[int, str] = {}
_scheduler_thread: threading.Thread | None = None
_today_view_fetched: str = ""
_all_today_fixtures: list[dict] = []
_last_incremental_check: datetime | None = None
_last_status_refresh: datetime | None = None
_fixture_finalized_at: dict[int, datetime] = {}
INCREMENTAL_CHECK_INTERVAL = timedelta(minutes=5)
STATUS_REFRESH_INTERVAL = timedelta(seconds=60)
OUTSIDE_WINDOW_MIN_SLEEP = 30


@dataclass
class DaySchedule:
    date: str
    fixtures: list[dict]
    n_matches: int
    live_cycles: int = 0
    calls_used: int = 0


def parse_kickoff(fixture: dict) -> datetime:
    return parse_kickoff_utc(fixture)


def _sync_final_fixtures_for_training() -> None:
    state = load_training_state()
    trained = set(state.get("trained_fixture_ids", []))
    sources = list(_all_today_fixtures)
    if schedule:
        sources.extend(schedule.fixtures)
    seen: set[int] = set()
    for f in sources:
        fid = f["fixture"]["id"]
        if fid in seen:
            continue
        seen.add(fid)
        status = cached_status.get(fid) or f["fixture"]["status"]["short"]
        if status in FINAL_STATUSES and fid not in trained:
            mark_fixture_for_training(fid)


def _maybe_run_incremental_training() -> None:
    global _last_incremental_check
    now = datetime.now(timezone.utc)
    if _last_incremental_check and (now - _last_incremental_check) < INCREMENTAL_CHECK_INTERVAL:
        return
    _last_incremental_check = now
    _sync_final_fixtures_for_training()
    try:
        result = process_pending_training()
        if result and result.get("status") == "success":
            log.info(
                "Incremental training complete: %s new WC matches",
                result.get("new_matches_used", 0),
            )
        elif result and result.get("status") == "skipped":
            log.debug("Incremental training skipped: %s", result.get("reason"))
    except Exception as exc:
        log.error("Incremental training failed: %s", exc)


def _refresh_fixture_statuses(*, force: bool = False) -> None:
    """Detect FT transitions (~1 API call/minute max, or forced on manual refresh)."""
    global _last_status_refresh
    now = datetime.now(timezone.utc)
    if (
        not force
        and _last_status_refresh
        and (now - _last_status_refresh) < STATUS_REFRESH_INTERVAL
    ):
        return
    _last_status_refresh = now
    try:
        fixtures = _fetch_wc_fixtures_for_local_day()
    except APIFootballError as exc:
        log.warning("Status refresh failed: %s", exc)
        return
    wc = fixtures
    _all_today_fixtures[:] = wc
    for f in wc:
        fid = f["fixture"]["id"]
        status = f["fixture"]["status"]["short"]
        old = cached_status.get(fid)
        cached_status[fid] = status
        if status in FINAL_STATUSES:
            if old not in FINAL_STATUSES:
                _fixture_finalized_at[fid] = now
                log.info("Match %s finished (%s) — queued for incremental training", fid, status)
                mark_fixture_for_training(fid)
                mark_fixture_final(fid)
            elif old is None:
                state = load_training_state()
                if fid not in state.get("trained_fixture_ids", []):
                    mark_fixture_for_training(fid)


def morning_init() -> DaySchedule | None:
    global _all_today_fixtures
    today = local_today_iso()
    try:
        wc_fixtures = _fetch_wc_fixtures_for_local_day()
    except APIFootballError as exc:
        log.error("morning_init failed: %s", exc)
        return None

    _all_today_fixtures = list(wc_fixtures)
    trained = set(load_training_state().get("trained_fixture_ids", []))
    for f in wc_fixtures:
        fid = f["fixture"]["id"]
        cached_status[fid] = f["fixture"]["status"]["short"]
        if f["fixture"]["status"]["short"] in FINAL_STATUSES and fid not in trained:
            mark_fixture_for_training(fid)

    active = [f for f in wc_fixtures if f["fixture"]["status"]["short"] not in SKIP_STATUSES]
    log.info(
        "Day init: %s WC fixtures on local %s (%s active, tz %s)",
        len(wc_fixtures), today, len(active), display_timezone_label(),
    )
    return DaySchedule(date=today, fixtures=wc_fixtures, n_matches=len(active))


def should_poll_api_football() -> bool:
    """True when any fixture is in its T-15min … FT+15min API polling window."""
    return _any_match_window()


def refresh_poll_window_statuses(*, force: bool = False) -> None:
    """Refresh today's fixture list and cached statuses during the poll window."""
    if force or should_poll_api_football():
        _refresh_fixture_statuses(force=force)


def get_in_play_wc_fixtures() -> list[dict]:
    """WC fixtures currently in play from the scheduler's today list."""
    out: list[dict] = []
    for f in _all_today_fixtures:
        fid = f["fixture"]["id"]
        status = (cached_status.get(fid) or f["fixture"]["status"]["short"] or "").upper()
        if status in LIVE_STATUSES:
            out.append(f)
    return out


def get_trading_fixture_snapshots() -> dict[int, dict]:
    """Latest status/score for fixtures on today's schedule (for trading scans)."""
    snapshots: dict[int, dict] = {}
    for f in _all_today_fixtures:
        fix = f["fixture"]
        fid = fix["id"]
        status = (cached_status.get(fid) or fix["status"]["short"] or "NS").upper()
        goals = f.get("goals") or {}
        sh = goals.get("home")
        sa = goals.get("away")
        snapshots[fid] = {
            "fixture_id": fid,
            "status": status,
            "score_home": int(sh) if sh is not None else 0,
            "score_away": int(sa) if sa is not None else 0,
            "is_live": status in LIVE_STATUSES,
            "elapsed": (fix.get("status") or {}).get("elapsed"),
            "home_api": f["teams"]["home"]["name"],
            "away_api": f["teams"]["away"]["name"],
        }
    return snapshots


def _any_match_window() -> bool:
    return any_fixture_in_poll_window(
        _all_today_fixtures, cached_status, _fixture_finalized_at,
    )


def _seconds_until_poll_window() -> float:
    return seconds_until_next_poll_window(
        _all_today_fixtures,
        cached_status,
        _fixture_finalized_at,
        min_sleep=OUTSIDE_WINDOW_MIN_SLEEP,
    )


def run_scheduler() -> None:
    global schedule
    while True:
        now_date = local_today_iso()

        if schedule is None or schedule.date != now_date:
            log.info("Running morning init for %s", now_date)
            schedule = morning_init()
            if schedule is None:
                log.info("Init failed — sleeping 30 min")
                time.sleep(1800)
                continue

        in_poll_window = _any_match_window()
        if in_poll_window:
            _refresh_fixture_statuses()

        if calls_remaining() > RESERVE_CALLS and in_poll_window:
            try:
                result = run_live_cycle()
                if schedule:
                    schedule.live_cycles += 1
                    schedule.calls_used += result.get("calls_used", 0)
                if result.get("live_count", 0) > 0:
                    log.info(
                        "[live] %s matches in play, %s updated, %s calls left",
                        result["live_count"], result.get("updated", 0), calls_remaining(),
                    )
                try:
                    from trading_service import refresh_trading_cycle
                    refresh_trading_cycle()
                except Exception as trade_exc:
                    log.debug("Trading refresh skipped: %s", trade_exc)
            except Exception as exc:
                log.exception("Live cycle failed: %s", exc)
            _maybe_run_incremental_training()
            time.sleep(live_poll_interval_seconds())
        else:
            _maybe_run_incremental_training()
            sleep_secs = _seconds_until_poll_window() if not in_poll_window else 60
            if not in_poll_window and sleep_secs > OUTSIDE_WINDOW_MIN_SLEEP:
                log.debug("Outside API poll window — sleeping %.0fs until next match", sleep_secs)
            time.sleep(sleep_secs)


def start_scheduler() -> None:
    global _scheduler_thread
    if _scheduler_thread is not None and _scheduler_thread.is_alive():
        return
    try:
        from live_updater import prune_stale_live_predictions
        prune_stale_live_predictions()
    except Exception as exc:
        log.debug("Live prediction prune skipped: %s", exc)
    _scheduler_thread = threading.Thread(
        target=run_scheduler, daemon=True, name="wc2026-scheduler",
    )
    _scheduler_thread.start()
    log.info("Scheduler thread started (live poll every %ss)", live_poll_interval_seconds())


def is_scheduler_running() -> bool:
    return _scheduler_thread is not None and _scheduler_thread.is_alive()


def get_scheduler_status() -> dict:
    from apifootball_client import DAILY_LIMIT
    from live_updater import get_live_status
    live_status = get_live_status()
    in_window = _any_match_window()
    return {
        "date": schedule.date if schedule else local_today_iso(),
        "local_timezone": display_timezone_label(),
        "n_matches": schedule.n_matches if schedule else 0,
        "live_poll_interval_seconds": live_poll_interval_seconds(),
        "api_poll_window_active": in_window,
        "api_poll_pre_match_minutes": int(PRE_MATCH_BUFFER.total_seconds() // 60),
        "api_poll_post_ft_minutes": int(POST_FT_BUFFER.total_seconds() // 60),
        "seconds_until_poll_window": 0 if in_window else int(_seconds_until_poll_window()),
        "live_cycles": schedule.live_cycles if schedule else 0,
        "calls_used_today": schedule.calls_used if schedule else 0,
        "api_budget_remaining": calls_remaining(),
        "daily_limit": DAILY_LIMIT,
        "active_fixture_ids": list(cached_status.keys()),
        "cached_statuses": dict(cached_status),
        "live_status": live_status,
    }


def _fetch_wc_fixtures_for_local_day(*, include_live_merge: bool | None = None) -> list[dict]:
    """Load World Cup fixtures for the local calendar day.

    ``include_live_merge`` adds a live=all call when True. Defaults to True only
    while a fixture is in its T-15min … FT+15min poll window.
    """
    from local_schedule import fetch_wc_fixtures_for_local_day

    if include_live_merge is None:
        include_live_merge = _any_match_window()

    wc = fetch_wc_fixtures_for_local_day()
    if include_live_merge:
        wc = merge_live_wc_fixtures(wc)
    for f in wc:
        fid = f["fixture"]["id"]
        cached_status[fid] = f["fixture"]["status"]["short"]
    return wc


def _fetch_wc_fixtures_for_date(date_str: str) -> list[dict]:
    """Legacy: load by API date string (prefer _fetch_wc_fixtures_for_local_day)."""
    from apifootball_client import get_today_fixtures

    raw = get_today_fixtures(date_str)
    wc = [f for f in raw if f.get("league", {}).get("id") == 1]
    for f in wc:
        fid = f["fixture"]["id"]
        cached_status[fid] = f["fixture"]["status"]["short"]
    return wc


def _today_match_sort_key(entry: dict) -> tuple:
    status = (entry.get("status") or "NS").upper()
    if status in LIVE_STATUSES:
        bucket = 0
    elif status in FINAL_STATUSES:
        bucket = 2
    else:
        bucket = 1
    return bucket, entry.get("kickoff") or ""


def get_today_view(*, refresh: bool = False) -> dict:
    """Today's WC fixtures merged with live predictions."""
    global schedule, _today_view_fetched, _all_today_fixtures
    from apifootball_client import DAILY_LIMIT
    from config import APIFOOTBALL_KEY
    from live_trainer import get_live_state
    from live_updater import _load_live_predictions, extract_display_stats

    today = local_today_iso()
    fixtures: list[dict] = []
    if refresh and APIFOOTBALL_KEY:
        try:
            fixtures = _fetch_wc_fixtures_for_local_day(
                include_live_merge=should_poll_api_football(),
            )
            _all_today_fixtures = list(fixtures)
            _today_view_fetched = today
            active = [f for f in fixtures if f["fixture"]["status"]["short"] not in SKIP_STATUSES]
            schedule = DaySchedule(date=today, fixtures=fixtures, n_matches=len(active))
            if should_poll_api_football():
                _refresh_fixture_statuses(force=True)
        except APIFootballError as exc:
            log.warning("get_today_view refresh failed: %s", exc)
    elif _all_today_fixtures and schedule and schedule.date == today:
        fixtures = list(_all_today_fixtures)
    elif schedule and schedule.date == today:
        fixtures = list(schedule.fixtures)
    elif _today_view_fetched != today and APIFOOTBALL_KEY:
        try:
            fixtures = _fetch_wc_fixtures_for_local_day(include_live_merge=False)
            _all_today_fixtures = list(fixtures)
            _today_view_fetched = today
            if schedule is None or schedule.date != today:
                active = [f for f in fixtures if f["fixture"]["status"]["short"] not in SKIP_STATUSES]
                schedule = DaySchedule(date=today, fixtures=fixtures, n_matches=len(active))
        except APIFootballError as exc:
            log.warning("get_today_view fetch failed: %s", exc)
    elif refresh and should_poll_api_football():
        _refresh_fixture_statuses(force=True)
        if _all_today_fixtures and schedule and schedule.date == today:
            fixtures = list(_all_today_fixtures)

    live_doc = _load_live_predictions()
    live_by_teams = {
        (m.get("home"), m.get("away")): m
        for m in (live_doc.get("matches") or {}).values()
    }

    matches = []
    for f in fixtures:
        fix = f["fixture"]
        fid = fix["id"]
        home = f["teams"]["home"]
        away = f["teams"]["away"]
        ml_home = resolve_team_name(home["name"])
        ml_away = resolve_team_name(away["name"])
        status = cached_status.get(fid) or fix["status"]["short"]
        goals = f.get("goals") or {}
        live_state = get_live_state(fid)
        live_pred = live_by_teams.get((ml_home, ml_away))

        entry: dict = {
            "fixture_id": fid,
            "kickoff": fix["date"],
            "status": status,
            "elapsed": fix["status"].get("elapsed") or (live_state.elapsed if live_state else None),
            "home": {"id": home["id"], "name": home["name"]},
            "away": {"id": away["id"], "name": away["name"]},
            "ml_home": ml_home,
            "ml_away": ml_away,
            "score": {"home": goals.get("home"), "away": goals.get("away")},
            "venue": fix.get("venue") or {},
            "round": (f.get("league") or {}).get("round", ""),
            "is_live": status in LIVE_STATUSES,
        }

        src = live_pred or (live_state and {
            "minute": live_state.elapsed,
            "adj_lambda_home": live_state.adj_lambda_home,
            "adj_lambda_away": live_state.adj_lambda_away,
            "probabilities": live_state.probabilities,
            "momentum": live_state.momentum,
            "confidence": live_state.confidence,
            "xg_proxy": {"home": live_state.live_xg_home, "away": live_state.live_xg_away},
        })

        if src and entry["is_live"]:
            probs = src.get("probabilities") or {}
            display_stats = extract_display_stats(live_pred, live_state, fid)
            entry["live"] = {
                "elapsed": src.get("minute") or src.get("elapsed"),
                "adj_lambda_home": src.get("adj_lambda_home"),
                "adj_lambda_away": src.get("adj_lambda_away"),
                "adj_score_home": max(0, round(float(src.get("adj_lambda_home") or 0))),
                "adj_score_away": max(0, round(float(src.get("adj_lambda_away") or 0))),
                "probabilities": probs,
                "momentum": src.get("momentum"),
                "confidence": src.get("confidence"),
                "over_under": src.get("over_under"),
                "next_goal": src.get("next_goal"),
                **display_stats,
                "last_updated": live_state.last_updated if live_state else live_doc.get("updated_at"),
            }

        matches.append(entry)

    from future_fixture_predictions import attach_ml_predictions_to_today_matches

    attach_ml_predictions_to_today_matches(matches, fixtures)

    from today_kalshi_linker import attach_kalshi_links_to_today_matches

    attach_kalshi_links_to_today_matches(matches)

    matches.sort(key=_today_match_sort_key)

    live_count = sum(1 for m in matches if m["is_live"])
    finished_count = sum(
        1 for m in matches if (m.get("status") or "").upper() in FINAL_STATUSES
    )
    upcoming_count = len(matches) - live_count - finished_count

    return {
        "date": today,
        "local_date": today,
        "local_timezone": display_timezone_label(),
        "matches": matches,
        "n_matches": len(matches),
        "live_count": live_count,
        "finished_count": finished_count,
        "upcoming_count": max(0, upcoming_count),
        "active_count": live_count + max(0, upcoming_count),
        "api_budget_remaining": calls_remaining(),
        "daily_limit": DAILY_LIMIT,
        "scheduler_active": is_scheduler_running(),
        "live_meta": live_doc.get("live_meta") or {},
    }
