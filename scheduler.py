"""Smart daily scheduler: morning init + 20s live prediction loop."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from apifootball_client import (
    APIFootballError,
    calls_remaining,
    get_today_fixtures,
)
from incremental_trainer import mark_fixture_for_training, process_pending_training
from live_call_manager import REFRESH_INTERVAL
from live_updater import mark_fixture_final, run_live_cycle
from team_names import resolve_team_name
from training_store import load_training_state

log = logging.getLogger(__name__)

DAILY_BUDGET = 7500
RESERVE_CALLS = 5
SKIP_STATUSES = frozenset({"FT", "AET", "PEN", "PST", "CANC", "TBD"})
FINAL_STATUSES = frozenset({"FT", "AET", "PEN"})
LIVE_STATUSES = frozenset({"1H", "HT", "2H", "ET", "P", "LIVE", "BT"})

schedule: "DaySchedule | None" = None
cached_status: dict[int, str] = {}
_scheduler_thread: threading.Thread | None = None
_today_view_fetched: str = ""
_all_today_fixtures: list[dict] = []
_last_incremental_check: datetime | None = None
_last_status_refresh: datetime | None = None
INCREMENTAL_CHECK_INTERVAL = timedelta(minutes=5)
STATUS_REFRESH_INTERVAL = timedelta(seconds=60)


@dataclass
class DaySchedule:
    date: str
    fixtures: list[dict]
    n_matches: int
    live_cycles: int = 0
    calls_used: int = 0


def parse_kickoff(fixture: dict) -> datetime:
    raw = fixture["fixture"]["date"]
    if raw.endswith("Z"):
        raw = raw.replace("Z", "+00:00")
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


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


def _refresh_fixture_statuses() -> None:
    """Detect FT transitions (~1 API call/minute max)."""
    global _last_status_refresh
    now = datetime.now(timezone.utc)
    if _last_status_refresh and (now - _last_status_refresh) < STATUS_REFRESH_INTERVAL:
        return
    _last_status_refresh = now
    today = date.today().strftime("%Y-%m-%d")
    try:
        fixtures = get_today_fixtures(today)
    except APIFootballError as exc:
        log.warning("Status refresh failed: %s", exc)
        return
    wc = [f for f in fixtures if f.get("league", {}).get("id") == 1]
    _all_today_fixtures[:] = wc
    for f in wc:
        fid = f["fixture"]["id"]
        status = f["fixture"]["status"]["short"]
        old = cached_status.get(fid)
        cached_status[fid] = status
        if status in FINAL_STATUSES:
            if old not in FINAL_STATUSES:
                log.info("Match %s finished (%s) — queued for incremental training", fid, status)
                mark_fixture_for_training(fid)
                mark_fixture_final(fid)
            elif old is None:
                state = load_training_state()
                if fid not in state.get("trained_fixture_ids", []):
                    mark_fixture_for_training(fid)


def morning_init() -> DaySchedule | None:
    global _all_today_fixtures
    today = date.today().strftime("%Y-%m-%d")
    try:
        fixtures = get_today_fixtures(today)
    except APIFootballError as exc:
        log.error("morning_init failed: %s", exc)
        return None

    wc_fixtures = [f for f in fixtures if f.get("league", {}).get("id") == 1]
    _all_today_fixtures = list(wc_fixtures)
    trained = set(load_training_state().get("trained_fixture_ids", []))
    for f in wc_fixtures:
        fid = f["fixture"]["id"]
        cached_status[fid] = f["fixture"]["status"]["short"]
        if f["fixture"]["status"]["short"] in FINAL_STATUSES and fid not in trained:
            mark_fixture_for_training(fid)

    active = [f for f in wc_fixtures if f["fixture"]["status"]["short"] not in SKIP_STATUSES]
    log.info("Day init: %s WC fixtures today (%s active)", len(wc_fixtures), len(active))
    return DaySchedule(date=today, fixtures=wc_fixtures, n_matches=len(active))


def _any_match_window() -> bool:
    now = datetime.now(timezone.utc)
    for f in _all_today_fixtures:
        status = cached_status.get(f["fixture"]["id"]) or f["fixture"]["status"]["short"]
        if status in LIVE_STATUSES:
            return True
        if status in SKIP_STATUSES:
            continue
        kickoff = parse_kickoff(f)
        if kickoff <= now <= kickoff + timedelta(minutes=103):
            return True
    return False


def run_scheduler() -> None:
    global schedule
    while True:
        now_date = date.today().strftime("%Y-%m-%d")

        if schedule is None or schedule.date != now_date:
            log.info("Running morning init for %s", now_date)
            schedule = morning_init()
            if schedule is None:
                log.info("Init failed — sleeping 30 min")
                time.sleep(1800)
                continue

        _refresh_fixture_statuses()

        if calls_remaining() > RESERVE_CALLS and _any_match_window():
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
            except Exception as exc:
                log.exception("Live cycle failed: %s", exc)
            _maybe_run_incremental_training()
            time.sleep(REFRESH_INTERVAL)
        else:
            _maybe_run_incremental_training()
            time.sleep(60)


def start_scheduler() -> None:
    global _scheduler_thread
    if _scheduler_thread is not None and _scheduler_thread.is_alive():
        return
    _scheduler_thread = threading.Thread(
        target=run_scheduler, daemon=True, name="wc2026-scheduler",
    )
    _scheduler_thread.start()
    log.info("Scheduler thread started (live poll every %ss)", REFRESH_INTERVAL)


def is_scheduler_running() -> bool:
    return _scheduler_thread is not None and _scheduler_thread.is_alive()


def get_scheduler_status() -> dict:
    from apifootball_client import DAILY_LIMIT
    from live_updater import get_live_status
    live_status = get_live_status()
    return {
        "date": schedule.date if schedule else None,
        "n_matches": schedule.n_matches if schedule else 0,
        "live_poll_interval_seconds": REFRESH_INTERVAL,
        "live_cycles": schedule.live_cycles if schedule else 0,
        "calls_used_today": schedule.calls_used if schedule else 0,
        "api_budget_remaining": calls_remaining(),
        "daily_limit": DAILY_LIMIT,
        "active_fixture_ids": list(cached_status.keys()),
        "cached_statuses": dict(cached_status),
        "live_status": live_status,
    }


def get_today_view() -> dict:
    """Today's WC fixtures merged with live predictions."""
    global schedule, _today_view_fetched, _all_today_fixtures
    from apifootball_client import DAILY_LIMIT
    from config import APIFOOTBALL_KEY
    from live_trainer import get_live_state
    from live_updater import _load_live_predictions, extract_display_stats

    today = date.today().strftime("%Y-%m-%d")
    fixtures: list[dict] = []
    if _all_today_fixtures and schedule and schedule.date == today:
        fixtures = list(_all_today_fixtures)
    elif schedule and schedule.date == today:
        fixtures = list(schedule.fixtures)
    elif _today_view_fetched != today and APIFOOTBALL_KEY:
        try:
            raw = get_today_fixtures(today)
            fixtures = [f for f in raw if f.get("league", {}).get("id") == 1]
            _all_today_fixtures = list(fixtures)
            for f in fixtures:
                cached_status[f["fixture"]["id"]] = f["fixture"]["status"]["short"]
            _today_view_fetched = today
            if schedule is None or schedule.date != today:
                active = [f for f in fixtures if f["fixture"]["status"]["short"] not in SKIP_STATUSES]
                schedule = DaySchedule(date=today, fixtures=fixtures, n_matches=len(active))
        except APIFootballError as exc:
            log.warning("get_today_view fetch failed: %s", exc)

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

    matches.sort(key=lambda m: m["kickoff"])

    return {
        "date": today,
        "matches": matches,
        "n_matches": len(matches),
        "live_count": sum(1 for m in matches if m["is_live"]),
        "api_budget_remaining": calls_remaining(),
        "daily_limit": DAILY_LIMIT,
        "scheduler_active": is_scheduler_running(),
        "live_meta": live_doc.get("live_meta") or {},
    }
