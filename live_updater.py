"""Orchestrates live API polling, snapshots, predictions, and persistence."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from live_call_manager import (
    REFRESH_INTERVAL,
    clear_fixture_state,
    fetch_live_bundle,
    fetch_live_fixtures_list,
    should_refresh,
)
from live_predictor import calc_momentum, update_live_prediction_from_snapshot
from feature_builder import calc_xg_pair
from live_snapshot_store import append_snapshot, get_latest_snapshot, load_snapshots
from team_names import find_ml_match, resolve_team_name
from training_store import atomic_write_json, load_json

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
PREDICTIONS_PATH = ROOT / "predictions.json"
LIVE_PREDICTIONS_PATH = ROOT / "live_predictions.json"

LIVE_STATUSES = frozenset({"1H", "HT", "2H", "ET", "P", "LIVE", "BT"})
FINAL_STATUSES = frozenset({"FT", "AET", "PEN"})

_live_lock = threading.Lock()
_last_cycle_at: str | None = None
_last_cycle_stats: dict[str, Any] = {}


def _load_predictions() -> dict:
    return load_json(PREDICTIONS_PATH, {"ml_data": [], "team_elo": {}, "stats": None})


def _load_live_predictions() -> dict:
    return load_json(LIVE_PREDICTIONS_PATH, {
        "updated_at": None,
        "live_meta": {},
        "matches": {},
    })


def _save_live_predictions(data: dict) -> None:
    atomic_write_json(LIVE_PREDICTIONS_PATH, data)


def _extract_score(fixture: dict) -> dict[str, int | None]:
    goals = fixture.get("goals") or {}
    return {"home": goals.get("home"), "away": goals.get("away")}


def _extract_minute(fixture: dict, events: list[dict]) -> int:
    fix = fixture.get("fixture") or {}
    elapsed = (fix.get("status") or {}).get("elapsed")
    if elapsed is not None:
        return int(elapsed)
    if events:
        return max((ev.get("time") or {}).get("elapsed") or 0 for ev in events)
    return 0


def _build_snapshot_record(
    fixture: dict,
    bundle: dict[str, Any],
    prediction: dict[str, Any],
) -> dict[str, Any]:
    fix = fixture["fixture"]
    fid = fix["id"]
    home = fixture["teams"]["home"]
    away = fixture["teams"]["away"]
    stats = bundle.get("stats") or {}
    hs, aws = stats.get("home") or {}, stats.get("away") or {}
    events = bundle.get("events") or []
    minute = _extract_minute(fixture, events)
    status = (fix.get("status") or {}).get("short") or "NS"
    score = _extract_score(fixture)
    score_int = {
        "home": score["home"] if score["home"] is not None else 0,
        "away": score["away"] if score["away"] is not None else 0,
    }

    xg_proxy = calc_xg_pair(hs, aws, events, home.get("id"), away.get("id"))
    momentum = calc_momentum(hs, aws, events, home.get("id"), away.get("id"))

    return {
        "fixture_id": fid,
        "snapshot_time": datetime.now(timezone.utc).isoformat(),
        "minute": minute,
        "status": status,
        "score": score_int,
        "stats": stats,
        "events": events,
        "lineups": bundle.get("lineups") or {},
        "players": bundle.get("players") or {},
        "xg_proxy": xg_proxy,
        "momentum": momentum,
        "api_calls_used": bundle.get("api_calls_used", 0),
        "prediction": prediction,
        "home_team_id": home.get("id"),
        "away_team_id": away.get("id"),
        "home_name": home.get("name"),
        "away_name": away.get("name"),
    }


def _patch_ml_data(pred: dict, home_ml: str, away_ml: str, live_pred: dict) -> None:
    match = find_ml_match(home_ml, away_ml, pred.get("ml_data", []))
    if not match:
        return
    stats = live_pred.get("_stats") or {}
    hs, aws = stats.get("home") or {}, stats.get("away") or {}
    match["live_status"] = "live"
    match["live_elapsed"] = live_pred.get("minute")
    match["live_last_updated"] = datetime.now(timezone.utc).isoformat()
    match["live_adj_lambda_h"] = live_pred.get("adj_lambda_home")
    match["live_adj_lambda_a"] = live_pred.get("adj_lambda_away")
    match["live_probabilities"] = live_pred.get("probabilities")
    match["live_momentum"] = live_pred.get("momentum")
    match["live_confidence"] = live_pred.get("confidence")
    match["live_score"] = live_pred.get("score")
    match["live_over_under"] = live_pred.get("over_under")
    match["live_next_goal"] = live_pred.get("next_goal")
    match["live_stats"] = {
        "home_sot": int(hs.get("shots_on_goal") or 0),
        "away_sot": int(aws.get("shots_on_goal") or 0),
        "home_corners": int(hs.get("corner_kicks") or 0),
        "away_corners": int(aws.get("corner_kicks") or 0),
        "home_possession": _parse_poss(hs.get("ball_possession")),
        "away_possession": _parse_poss(aws.get("ball_possession")),
        "home_yellow_cards": int(hs.get("yellow_cards") or 0),
        "away_yellow_cards": int(aws.get("yellow_cards") or 0),
        "home_red_cards": int(hs.get("red_cards") or 0),
        "away_red_cards": int(aws.get("red_cards") or 0),
        "xg_proxy_home": live_pred.get("xg_proxy", {}).get("home"),
        "xg_proxy_away": live_pred.get("xg_proxy", {}).get("away"),
    }


def _parse_poss(val: Any) -> float:
    if val is None:
        return 0.5
    if isinstance(val, str) and val.endswith("%"):
        try:
            return float(val.replace("%", "").strip()) / 100.0
        except ValueError:
            return 0.5
    try:
        v = float(val)
        return v / 100.0 if v > 1 else v
    except (TypeError, ValueError):
        return 0.5


def _int_stat(val: Any) -> int:
    try:
        return int(val) if val is not None else 0
    except (TypeError, ValueError):
        return 0


def extract_display_stats(
    src: dict[str, Any] | None,
    live_state: Any = None,
    fixture_id: int | None = None,
) -> dict[str, Any]:
    """Build Today-tab stat fields from live_state, live_pred._stats, or latest snapshot."""
    if live_state is not None:
        return {
            "home_sot": live_state.home_shots_on_target,
            "away_sot": live_state.away_shots_on_target,
            "home_corners": live_state.home_corners,
            "away_corners": live_state.away_corners,
            "home_possession": live_state.home_possession,
            "away_possession": live_state.away_possession,
            "home_yellow_cards": live_state.home_yellow_cards,
            "away_yellow_cards": live_state.away_yellow_cards,
            "home_red_cards": live_state.home_red_cards,
            "away_red_cards": live_state.away_red_cards,
            "xg_proxy_home": round(live_state.live_xg_home, 3),
            "xg_proxy_away": round(live_state.live_xg_away, 3),
        }

    stats = (src or {}).get("_stats") or {}
    hs, aws = stats.get("home") or {}, stats.get("away") or {}

    if not hs and not aws and fixture_id is not None:
        snap = get_latest_snapshot(fixture_id)
        if snap:
            stats = snap.get("stats") or {}
            hs, aws = stats.get("home") or {}, stats.get("away") or {}
            xg = snap.get("xg_proxy") or {}
        else:
            xg = (src or {}).get("xg_proxy") or {}
    else:
        xg = (src or {}).get("xg_proxy") or {}

    if not hs and not aws:
        ls = (src or {}).get("live_stats") or {}
        if ls:
            return {
                "home_sot": ls.get("home_sot", 0),
                "away_sot": ls.get("away_sot", 0),
                "home_corners": ls.get("home_corners", 0),
                "away_corners": ls.get("away_corners", 0),
                "home_possession": ls.get("home_possession", 0.5),
                "away_possession": ls.get("away_possession", 0.5),
                "home_yellow_cards": ls.get("home_yellow_cards", 0),
                "away_yellow_cards": ls.get("away_yellow_cards", 0),
                "home_red_cards": ls.get("home_red_cards", 0),
                "away_red_cards": ls.get("away_red_cards", 0),
                "xg_proxy_home": ls.get("xg_proxy_home", xg.get("home", 0)),
                "xg_proxy_away": ls.get("xg_proxy_away", xg.get("away", 0)),
            }
        return {
            "home_sot": 0, "away_sot": 0,
            "home_corners": 0, "away_corners": 0,
            "home_possession": 0.5, "away_possession": 0.5,
            "home_yellow_cards": 0, "away_yellow_cards": 0,
            "home_red_cards": 0, "away_red_cards": 0,
            "xg_proxy_home": xg.get("home", 0), "xg_proxy_away": xg.get("away", 0),
        }

    return {
        "home_sot": _int_stat(hs.get("shots_on_goal")),
        "away_sot": _int_stat(aws.get("shots_on_goal")),
        "home_corners": _int_stat(hs.get("corner_kicks")),
        "away_corners": _int_stat(aws.get("corner_kicks")),
        "home_possession": _parse_poss(hs.get("ball_possession")),
        "away_possession": _parse_poss(aws.get("ball_possession")),
        "home_yellow_cards": _int_stat(hs.get("yellow_cards")),
        "away_yellow_cards": _int_stat(aws.get("yellow_cards")),
        "home_red_cards": _int_stat(hs.get("red_cards")),
        "away_red_cards": _int_stat(aws.get("red_cards")),
        "xg_proxy_home": xg.get("home") if xg else (src or {}).get("xg_proxy", {}).get("home", 0),
        "xg_proxy_away": xg.get("away") if xg else (src or {}).get("xg_proxy", {}).get("away", 0),
    }


def process_fixture_live(
    fixture: dict,
    *,
    force: bool = False,
) -> dict[str, Any] | None:
    """Fetch, snapshot, and predict for one live fixture."""
    fix = fixture["fixture"]
    fid = fix["id"]
    home = fixture["teams"]["home"]
    away = fixture["teams"]["away"]
    home_ml = resolve_team_name(home.get("name", ""))
    away_ml = resolve_team_name(away.get("name", ""))

    if not force and not should_refresh(fid):
        latest = get_latest_snapshot(fid)
        if latest and latest.get("prediction"):
            return latest["prediction"]
        return None

    bundle = fetch_live_bundle(fid, home.get("id"), away.get("id"), force=force)
    if bundle is None:
        return None

    from future_fixture_predictions import load_merged_ml_data

    merged_ml = load_merged_ml_data()
    base = find_ml_match(home_ml, away_ml, merged_ml) or {}

    snapshot_input = {
        "fixture_id": fid,
        "minute": _extract_minute(fixture, bundle.get("events") or []),
        "status": (fix.get("status") or {}).get("short"),
        "score": _extract_score(fixture),
        "stats": bundle.get("stats"),
        "events": bundle.get("events"),
        "home_team_id": home.get("id"),
        "away_team_id": away.get("id"),
    }
    snapshot_input["score"] = {
        "home": snapshot_input["score"]["home"] or 0,
        "away": snapshot_input["score"]["away"] or 0,
    }

    live_pred = update_live_prediction_from_snapshot(snapshot_input, base)
    live_pred["_stats"] = bundle.get("stats")

    record = _build_snapshot_record(fixture, bundle, live_pred)
    append_snapshot(fid, record)

    pred_file = _load_predictions()
    if find_ml_match(home_ml, away_ml, pred_file.get("ml_data", [])):
        _patch_ml_data(pred_file, home_ml, away_ml, live_pred)
        atomic_write_json(PREDICTIONS_PATH, pred_file)

    return live_pred


def _wc_fixtures_in_play(live_fixtures: list[dict]) -> list[dict]:
    """WC fixtures in LIVE status from live=all plus scheduler today list."""
    wc_live = [
        f for f in live_fixtures
        if f.get("league", {}).get("id") == 1
        and (f.get("fixture", {}).get("status", {}).get("short") in LIVE_STATUSES)
    ]
    seen = {f["fixture"]["id"] for f in wc_live}
    try:
        import scheduler as sched
        for f in sched.get_in_play_wc_fixtures():
            fid = f["fixture"]["id"]
            if f.get("league", {}).get("id") == 1 and fid not in seen:
                wc_live.append(f)
                seen.add(fid)
    except Exception as exc:
        log.debug("Scheduler WC live supplement skipped: %s", exc)
    return wc_live


def run_live_cycle(*, force: bool = False) -> dict[str, Any]:
    """Poll live fixtures, update snapshots and live_predictions.json."""
    global _last_cycle_at, _last_cycle_stats

    with _live_lock:
        live_fixtures = fetch_live_fixtures_list()
        wc_live = _wc_fixtures_in_play(live_fixtures)

        matches: dict[str, Any] = {}
        calls_used = 1 if live_fixtures else 0
        updated = 0

        for fx in wc_live:
            fid = fx["fixture"]["id"]
            try:
                live_pred = process_fixture_live(fx, force=force)
                if live_pred:
                    home_ml = resolve_team_name(fx["teams"]["home"]["name"])
                    away_ml = resolve_team_name(fx["teams"]["away"]["name"])
                    status = (fx.get("fixture", {}).get("status", {}) or {}).get("short") or "LIVE"
                    matches[str(fid)] = {
                        **live_pred,
                        "fixture_id": fid,
                        "status": status,
                        "home": home_ml,
                        "away": away_ml,
                        "home_api": fx["teams"]["home"]["name"],
                        "away_api": fx["teams"]["away"]["name"],
                        "live_stats": extract_display_stats(live_pred, fixture_id=fid),
                    }
                    updated += 1
                    latest = get_latest_snapshot(fid)
                    if latest:
                        calls_used += latest.get("api_calls_used", 0)
            except Exception as exc:
                log.exception("Live update failed for fixture %s: %s", fid, exc)

        now = datetime.now(timezone.utc).isoformat()
        live_doc = _load_live_predictions()
        # Keep only fixtures currently in play — never accumulate stale finished matches.
        live_doc["matches"] = matches
        live_doc["updated_at"] = now
        live_doc["live_meta"] = {
            "fetched_at": now,
            "fixtures_live": len(wc_live),
            "fixtures_updated": updated,
            "poll_interval_seconds": REFRESH_INTERVAL,
            "api_calls_this_cycle": calls_used,
        }
        _save_live_predictions(live_doc)

        _last_cycle_at = now
        _last_cycle_stats = {
            "live_count": len(wc_live),
            "updated": updated,
            "calls_used": calls_used,
        }
        return {
            "live_count": len(wc_live),
            "updated": updated,
            "calls_used": calls_used,
            "fetched_at": now,
        }


def get_live_status() -> dict[str, Any]:
    live_doc = _load_live_predictions()
    return {
        "last_cycle_at": _last_cycle_at,
        "last_cycle": _last_cycle_stats,
        "live_meta": live_doc.get("live_meta") or {},
        "active_matches": len(live_doc.get("matches") or {}),
        "poll_interval_seconds": REFRESH_INTERVAL,
    }


def build_live_api_response() -> dict[str, Any]:
    """Merge base predictions with live overlay for GET /api/live."""
    pred = _load_predictions()
    live_doc = _load_live_predictions()
    live_matches = live_doc.get("matches") or {}

    for match in pred.get("ml_data", []):
        for live in live_matches.values():
            if live.get("home") != match.get("home") or live.get("away") != match.get("away"):
                continue
            status = (live.get("status") or "").upper()
            if status in FINAL_STATUSES:
                continue
            if status not in LIVE_STATUSES and not live.get("is_live"):
                continue
            match["live_probabilities"] = live.get("probabilities")
            match["live_momentum"] = live.get("momentum")
            match["live_confidence"] = live.get("confidence")
            match["live_adj_lambda_h"] = live.get("adj_lambda_home")
            match["live_adj_lambda_a"] = live.get("adj_lambda_away")
            match["live_elapsed"] = live.get("minute")
            match["live_status"] = "live" if status in LIVE_STATUSES else match.get("live_status")
            match["live_score"] = live.get("score")
            match["live_over_under"] = live.get("over_under")
            match["live_next_goal"] = live.get("next_goal")
            break

    return {
        **pred,
        "live_predictions": live_matches,
        "live_meta": live_doc.get("live_meta") or {},
        "live_updated_at": live_doc.get("updated_at"),
    }


def prune_stale_live_predictions() -> int:
    """Drop live prediction rows that are not actually in play on API-Football."""
    from apifootball_client import get_all_live_fixtures
    from team_names import resolve_team_name

    live_doc = _load_live_predictions()
    matches = live_doc.get("matches") or {}
    if not matches:
        return 0

    wc_live_teams: set[tuple[str, str]] = set()
    try:
        for f in get_all_live_fixtures():
            if f.get("league", {}).get("id") != 1:
                continue
            status = f["fixture"]["status"]["short"]
            if status not in LIVE_STATUSES:
                continue
            home = resolve_team_name(f["teams"]["home"]["name"])
            away = resolve_team_name(f["teams"]["away"]["name"])
            wc_live_teams.add((home, away))
    except Exception as exc:
        log.warning("Live prune API check failed, using status filter: %s", exc)
        wc_live_teams = None

    if wc_live_teams is not None:
        kept = {
            fid: row
            for fid, row in matches.items()
            if (row.get("home"), row.get("away")) in wc_live_teams
        }
    else:
        kept = {
            fid: row
            for fid, row in matches.items()
            if (row.get("status") or "").upper() in LIVE_STATUSES
        }

    removed = len(matches) - len(kept)
    if removed:
        live_doc["matches"] = kept
        _save_live_predictions(live_doc)
        log.info("Pruned %s stale live prediction(s)", removed)
    return removed


def mark_fixture_final(fixture_id: int) -> None:
    clear_fixture_state(fixture_id)
    live_doc = _load_live_predictions()
    live_doc.get("matches", {}).pop(str(fixture_id), None)
    _save_live_predictions(live_doc)
