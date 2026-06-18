"""Incremental live lambda updates from API-Football snapshots."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from feature_builder import calc_momentum, calc_xg_proxy
from live_predictor import update_live_prediction_from_snapshot
from live_snapshot_store import append_snapshot
from team_names import find_ml_match, resolve_team_name

log = logging.getLogger(__name__)

PREDICTIONS_PATH = Path(__file__).parent / "predictions.json"
_predictions_cache: dict | None = None

_match_states: dict[int, "LiveMatchState"] = {}


@dataclass
class LiveMatchState:
    fixture_id: int
    home_name: str
    away_name: str
    snapshots: list[dict] = field(default_factory=list)
    live_xg_home: float = 0.0
    live_xg_away: float = 0.0
    elapsed: int = 0
    home_red_cards: int = 0
    away_red_cards: int = 0
    home_yellow_cards: int = 0
    away_yellow_cards: int = 0
    home_corners: int = 0
    away_corners: int = 0
    home_shots_on_target: int = 0
    away_shots_on_target: int = 0
    home_total_shots: int = 0
    away_total_shots: int = 0
    home_possession: float = 0.5
    away_possession: float = 0.5
    adj_lambda_home: float | None = None
    adj_lambda_away: float | None = None
    probabilities: dict[str, float] | None = None
    momentum: dict[str, float] | None = None
    confidence: float | None = None
    last_updated: str = ""


def _load_predictions() -> dict:
    global _predictions_cache
    if _predictions_cache is None and PREDICTIONS_PATH.exists():
        with open(PREDICTIONS_PATH, encoding="utf-8") as f:
            _predictions_cache = json.load(f)
    return _predictions_cache or {"ml_data": []}


def _invalidate_predictions_cache() -> None:
    global _predictions_cache
    _predictions_cache = None


def _get_base_lambdas(home_name: str, away_name: str) -> dict[str, float]:
    data = _load_predictions()
    match = find_ml_match(home_name, away_name, data.get("ml_data", []))
    if match:
        models = match.get("models") or {}
        if models:
            rh = [m["rh"] for m in models.values()]
            ra = [m["ra"] for m in models.values()]
            return {"lambda_h": sum(rh) / len(rh), "lambda_a": sum(ra) / len(ra)}
    return {"lambda_h": 1.2, "lambda_a": 0.9}


def _parse_possession(val: Any) -> float:
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


def _int_or_zero(val: Any) -> int:
    try:
        return int(val) if val is not None else 0
    except (TypeError, ValueError):
        return 0


def _count_cards_from_events(events: list[dict], home_id: int | None, away_id: int | None) -> dict[str, int]:
    counts = {"home_yellow": 0, "away_yellow": 0, "home_red": 0, "away_red": 0}
    for ev in events:
        if ev.get("type") != "Card":
            continue
        detail = (ev.get("detail") or "").lower()
        tid = (ev.get("team") or {}).get("id")
        side = None
        if tid == home_id:
            side = "home"
        elif tid == away_id:
            side = "away"
        if side is None:
            continue
        if "red" in detail:
            counts[f"{side}_red"] += 1
        elif "yellow" in detail:
            counts[f"{side}_yellow"] += 1
    return counts


def _write_predictions_atomic(data: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=PREDICTIONS_PATH.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, PREDICTIONS_PATH)
        _invalidate_predictions_cache()
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def ingest_live_snapshot(
    fixture_id: int,
    data: dict,
    home_name: str = "",
    away_name: str = "",
    home_team_id: int | None = None,
    away_team_id: int | None = None,
    score_home: int | None = None,
    score_away: int | None = None,
    status: str = "1H",
) -> LiveMatchState:
    if fixture_id not in _match_states:
        _match_states[fixture_id] = LiveMatchState(
            fixture_id=fixture_id,
            home_name=resolve_team_name(home_name),
            away_name=resolve_team_name(away_name),
        )
    state = _match_states[fixture_id]
    if home_name:
        state.home_name = resolve_team_name(home_name)
    if away_name:
        state.away_name = resolve_team_name(away_name)

    events = data.get("events") or []
    if events:
        state.elapsed = max(
            (ev.get("time") or {}).get("elapsed") or 0 for ev in events
        )

    stats = data.get("stats") or {}
    hs = stats.get("home") or {}
    aws = stats.get("away") or {}

    state.home_shots_on_target = _int_or_zero(hs.get("shots_on_goal"))
    state.away_shots_on_target = _int_or_zero(aws.get("shots_on_goal"))
    state.home_total_shots = _int_or_zero(hs.get("shots_total"))
    state.away_total_shots = _int_or_zero(aws.get("shots_total"))
    state.home_corners = _int_or_zero(hs.get("corner_kicks"))
    state.away_corners = _int_or_zero(aws.get("corner_kicks"))
    state.home_possession = _parse_possession(hs.get("ball_possession"))
    state.away_possession = _parse_possession(aws.get("ball_possession"))

    card_counts = _count_cards_from_events(events, home_team_id, away_team_id)
    if events:
        state.home_yellow_cards = card_counts["home_yellow"]
        state.away_yellow_cards = card_counts["away_yellow"]
        state.home_red_cards = card_counts["home_red"]
        state.away_red_cards = card_counts["away_red"]
    else:
        state.home_yellow_cards = _int_or_zero(hs.get("yellow_cards"))
        state.away_yellow_cards = _int_or_zero(aws.get("yellow_cards"))
        state.home_red_cards = _int_or_zero(hs.get("red_cards"))
        state.away_red_cards = _int_or_zero(aws.get("red_cards"))

    xg_proxy_home = calc_xg_proxy(hs)
    xg_proxy_away = calc_xg_proxy(aws)
    state.live_xg_home = xg_proxy_home
    state.live_xg_away = xg_proxy_away
    momentum = calc_momentum(hs, aws, events, home_team_id, away_team_id)
    state.momentum = momentum

    pred = _load_predictions()
    base = find_ml_match(state.home_name, state.away_name, pred.get("ml_data", [])) or {}

    snapshot_input = {
        "fixture_id": fixture_id,
        "minute": state.elapsed,
        "status": status,
        "score": {"home": score_home or 0, "away": score_away or 0},
        "stats": stats,
        "events": events,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "xg_proxy": {"home": xg_proxy_home, "away": xg_proxy_away},
        "momentum": momentum,
    }
    live_pred = update_live_prediction_from_snapshot(snapshot_input, base)

    adj_lambda_home = live_pred["adj_lambda_home"]
    adj_lambda_away = live_pred["adj_lambda_away"]
    state.adj_lambda_home = adj_lambda_home
    state.adj_lambda_away = adj_lambda_away
    state.probabilities = live_pred.get("probabilities")
    conf = live_pred.get("confidence") or {}
    state.confidence = conf.get("score") if isinstance(conf, dict) else conf
    state.last_updated = datetime.now(timezone.utc).isoformat()

    record = {
        "fixture_id": fixture_id,
        "snapshot_time": state.last_updated,
        "minute": state.elapsed,
        "status": status,
        "score": snapshot_input["score"],
        "stats": stats,
        "events": events,
        "lineups": data.get("lineups") or {},
        "players": data.get("players") or {},
        "xg_proxy": {"home": xg_proxy_home, "away": xg_proxy_away},
        "momentum": momentum,
        "api_calls_used": data.get("api_calls_used", 2),
        "prediction": live_pred,
    }
    append_snapshot(fixture_id, record)

    state.snapshots.append(data)
    if len(state.snapshots) > 20:
        state.snapshots = state.snapshots[-20:]

    match = find_ml_match(state.home_name, state.away_name, pred.get("ml_data", []))
    if match:
        match["live_adj_lambda_h"] = adj_lambda_home
        match["live_adj_lambda_a"] = adj_lambda_away
        match["live_elapsed"] = state.elapsed
        match["live_status"] = "live"
        match["live_last_updated"] = state.last_updated
        match["live_probabilities"] = live_pred.get("probabilities")
        match["live_momentum"] = momentum
        match["live_confidence"] = live_pred.get("confidence")
        match["live_explanation"] = live_pred.get("explanation")
        match["live_score"] = live_pred.get("score")
        match["live_stats"] = {
            "home_sot": state.home_shots_on_target,
            "away_sot": state.away_shots_on_target,
            "home_corners": state.home_corners,
            "away_corners": state.away_corners,
            "home_possession": state.home_possession,
            "away_possession": state.away_possession,
            "home_yellow_cards": state.home_yellow_cards,
            "away_yellow_cards": state.away_yellow_cards,
            "home_red_cards": state.home_red_cards,
            "away_red_cards": state.away_red_cards,
            "xg_proxy_home": round(xg_proxy_home, 3),
            "xg_proxy_away": round(xg_proxy_away, 3),
        }

    _write_predictions_atomic(pred)
    return state


def get_live_state(fixture_id: int) -> LiveMatchState | None:
    return _match_states.get(fixture_id)


def get_all_live_states() -> dict[int, dict]:
    return {fid: asdict(st) for fid, st in _match_states.items()}
