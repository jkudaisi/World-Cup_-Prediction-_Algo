"""Unified feature engineering for pre-match, live, and training rows."""

from __future__ import annotations

import math
from typing import Any


def elo_prob(ea: float, eb: float) -> float:
    return 1 / (1 + 10 ** ((eb - ea) / 400))


def _team_stats():
    from wc2026_ml_pipeline import TEAM_STATS
    return TEAM_STATS


# Re-export base team feature builder
def build_team_features(
    home: str,
    away: str,
    *,
    home_stats: dict[str, Any] | None = None,
    away_stats: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Pre-match features from team database (Elo, xG, form, etc.)."""
    from src.ratings.extended_team_stats import get_team_stats

    h = home_stats if home_stats is not None else get_team_stats(home)
    a = away_stats if away_stats is not None else get_team_stats(away)
    ep_h = elo_prob(h["elo"], a["elo"])
    ep_a = 1 - ep_h
    atk_h = h["xg"] / max(a["xga"], 0.5)
    atk_a = a["xg"] / max(h["xga"], 0.5)
    def_h = 1.0 / max(h["xga"], 0.5)
    def_a = 1.0 / max(a["xga"], 0.5)
    return {
        "elo_h": h["elo"],
        "elo_a": a["elo"],
        "elo_diff": h["elo"] - a["elo"],
        "elo_prob_h": ep_h,
        "elo_prob_a": ep_a,
        "xg_h": h["xg"],
        "xg_a": a["xg"],
        "xga_h": h["xga"],
        "xga_a": a["xga"],
        "xg_diff": h["xg"] - a["xg"],
        "xg_net_h": h["xg"] - a["xga"],
        "xg_net_a": a["xg"] - h["xga"],
        "rank_h": h["rank"],
        "rank_a": a["rank"],
        "rank_diff": a["rank"] - h["rank"],
        "wc_apps_h": h["wc_apps"],
        "wc_apps_a": a["wc_apps"],
        "wc_apps_diff": h["wc_apps"] - a["wc_apps"],
        "titles_h": h["titles"],
        "titles_a": a["titles"],
        "form_h": h["form"],
        "form_a": a["form"],
        "form_diff": h["form"] - a["form"],
        "sqval_h": h["sq_val"],
        "sqval_a": a["sq_val"],
        "sqval_ratio": h["sq_val"] / max(a["sq_val"], 1),
        "yc_h": h["yc"],
        "yc_a": a["yc"],
        "rc_h": h["rc"],
        "rc_a": a["rc"],
        "press_h": h["press"],
        "press_a": a["press"],
        "press_diff": h["press"] - a["press"],
        "dribble_h": h["dribble"],
        "dribble_a": a["dribble"],
        "aerial_h": h["aerial"],
        "aerial_a": a["aerial"],
        "attacking_strength_h": atk_h,
        "attacking_strength_a": atk_a,
        "defensive_weakness_h": def_h,
        "defensive_weakness_a": def_a,
        "tournament_exp_h": min(h["wc_apps"] / 20.0, 1.0),
        "tournament_exp_a": min(a["wc_apps"] / 20.0, 1.0),
        "lambda_h": max(0.3, min(4.5, h["xg"] * (a["xga"] / 1.2) * ep_h * 2.0)),
        "lambda_a": max(0.3, min(4.5, a["xg"] * (h["xga"] / 1.2) * ep_a * 2.0)),
    }


def build_features(
    home: str,
    away: str,
    context: dict[str, Any] | None = None,
    *,
    home_stats: dict[str, Any] | None = None,
    away_stats: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Pre-match features + optional match context."""
    feats = build_team_features(home, away, home_stats=home_stats, away_stats=away_stats)
    ctx = context or {}
    feats["neutral_venue"] = float(ctx.get("neutral_venue", 1.0))
    feats["knockout_stage"] = float(ctx.get("knockout_stage", 0.0))
    feats["rest_days_h"] = float(ctx.get("rest_days_h", 4.0))
    feats["rest_days_a"] = float(ctx.get("rest_days_a", 4.0))
    feats["rest_days_diff"] = feats["rest_days_h"] - feats["rest_days_a"]
    return feats


def _int_or(val: Any, default: int = 0) -> int:
    try:
        return int(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def _float_or(val: Any, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        if isinstance(val, str):
            val = val.replace("%", "").strip()
            if not val:
                return default
        return float(val)
    except (TypeError, ValueError):
        return default


def _parse_possession(val: Any) -> float:
    v = _float_or(val, 0.5)
    return v / 100.0 if v > 1 else v


def _count_penalties(events: list[dict], team_id: int | None) -> int:
    if not events or team_id is None:
        return 0
    n = 0
    for ev in events:
        if ev.get("type") != "Goal":
            continue
        detail = (ev.get("detail") or "").lower()
        if "penalty" in detail and (ev.get("team") or {}).get("id") == team_id:
            n += 1
    return n


def calc_xg_proxy(
    stats: dict[str, Any],
    events: list[dict] | None = None,
    team_id: int | None = None,
) -> float:
    """Improved xG proxy; uses API expected_goals when present."""
    api_xg = stats.get("expected_goals")
    if api_xg is not None:
        try:
            return max(0.0, float(str(api_xg)))
        except (TypeError, ValueError):
            pass
    sot = _int_or(stats.get("shots_on_goal"))
    off = _int_or(stats.get("shots_off_goal"))
    blocked = _int_or(stats.get("shots_blocked"))
    total = _int_or(stats.get("shots_total")) or (sot + off + blocked)
    corners = _int_or(stats.get("corner_kicks"))
    penalties = _count_penalties(events or [], team_id)
    big_chances = _int_or(stats.get("big_chances"))
    return (
        sot * 0.32
        + off * 0.06
        + blocked * 0.03
        + corners * 0.04
        + big_chances * 0.30
        + penalties * 0.76
    )


def calc_xg_pair(
    home_stats: dict,
    away_stats: dict,
    events: list[dict] | None = None,
    home_id: int | None = None,
    away_id: int | None = None,
) -> dict[str, float]:
    xg_h = calc_xg_proxy(home_stats, events, home_id)
    xg_a = calc_xg_proxy(away_stats, events, away_id)
    total_h = max(_int_or(home_stats.get("shots_total")), 1)
    total_a = max(_int_or(away_stats.get("shots_total")), 1)
    return {
        "home": round(xg_h, 3),
        "away": round(xg_a, 3),
        "diff": round(xg_h - xg_a, 3),
        "home_per_shot": round(xg_h / total_h, 3),
        "away_per_shot": round(xg_a / total_a, 3),
    }


def build_lineup_features(
    lineups: dict | None,
    players: dict | None,
) -> dict[str, float]:
    """Lineup strength proxies; safe defaults when data missing."""
    out = {
        "lineup_available": 0.0,
        "lineup_strength_h": 0.5,
        "lineup_strength_a": 0.5,
        "formation_known": 0.0,
        "avg_rating_h": 6.5,
        "avg_rating_a": 6.5,
        "missing_key_players_h": 0.0,
        "missing_key_players_a": 0.0,
    }
    if not lineups:
        return out
    out["lineup_available"] = 1.0
    for side, key in (("home", "lineup_strength_h"), ("away", "lineup_strength_a")):
        block = lineups.get(side) or {}
        xi = block.get("startXI") or []
        out[key] = min(1.0, len(xi) / 11.0) if xi else 0.5
        if block.get("formation"):
            out["formation_known"] = 1.0
    if players:
        for side, rk in (("home", "avg_rating_h"), ("away", "avg_rating_a")):
            plist = players.get(side) or []
            ratings = []
            for pblock in plist:
                for pl in (pblock.get("players") or [] if isinstance(pblock, dict) else []):
                    rating = (pl.get("statistics") or [{}])[0].get("games", {}).get("rating")
                    if rating:
                        try:
                            ratings.append(float(rating))
                        except (TypeError, ValueError):
                            pass
            if ratings:
                out[rk] = sum(ratings) / len(ratings)
    return out


def build_live_features(
    snapshot: dict[str, Any],
) -> dict[str, float]:
    """Live in-match differential features."""
    minute = _int_or(snapshot.get("minute"))
    score = snapshot.get("score") or {}
    sh, sa = _int_or(score.get("home")), _int_or(score.get("away"))
    stats = snapshot.get("stats") or {}
    hs, aws = stats.get("home") or {}, stats.get("away") or {}
    events = snapshot.get("events") or []
    home_id = snapshot.get("home_team_id")
    away_id = snapshot.get("away_team_id")
    xg = calc_xg_pair(hs, aws, events, home_id, away_id)

    def diff(hk, ak):
        return float(_int_or(hs.get(hk)) - _int_or(aws.get(ak or hk)))

    time_remaining = max(90 - minute, 0)
    return {
        "live_minute": float(minute),
        "live_time_remaining": float(time_remaining),
        "live_score_diff": float(sh - sa),
        "live_possession_diff": _parse_possession(hs.get("ball_possession")) - _parse_possession(aws.get("ball_possession")),
        "live_sot_diff": diff("shots_on_goal", "shots_on_goal"),
        "live_shots_diff": diff("shots_total", "shots_total"),
        "live_corners_diff": diff("corner_kicks", "corner_kicks"),
        "live_fouls_diff": diff("fouls", "fouls"),
        "live_yellow_diff": diff("yellow_cards", "yellow_cards"),
        "live_red_diff": diff("red_cards", "red_cards"),
        "live_saves_diff": diff("goalkeeper_saves", "goalkeeper_saves"),
        "live_pass_acc_diff": _float_or(hs.get("passes_pct"), 50) - _float_or(aws.get("passes_pct"), 50),
        "live_xg_diff": xg["diff"],
        "live_xg_home": xg["home"],
        "live_xg_away": xg["away"],
        "live_xg_home_per_shot": xg["home_per_shot"],
        "live_xg_away_per_shot": xg["away_per_shot"],
        "live_home_red_cards": float(_int_or(hs.get("red_cards"))),
        "live_away_red_cards": float(_int_or(aws.get("red_cards"))),
        "live_home_score": float(sh),
        "live_away_score": float(sa),
    }


def score_data_quality(data: dict[str, Any]) -> dict[str, Any]:
    """0–1 quality score from available API fields."""
    checks = {
        "stats": bool(data.get("stats") and (data["stats"].get("home") or data["stats"].get("away"))),
        "events": bool(data.get("events")),
        "lineups": bool(data.get("lineups")),
        "players": bool(data.get("players")),
        "odds": bool(data.get("odds")),
        "score": data.get("score") is not None or data.get("goals") is not None,
    }
    weights = {"stats": 0.35, "events": 0.20, "lineups": 0.15, "players": 0.10, "odds": 0.10, "score": 0.10}
    score = sum(weights[k] for k, ok in checks.items() if ok)
    missing = [k for k, ok in checks.items() if not ok]
    return {
        "score": round(score, 3),
        "checks": checks,
        "missing": missing,
    }


def _recent_event_weight(events: list[dict], home_id: int | None, away_id: int | None,
                         window_minutes: int = 15) -> dict[str, float]:
    weights = {"home": 0.0, "away": 0.0}
    if not events:
        return weights
    latest = max((ev.get("time") or {}).get("elapsed") or 0 for ev in events)
    cutoff = max(0, latest - window_minutes)
    for ev in events:
        elapsed = (ev.get("time") or {}).get("elapsed") or 0
        if elapsed < cutoff:
            continue
        tid = (ev.get("team") or {}).get("id")
        side = "home" if tid == home_id else "away" if tid == away_id else None
        if side is None:
            continue
        ev_type = ev.get("type") or ""
        detail = (ev.get("detail") or "").lower()
        if ev_type == "Goal":
            weights[side] += 12.0
        elif ev_type == "Card" and "red" in detail:
            weights["away" if side == "home" else "home"] += 8.0
            weights[side] -= 6.0
        elif ev_type == "Card":
            weights[side] -= 1.0
        elif ev_type == "subst":
            weights[side] += 0.5
    return weights


def calc_momentum(
    home_stats: dict[str, Any],
    away_stats: dict[str, Any],
    events: list[dict] | None = None,
    home_id: int | None = None,
    away_id: int | None = None,
) -> dict[str, float]:
    hs, aws = home_stats or {}, away_stats or {}
    home_raw = (
        _int_or(hs.get("shots_on_goal")) * 3.0
        + _int_or(hs.get("shots_total")) * 0.8
        + _parse_possession(hs.get("ball_possession")) * 25.0
        + _int_or(hs.get("corner_kicks")) * 1.5
        + _int_or(hs.get("dangerous_attacks")) * 0.5
        - _int_or(hs.get("red_cards")) * 15.0
        - _int_or(hs.get("yellow_cards")) * 2.0
    )
    away_raw = (
        _int_or(aws.get("shots_on_goal")) * 3.0
        + _int_or(aws.get("shots_total")) * 0.8
        + _parse_possession(aws.get("ball_possession")) * 25.0
        + _int_or(aws.get("corner_kicks")) * 1.5
        + _int_or(aws.get("dangerous_attacks")) * 0.5
        - _int_or(aws.get("red_cards")) * 15.0
        - _int_or(aws.get("yellow_cards")) * 2.0
    )
    if events:
        recent = _recent_event_weight(events, home_id, away_id)
        home_raw += recent["home"]
        away_raw += recent["away"]
    total = home_raw + away_raw
    if total <= 0:
        return {"home": 50.0, "away": 50.0}
    return {"home": round(100.0 * home_raw / total, 1), "away": round(100.0 * away_raw / total, 1)}


def sample_weight_for_row(row: dict[str, Any]) -> float:
    """Training sample weight: WC matches weighted higher, capped."""
    source = row.get("source", "historical")
    if source == "world_cup" or row.get("fixture_id"):
        return min(1.5, 1.0 + 0.1 * min(row.get("wc_match_index", 1), 5))
    if source == "recent_international":
        return 1.3
    return 1.0


def get_feature_cols() -> list[str]:
    teams = list(_team_stats().keys())
    return list(build_features(teams[0], teams[1]).keys())


def feature_defaults() -> dict[str, float]:
    """Safe defaults for optional/context columns."""
    teams = list(_team_stats().keys())
    return build_features(teams[0], teams[1])


def sanitize_training_frame(df: "pd.DataFrame", feature_cols: list[str] | None = None) -> "pd.DataFrame":
    """Ensure all feature columns exist with finite values (no NaN/inf)."""
    import numpy as np
    import pandas as pd

    if feature_cols is None:
        feature_cols = get_feature_cols()
    defaults = feature_defaults()
    out = df.copy()
    for col in feature_cols:
        if col not in out.columns:
            out[col] = defaults.get(col, 0.0)
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(defaults.get(col, 0.0))
    out = out.replace([np.inf, -np.inf], 0.0)
    keep = feature_cols + [c for c in ("goals_h", "goals_a") if c in out.columns]
    return out[keep]
