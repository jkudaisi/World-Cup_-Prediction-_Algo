"""Live in-match probability adjustments using score-state logic and calibration."""

from __future__ import annotations

import math
from typing import Any

from calibration import calibrate_outcome_probs, compute_confidence, normalize_outcome_probs
from ensemble import (
    both_teams_score_prob,
    build_over_under_payload,
    outcome_probs_from_lambdas,
    over_under_prob,
    weighted_ensemble_goals,
)
from feature_builder import build_live_features, calc_momentum, calc_xg_pair, score_data_quality


def _int_or_zero(val: Any) -> int:
    try:
        return int(val) if val is not None else 0
    except (TypeError, ValueError):
        return 0


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _next_goal_prob(rem_lh: float, rem_la: float) -> dict[str, float]:
    total = rem_lh + rem_la
    if total <= 0:
        return {"home": 0.33, "away": 0.33, "none": 0.34}
    p_any = 1.0 - math.exp(-total)
    if p_any <= 0:
        return {"home": 0.0, "away": 0.0, "none": 1.0}
    return {
        "home": round(p_any * rem_lh / total, 4),
        "away": round(p_any * rem_la / total, 4),
        "none": round(1.0 - p_any, 4),
    }


def _base_lambdas_from_prediction(base_prediction: dict[str, Any]) -> tuple[float, float, dict[str, float]]:
    pred = base_prediction.get("prediction") or {}
    if pred.get("projected_home_goals") is not None:
        return (
            float(pred["projected_home_goals"]),
            float(pred["projected_away_goals"]),
            {
                "home_win": pred.get("home_win", 0.33),
                "draw": pred.get("draw", 0.33),
                "away_win": pred.get("away_win", 0.33),
            },
        )
    models = base_prediction.get("models") or {}
    if models:
        rh = sum(m["rh"] for m in models.values()) / len(models)
        ra = sum(m["ra"] for m in models.values()) / len(models)
        prematch = outcome_probs_from_lambdas(rh, ra, calibrate=True)
        return rh, ra, prematch
    return 1.2, 0.9, {"home_win": 0.40, "draw": 0.28, "away_win": 0.32}


def _apply_score_state_adjustments(
    probs: dict[str, float],
    cur_h: int,
    cur_a: int,
    minute: int,
    xg_diff: float,
    home_red: int,
    away_red: int,
) -> dict[str, float]:
    """Late-game and score-state nudges."""
    out = dict(probs)
    lead = cur_h - cur_a
    time_factor = min(minute / 90.0, 1.0)

    if lead > 0 and minute >= 60:
        boost = 0.04 + 0.06 * time_factor
        if lead == 1:
            out["draw"] += 0.03 * time_factor
            out["home_win"] += boost
            out["away_win"] -= boost + 0.03 * time_factor
        else:
            out["home_win"] += boost
            out["away_win"] -= boost * 0.8
            out["draw"] -= boost * 0.2
    elif lead < 0 and minute >= 60:
        boost = 0.04 + 0.06 * time_factor
        if lead == -1:
            out["draw"] += 0.03 * time_factor
            out["away_win"] += boost
            out["home_win"] -= boost + 0.03 * time_factor
        else:
            out["away_win"] += boost
            out["home_win"] -= boost * 0.8
            out["draw"] -= boost * 0.2
    elif lead == 0 and minute >= 70:
        out["draw"] += 0.05 * time_factor
        if xg_diff > 0.25:
            out["home_win"] += 0.03
            out["away_win"] -= 0.03
        elif xg_diff < -0.25:
            out["away_win"] += 0.03
            out["home_win"] -= 0.03

    if home_red > away_red:
        out["home_win"] -= 0.06 * home_red
        out["away_win"] += 0.05 * home_red
    elif away_red > home_red:
        out["away_win"] -= 0.06 * away_red
        out["home_win"] += 0.05 * away_red

    if lead < 0 and xg_diff > 0.35 and minute < 80:
        out["home_win"] += 0.04
        out["away_win"] -= 0.03
    elif lead > 0 and xg_diff < -0.35 and minute < 80:
        out["away_win"] += 0.04
        out["home_win"] -= 0.03

    return calibrate_outcome_probs(normalize_outcome_probs(out))


def _build_explanation(
    snapshot: dict,
    live_feats: dict,
    prematch: dict,
    blended: dict,
    base_prediction: dict,
) -> dict[str, list[str]]:
    top, risk = [], []
    minute = _int_or_zero(snapshot.get("minute"))
    sh = _int_or_zero((snapshot.get("score") or {}).get("home"))
    sa = _int_or_zero((snapshot.get("score") or {}).get("away"))

    if live_feats.get("live_xg_diff", 0) > 0.2:
        top.append("Live xG proxy favors home team")
    elif live_feats.get("live_xg_diff", 0) < -0.2:
        top.append("Live xG proxy favors away team")
    if sh > sa:
        top.append("Current score advantage home")
    elif sa > sh:
        top.append("Current score advantage away")
    if live_feats.get("live_home_red_cards", 0):
        top.append("Home red card")
    if live_feats.get("live_away_red_cards", 0):
        top.append("Away red card")
    if prematch.get("home_win", 0) > blended.get("home_win", 0) + 0.08:
        top.append("Live state reduced home win chance vs pre-match")
    elif blended.get("home_win", 0) > prematch.get("home_win", 0) + 0.08:
        top.append("Live stats boosted home win chance")

    dq = score_data_quality(snapshot)
    if dq["score"] < 0.5:
        risk.append("Limited API data quality")
    if not snapshot.get("lineups"):
        risk.append("Missing lineup data")
    stats = snapshot.get("stats") or {}
    if not stats.get("home") or not stats.get("away"):
        risk.append("Incomplete live statistics")
    if minute < 15:
        risk.append("Early match — high uncertainty")

    models = base_prediction.get("models") or {}
    if models:
        scores = {f"{m['gh']}-{m['ga']}" for m in models.values()}
        if len(scores) > 2:
            risk.append("Models disagree")

    return {"top_factors": top[:5], "risk_factors": risk[:5]}


def update_live_prediction_from_snapshot(
    snapshot: dict[str, Any],
    base_prediction: dict[str, Any],
) -> dict[str, Any]:
    """Blend pre-match model with live score, xG, momentum, and score-state logic."""
    minute = _int_or_zero(snapshot.get("minute"))
    status = snapshot.get("status") or "NS"
    score = snapshot.get("score") or {}
    cur_h = _int_or_zero(score.get("home"))
    cur_a = _int_or_zero(score.get("away"))
    stats = snapshot.get("stats") or {}
    hs = stats.get("home") or {}
    aws = stats.get("away") or {}
    events = snapshot.get("events") or []
    home_id = snapshot.get("home_team_id")
    away_id = snapshot.get("away_team_id")

    xg = calc_xg_pair(hs, aws, events, home_id, away_id)
    momentum = snapshot.get("momentum") or calc_momentum(hs, aws, events, home_id, away_id)
    live_feats = build_live_features(snapshot)

    base_lh, base_la, prematch = _base_lambdas_from_prediction(base_prediction)

    pre_match_weight = max(0.15, 1.0 - minute / 90.0)
    live_match_weight = 1.0 - pre_match_weight

    elapsed = max(minute, 1)
    projected_h = xg["home"] * (90.0 / elapsed) if elapsed > 5 else base_lh
    projected_a = xg["away"] * (90.0 / elapsed) if elapsed > 5 else base_la

    adj_lh = (1.0 - live_match_weight) * base_lh + live_match_weight * projected_h
    adj_la = (1.0 - live_match_weight) * base_la + live_match_weight * projected_a

    mom_factor_h = 0.88 + (momentum["home"] / 100.0) * 0.24
    mom_factor_a = 0.88 + (momentum["away"] / 100.0) * 0.24
    adj_lh *= mom_factor_h
    adj_la *= mom_factor_a

    home_red = _int_or_zero(hs.get("red_cards"))
    away_red = _int_or_zero(aws.get("red_cards"))
    for _ in range(home_red):
        adj_lh *= 0.82
        adj_la *= 1.10
    for _ in range(away_red):
        adj_la *= 0.82
        adj_lh *= 1.10

    adj_lh = max(0.1, min(5.0, adj_lh))
    adj_la = max(0.1, min(5.0, adj_la))

    remaining_frac = max(90 - minute, 1) / 90.0
    rem_lh = adj_lh * remaining_frac
    rem_la = adj_la * remaining_frac

    live_probs = outcome_probs_from_lambdas(rem_lh, rem_la, cur_h, cur_a, calibrate=False)
    blended = normalize_outcome_probs({
        k: pre_match_weight * prematch[k] + live_match_weight * live_probs[k]
        for k in ("home_win", "draw", "away_win")
    })
    blended = _apply_score_state_adjustments(
        blended, cur_h, cur_a, minute, xg["diff"], home_red, away_red,
    )

    dq = score_data_quality(snapshot)
    conf = compute_confidence(
        model_agreement=base_prediction.get("ensemble", {}).get("model_agreement", 0.5),
        data_quality=dq["score"],
        lineup_completeness=1.0 if snapshot.get("lineups") else 0.3,
        live_stats_completeness=0.9 if hs and aws else 0.2,
        minute=minute,
    )
    explanation = _build_explanation(snapshot, live_feats, prematch, blended, base_prediction)

    ou_payload = build_over_under_payload(rem_lh, rem_la, cur_h, cur_a)
    btts = both_teams_score_prob(adj_lh, adj_la)

    return {
        "fixture_id": snapshot.get("fixture_id"),
        "minute": minute,
        "status": status,
        "score": {"home": cur_h, "away": cur_a},
        "probabilities": {
            "home_win": round(blended["home_win"] * 100, 1),
            "draw": round(blended["draw"] * 100, 1),
            "away_win": round(blended["away_win"] * 100, 1),
        },
        "prediction": {
            "home_win": round(blended["home_win"], 4),
            "draw": round(blended["draw"], 4),
            "away_win": round(blended["away_win"], 4),
            "projected_home_goals": round(adj_lh, 2),
            "projected_away_goals": round(adj_la, 2),
            "over_2_5": ou_payload["2.5"]["over"],
            "over_3_5": ou_payload["3.5"]["over"],
            "both_teams_score": btts,
        },
        "projected_goals": {"home": round(adj_lh, 2), "away": round(adj_la, 2)},
        "over_under": ou_payload,
        "next_goal": _next_goal_prob(rem_lh, rem_la),
        "confidence": conf,
        "momentum": momentum,
        "adj_lambda_home": round(adj_lh, 3),
        "adj_lambda_away": round(adj_la, 3),
        "xg_proxy": {"home": xg["home"], "away": xg["away"]},
        "xg_diff": xg["diff"],
        "prematch_probabilities": {
            "home_win": round(prematch["home_win"] * 100, 1),
            "draw": round(prematch["draw"] * 100, 1),
            "away_win": round(prematch["away_win"] * 100, 1),
        },
        "weights": {
            "pre_match": round(pre_match_weight, 3),
            "live": round(live_match_weight, 3),
        },
        "explanation": explanation,
        "data_quality": dq,
        "live": {
            "is_live": True,
            "minute": minute,
            "score": {"home": cur_h, "away": cur_a},
            "momentum": momentum,
            "xg_proxy": {"home": xg["home"], "away": xg["away"]},
        },
    }