"""Weighted model ensemble and outcome probability from goal lambdas."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from calibration import calibrate_outcome_probs, compute_confidence, normalize_outcome_probs
from feature_builder import build_features
from score_matrix import build_score_matrix

ROOT = Path(__file__).parent
WEIGHTS_PATH = ROOT / "model_weights.json"
MAX_GOALS = 15

INTERNAL_NAMES = [
    "Poisson Regression",
    "Ridge Regression",
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
    "LightGBM",
    "Neural Network",
]

DEFAULT_WEIGHTS = {
    "Poisson Regression": 0.25,
    "Ridge Regression": 0.10,
    "Random Forest": 0.18,
    "Gradient Boosting": 0.18,
    "XGBoost": 0.17,
    "LightGBM": 0.10,
    "Neural Network": 0.02,
}

SCALED_MODELS = {"Poisson Regression", "Ridge Regression", "Neural Network"}


def load_model_weights() -> dict[str, float]:
    if WEIGHTS_PATH.exists():
        with open(WEIGHTS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if "weights" in data:
            return {k: float(v) for k, v in data["weights"].items()}
        return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
    return dict(DEFAULT_WEIGHTS)


def save_model_weights(weights: dict[str, float], metrics: dict | None = None) -> None:
    payload = {"weights": weights, "metrics": metrics or {}, "source": "backtest"}
    with open(WEIGHTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def outcome_probs_from_lambdas(
    lambda_h: float,
    lambda_a: float,
    score_h: int = 0,
    score_a: int = 0,
    *,
    calibrate: bool = True,
) -> dict[str, float]:
    probs = {"home_win": 0.0, "draw": 0.0, "away_win": 0.0}
    for add_h in range(MAX_GOALS + 1):
        for add_a in range(MAX_GOALS + 1):
            p = _poisson_pmf(add_h, lambda_h) * _poisson_pmf(add_a, lambda_a)
            fh, fa = score_h + add_h, score_a + add_a
            if fh > fa:
                probs["home_win"] += p
            elif fh == fa:
                probs["draw"] += p
            else:
                probs["away_win"] += p
    probs = normalize_outcome_probs(probs)
    if calibrate:
        probs = calibrate_outcome_probs(probs)
    return probs


def both_teams_score_prob(lambda_h: float, lambda_a: float) -> float:
    p_h_scores = 1.0 - math.exp(-max(lambda_h, 0))
    p_a_scores = 1.0 - math.exp(-max(lambda_a, 0))
    return round(p_h_scores * p_a_scores, 4)


def over_under_prob(lambda_h: float, lambda_a: float, score_h: int, score_a: int, line: float = 2.5) -> float:
    over = 0.0
    for add_h in range(MAX_GOALS + 1):
        for add_a in range(MAX_GOALS + 1):
            p = _poisson_pmf(add_h, lambda_h) * _poisson_pmf(add_a, lambda_a)
            if score_h + add_h + score_a + add_a > line:
                over += p
    return round(min(1.0, max(0.0, over)), 4)


DEFAULT_OU_LINES = (2.5, 3.5)


def build_over_under_payload(
    lambda_h: float,
    lambda_a: float,
    score_h: int = 0,
    score_a: int = 0,
    lines: tuple[float, ...] = DEFAULT_OU_LINES,
) -> dict[str, Any]:
    """Over/under probabilities for multiple goal lines (uses current score in live context)."""
    by_line: dict[str, dict[str, float]] = {}
    for line in lines:
        over = over_under_prob(lambda_h, lambda_a, score_h, score_a, line=line)
        by_line[str(line)] = {
            "line": line,
            "over": over,
            "under": round(1.0 - over, 4),
        }
    primary = by_line.get("2.5") or next(iter(by_line.values()))
    return {
        **by_line,
        "line": primary["line"],
        "over": primary["over"],
        "under": primary["under"],
    }


def weighted_ensemble_goals(
    model_preds: dict[str, tuple[float, float]],
    weights: dict[str, float] | None = None,
) -> tuple[float, float, float]:
    """Return weighted rh, ra, agreement score."""
    w = weights or load_model_weights()
    total_w = 0.0
    rh, ra = 0.0, 0.0
    scores = []
    for name, (gh, ga, raw_h, raw_a) in model_preds.items():
        wt = w.get(name, 0.01)
        rh += raw_h * wt
        ra += raw_a * wt
        total_w += wt
        scores.append(f"{round(raw_h)}-{round(raw_a)}")
    if total_w > 0:
        rh /= total_w
        ra /= total_w
    agreement = len(set(scores)) / max(len(scores), 1)
    model_agreement = 1.0 - (agreement - 1 / max(len(scores), 1)) / max(1 - 1 / max(len(scores), 1), 0.01)
    model_agreement = max(0.0, min(1.0, model_agreement))
    return rh, ra, model_agreement


def build_prediction_envelope(
    home: str,
    away: str,
    model_preds: dict[str, tuple[int, int, float, float]],
    *,
    fixture_id: int | None = None,
    data_quality: float = 0.6,
    lineup_completeness: float = 0.5,
    training_meta: dict | None = None,
    live: dict | None = None,
) -> dict[str, Any]:
    """Rich prediction output for dashboard/API."""
    weights = load_model_weights()
    rh, ra, model_agreement = weighted_ensemble_goals(model_preds, weights)
    probs = outcome_probs_from_lambdas(rh, ra, calibrate=True)
    _live = live or {}
    _score = _live.get("score") or {}
    cur_h = int(_live.get("score_home", _score.get("home", 0)))
    cur_a = int(_live.get("score_away", _score.get("away", 0)))
    ou_payload = build_over_under_payload(rh, ra, cur_h, cur_a)
    ou_25 = ou_payload["2.5"]["over"]
    ou_35 = ou_payload["3.5"]["over"]
    btts = both_teams_score_prob(rh, ra)
    score_matrix = build_score_matrix(rh, ra, score_h=cur_h, score_a=cur_a)

    top_factors = []
    risk_factors = []
    feats = build_features(home, away)
    if feats["elo_diff"] > 50:
        top_factors.append("Home team has stronger pre-match Elo rating")
    elif feats["elo_diff"] < -50:
        top_factors.append("Away team has stronger pre-match Elo rating")
    if feats["xg_diff"] > 0.3:
        top_factors.append("Home team stronger attacking xG profile")
    elif feats["xg_diff"] < -0.3:
        top_factors.append("Away team stronger attacking xG profile")
    if model_agreement < 0.5:
        risk_factors.append("Models disagree on scoreline")

    conf = compute_confidence(
        model_agreement=model_agreement,
        data_quality=data_quality,
        lineup_completeness=lineup_completeness,
    )

    if live:
        if live.get("xg_proxy", {}).get("home", 0) > live.get("xg_proxy", {}).get("away", 0) + 0.3:
            top_factors.append("Live xG proxy favors home team")
        if live.get("home_red_cards", 0) or live.get("away_red_cards", 0):
            top_factors.append("Red card affecting match state")
        conf = compute_confidence(
            model_agreement=model_agreement,
            data_quality=data_quality,
            lineup_completeness=lineup_completeness,
            live_stats_completeness=0.8,
            minute=live.get("minute", 0),
        )

    return {
        "fixture_id": fixture_id,
        "home_team": home,
        "away_team": away,
        "prediction": {
            "home_win": probs["home_win"],
            "draw": probs["draw"],
            "away_win": probs["away_win"],
            "projected_home_goals": round(rh, 2),
            "projected_away_goals": round(ra, 2),
            "over_2_5": ou_25,
            "over_3_5": ou_35,
            "both_teams_score": btts,
            "over_under": ou_payload,
            "score_matrix": {
                "top_exact_scores": score_matrix["top_exact_scores"],
                "score_matrix": score_matrix["score_matrix"],
            },
        },
        "live": live,
        "confidence": {
            **conf,
            "data_quality": round(data_quality, 3),
        },
        "explanation": {
            "top_factors": top_factors[:4],
            "risk_factors": risk_factors[:4],
        },
        "training_metadata": training_meta or {},
        "ensemble": {
            "weights": {k: weights.get(k, 0) for k in INTERNAL_NAMES if k in weights},
            "model_agreement": round(model_agreement, 3),
        },
    }
