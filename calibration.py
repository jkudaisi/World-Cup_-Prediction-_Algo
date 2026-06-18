"""Probability calibration, normalization, and confidence scoring."""

from __future__ import annotations

import math
from typing import Any

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    SKLEARN_CALIBRATION = True
except ImportError:
    SKLEARN_CALIBRATION = False


def normalize_outcome_probs(probs: dict[str, float]) -> dict[str, float]:
    """Ensure home_win + draw + away_win = 1.0."""
    h = max(0.0, float(probs.get("home_win", 0)))
    d = max(0.0, float(probs.get("draw", 0)))
    a = max(0.0, float(probs.get("away_win", 0)))
    total = h + d + a
    if total <= 0:
        return {"home_win": 1 / 3, "draw": 1 / 3, "away_win": 1 / 3}
    return {
        "home_win": round(h / total, 4),
        "draw": round(d / total, 4),
        "away_win": round(a / total, 4),
    }


def shrink_toward_uniform(probs: dict[str, float], strength: float = 0.15) -> dict[str, float]:
    """Reduce overconfidence by blending with uniform distribution."""
    u = 1 / 3
    return normalize_outcome_probs({
        k: (1 - strength) * probs.get(k, u) + strength * u
        for k in ("home_win", "draw", "away_win")
    })


def platt_calibrate(raw_prob: float, a: float = 1.0, b: float = 0.0) -> float:
    """Sigmoid calibration on a single probability."""
    z = a * raw_prob + b
    return 1.0 / (1.0 + math.exp(-max(-20, min(20, z))))


def calibrate_outcome_probs(
    probs: dict[str, float],
    *,
    shrink: float = 0.12,
    max_peak: float = 0.82,
) -> dict[str, float]:
    """Post-hoc calibration: shrink + cap peak probability."""
    out = shrink_toward_uniform(probs, strength=shrink)
    peak_key = max(out, key=out.get)
    if out[peak_key] > max_peak:
        excess = out[peak_key] - max_peak
        out[peak_key] = max_peak
        others = [k for k in out if k != peak_key]
        share = excess / len(others)
        for k in others:
            out[k] += share
    return normalize_outcome_probs(out)


def fit_isotonic_calibrator(y_true: list[int], y_prob: list[float]):
    """Fit isotonic regression when enough samples exist."""
    if not SKLEARN_CALIBRATION or len(y_true) < 30:
        return None
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_prob, y_true)
    return iso


def confidence_label(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.55:
        return "medium"
    if score >= 0.35:
        return "low"
    return "very_low"


def compute_confidence(
    *,
    model_agreement: float = 0.5,
    data_quality: float = 0.5,
    lineup_completeness: float = 0.5,
    live_stats_completeness: float = 0.0,
    minute: int = 0,
    calibration_uncertainty: float = 0.2,
) -> dict[str, Any]:
    """Combined confidence score 0–1."""
    live_boost = min(0.15, (minute / 90.0) * 0.15) if minute > 0 else 0
    raw = (
        0.30 * model_agreement
        + 0.25 * data_quality
        + 0.15 * lineup_completeness
        + 0.15 * live_stats_completeness
        + live_boost
        - 0.15 * calibration_uncertainty
    )
    score = round(max(0.1, min(0.95, raw)), 3)
    return {"score": score, "label": confidence_label(score)}


def brier_score(probs: list[float], outcomes: list[int]) -> float:
    if not probs:
        return 1.0
    return sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / len(probs)


def log_loss_multiclass(
    prob_rows: list[dict[str, float]],
    outcomes: list[str],
) -> float:
    """Outcomes: 'home_win', 'draw', 'away_win'."""
    eps = 1e-15
    total = 0.0
    for row, outcome in zip(prob_rows, outcomes):
        p = max(eps, min(1 - eps, row.get(outcome, eps)))
        total -= math.log(p)
    return total / max(len(outcomes), 1)
