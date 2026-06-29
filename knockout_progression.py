"""Knockout match progression: regulation → extra time → penalties → qualification."""

from __future__ import annotations

import math
from typing import Any

from calibration import normalize_outcome_probs
from score_matrix import build_score_matrix

# Extra-time segment: 30 minutes vs 90 regulation minutes
ET_MINUTES = 30.0
REG_MINUTES = 90.0
ET_INTENSITY = 0.92  # slightly lower scoring rate per minute in ET

DEFAULT_HOME_PEN_SKILL = 0.52  # home advantage in shootouts (literature ~51–54%)


def _clamp(p: float) -> float:
    return round(min(1.0, max(0.0, float(p))), 6)


def et_lambdas(lambda_h: float, lambda_a: float) -> tuple[float, float]:
    """Scale regulation lambdas to a 30-minute extra-time segment."""
    scale = (ET_MINUTES / REG_MINUTES) * ET_INTENSITY
    return max(0.01, lambda_h * scale), max(0.01, lambda_a * scale)


def regulation_outcomes(
    lambda_h: float,
    lambda_a: float,
    *,
    score_h: int = 0,
    score_a: int = 0,
) -> dict[str, float]:
    """90-minute win / draw / loss probabilities from Poisson lambdas."""
    matrix = build_score_matrix(lambda_h, lambda_a, score_h=score_h, score_a=score_a)
    return normalize_outcome_probs({
        "home_win": matrix["home_win"],
        "draw": matrix["draw"],
        "away_win": matrix["away_win"],
    })


def extra_time_outcomes_given_draw(
    lambda_h: float,
    lambda_a: float,
    *,
    score_h: int = 0,
    score_a: int = 0,
) -> dict[str, float]:
    """ET outcomes conditional on entering ET tied at score_h:score_a."""
    lh, la = et_lambdas(lambda_h, lambda_a)
    matrix = build_score_matrix(lh, la, score_h=score_h, score_a=score_a)
    return normalize_outcome_probs({
        "home_win": matrix["home_win"],
        "draw": matrix["draw"],
        "away_win": matrix["away_win"],
    })


def build_knockout_progression(
    lambda_h: float,
    lambda_a: float,
    *,
    score_h: int = 0,
    score_a: int = 0,
    home_pen_skill: float = DEFAULT_HOME_PEN_SKILL,
    away_pen_skill: float | None = None,
    home: str | None = None,
    away: str | None = None,
) -> dict[str, Any]:
    """
    Full knockout cascade with mathematically consistent qualification probs.

    When home/away are provided and trained knockout models exist, ET conditional
    rates and penalty skill are blended with Poisson-derived defaults.

    Qualification:
      P(home) = P(home 90) + P(draw 90)*P(home ET|draw)
                + P(draw 90)*P(draw ET|draw)*P(home pens)
    """
    if away_pen_skill is None:
        away_pen_skill = 1.0 - home_pen_skill

    ml_adj: dict[str, Any] = {"available": False}
    if home and away:
        try:
            from knockout_models import predict_knockout_adjustments
            ml_adj = predict_knockout_adjustments(home, away, lambda_h, lambda_a)
            if ml_adj.get("available"):
                home_pen_skill = float(ml_adj.get("home_pen_skill", home_pen_skill))
                away_pen_skill = 1.0 - home_pen_skill
        except Exception:
            pass

    reg = regulation_outcomes(lambda_h, lambda_a, score_h=score_h, score_a=score_a)
    p_h90, p_d90, p_a90 = reg["home_win"], reg["draw"], reg["away_win"]

    if ml_adj.get("available") and ml_adj.get("et_conditional"):
        et = ml_adj["et_conditional"]
        p_h_et, p_d_et, p_a_et = et["home_win"], et["draw"], et["away_win"]
    else:
        et = extra_time_outcomes_given_draw(lambda_h, lambda_a, score_h=score_h, score_a=score_a)
        p_h_et, p_d_et, p_a_et = et["home_win"], et["draw"], et["away_win"]

    p_reach_et = _clamp(p_d90)
    p_reach_pens = _clamp(p_d90 * p_d_et)

    p_home_via_et = p_d90 * p_h_et
    p_away_via_et = p_d90 * p_a_et
    p_home_via_pens = p_d90 * p_d_et * home_pen_skill
    p_away_via_pens = p_d90 * p_d_et * away_pen_skill

    p_home_qual = _clamp(p_h90 + p_home_via_et + p_home_via_pens)
    p_away_qual = _clamp(p_a90 + p_away_via_et + p_away_via_pens)

    # Optional blend toward direct qualification model (keeps cascade as primary)
    if ml_adj.get("available") and ml_adj.get("home_qualifies_ml") is not None:
        bw = float(ml_adj.get("blend_weight") or 0.0) * 0.5
        p_home_ml = float(ml_adj["home_qualifies_ml"])
        p_home_qual = _clamp((1 - bw) * p_home_qual + bw * p_home_ml)
        p_away_qual = _clamp((1 - bw) * p_away_qual + bw * (1 - p_home_ml))

    total = p_home_qual + p_away_qual
    if total > 0:
        p_home_qual = _clamp(p_home_qual / total)
        p_away_qual = _clamp(p_away_qual / total)

    result = {
        "regulation": {
            "home_win": _clamp(p_h90),
            "draw": _clamp(p_d90),
            "away_win": _clamp(p_a90),
            "label": "90 minutes",
        },
        "extra_time": {
            "reach_probability": p_reach_et,
            "conditional_if_draw_at_90": {
                "home_win": _clamp(p_h_et),
                "draw": _clamp(p_d_et),
                "away_win": _clamp(p_a_et),
                "label": "Extra time (if level after 90)",
            },
            "home_win_via_et": _clamp(p_home_via_et),
            "away_win_via_et": _clamp(p_away_via_et),
        },
        "penalties": {
            "reach_probability": p_reach_pens,
            "home_win_skill": _clamp(home_pen_skill),
            "away_win_skill": _clamp(away_pen_skill),
            "home_win_via_pens": _clamp(p_home_via_pens),
            "away_win_via_pens": _clamp(p_away_via_pens),
            "label": "Penalty shootout (if level after ET)",
        },
        "qualification": {
            "home": p_home_qual,
            "away": p_away_qual,
            "home_pct": round(p_home_qual * 100, 2),
            "away_pct": round(p_away_qual * 100, 2),
        },
        "path_summary": {
            "home_win_90": _clamp(p_h90),
            "away_win_90": _clamp(p_a90),
            "decided_in_90": _clamp(p_h90 + p_a90),
            "decided_in_et": _clamp(p_home_via_et + p_away_via_et),
            "decided_in_pens": _clamp(p_home_via_pens + p_away_via_pens),
        },
        "ml_adjustments": {
            "available": ml_adj.get("available", False),
            "blend_weight": ml_adj.get("blend_weight", 0.0),
            "models_used": list((ml_adj.get("meta") or {}).get("models", {}).keys()),
        },
    }
    return result


def is_knockout_round(group_or_round: str | None) -> bool:
    if not group_or_round:
        return False
    g = group_or_round.upper().strip()
    return g in {"R32", "R16", "QF", "SF", "3P", "F", "KO"} or "ROUND OF" in g.upper() or "KNOCKOUT" in g.upper()
