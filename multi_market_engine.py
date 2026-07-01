"""Multi-market probability engine — extends existing ML pipeline without replacing it."""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from typing import Any

from calibration import compute_confidence
from goal_markets import build_goal_markets, clamp_prob
from knockout_progression import build_knockout_progression, is_knockout_round
from market_trading_metrics import build_kalshi_market_metrics
from monte_carlo_simulator import simulate_knockout_match
from score_matrix import build_score_matrix

log = logging.getLogger(__name__)

# First-half share of expected goals (empirical WC average ~43–46%)
HT_GOAL_SHARE = 0.45

# Registry: which specialized model produces each market family
MARKET_MODEL_REGISTRY = {
    "regulation_1x2": "PoissonGoalsEnsemble",
    "qualification": "KnockoutCascadeModel",
    "extra_time": "KnockoutETModel",
    "penalties": "KnockoutPenaltyModel",
    "qualification": "KnockoutQualificationModel",
    "goals_totals": "PoissonGoalsModel",
    "btts": "PoissonGoalsModel",
    "correct_score": "ScoreMatrixModel",
    "half_time": "PoissonHalfTimeModel",
    "first_goal": "PoissonFirstGoalModel",
    "team_totals": "PoissonTeamTotalsModel",
    "double_chance": "PoissonOutcomesModel",
    "simulation": "MonteCarloKnockoutSimulator",
}


def _lambdas_from_ml_match(ml_match: dict) -> tuple[float, float]:
    pred = ml_match.get("prediction") or {}
    lh = float(pred.get("projected_home_goals") or ml_match.get("ens_h") or 1.2)
    la = float(pred.get("projected_away_goals") or ml_match.get("ens_a") or 1.0)
    return lh, la


def _first_goal_probs(lambda_h: float, lambda_a: float) -> dict[str, float]:
    """Approximate first/last goal scorer markets from Poisson rates."""
    p_no_goal = math.exp(-(lambda_h + lambda_a))
    p_any = 1.0 - p_no_goal
    if p_any <= 0:
        return {"home_first": 0.0, "away_first": 0.0, "no_goal": 1.0}
    home_first = clamp_prob(lambda_h / (lambda_h + lambda_a) * p_any)
    away_first = clamp_prob(lambda_a / (lambda_h + lambda_a) * p_any)
    return {
        "home_first": home_first,
        "away_first": away_first,
        "no_goal": clamp_prob(p_no_goal),
    }


def _half_time_markets(lambda_h: float, lambda_a: float) -> dict[str, Any]:
    lh_ht, la_ht = lambda_h * HT_GOAL_SHARE, lambda_a * HT_GOAL_SHARE
    matrix = build_score_matrix(lh_ht, la_ht)
    btts_ht = clamp_prob(
        (1 - math.exp(-lh_ht)) * (1 - math.exp(-la_ht)),
    )
    return {
        "winner": {
            "home_win": matrix["home_win"],
            "draw": matrix["draw"],
            "away_win": matrix["away_win"],
        },
        "btts_yes": btts_ht,
        "btts_no": clamp_prob(1 - btts_ht),
        "expected_goals": round(lh_ht + la_ht, 3),
    }


def _second_half_markets(lambda_h: float, lambda_a: float) -> dict[str, Any]:
    lh_2h, la_2h = lambda_h * (1 - HT_GOAL_SHARE), lambda_a * (1 - HT_GOAL_SHARE)
    matrix = build_score_matrix(lh_2h, la_2h)
    btts_2h = clamp_prob(
        (1 - math.exp(-lh_2h)) * (1 - math.exp(-la_2h)),
    )
    return {
        "winner": {
            "home_win": matrix["home_win"],
            "draw": matrix["draw"],
            "away_win": matrix["away_win"],
        },
        "btts_yes": btts_2h,
        "btts_no": clamp_prob(1 - btts_2h),
        "expected_goals": round(lh_2h + la_2h, 3),
    }


def _clean_sheet_probs(lambda_h: float, lambda_a: float) -> dict[str, float]:
    return {
        "home_clean_sheet": clamp_prob(math.exp(-lambda_a)),
        "away_clean_sheet": clamp_prob(math.exp(-lambda_h)),
        "both_clean_sheet": clamp_prob(math.exp(-(lambda_h + lambda_a))),
    }


def _goal_count_distribution(total_goals_probs: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    five_plus = 0.0
    for k, v in total_goals_probs.items():
        n = int(k)
        if n >= 5:
            five_plus += v
        else:
            out[str(n)] = round(v, 6)
    out["5+"] = round(five_plus, 6)
    return out


def build_multi_market_bundle(
    ml_match: dict,
    *,
    score_h: int = 0,
    score_a: int = 0,
    live: bool = False,
    run_simulation: bool = True,
    simulation_count: int = 100_000,
    kalshi_prices: dict[str, float | None] | None = None,
) -> dict[str, Any]:
    """
    Full multi-market probability bundle for one fixture.

    Reuses ensemble lambdas from existing ml_match; extends with knockout cascade,
    half-time markets, first goal, and optional Monte Carlo validation.
    """
    t0 = time.perf_counter()
    home = ml_match.get("home", "")
    away = ml_match.get("away", "")
    group = ml_match.get("group") or ""
    knockout = is_knockout_round(group) or bool(ml_match.get("knockout"))

    lambda_h, lambda_a = _lambdas_from_ml_match(ml_match)
    goal_mkts = build_goal_markets(
        lambda_h, lambda_a, score_h=score_h, score_a=score_a, live=live,
    )
    matrix = build_score_matrix(lambda_h, lambda_a, score_h=score_h, score_a=score_a)
    outcomes = goal_mkts.get("outcomes") or {}
    top_scores = sorted(
        (matrix.get("score_matrix") or {}).items(), key=lambda x: x[1], reverse=True,
    )[:10]

    first_goal = _first_goal_probs(lambda_h, lambda_a)
    ht = _half_time_markets(lambda_h, lambda_a)
    sh = _second_half_markets(lambda_h, lambda_a)
    clean = _clean_sheet_probs(lambda_h, lambda_a)
    total_goals = matrix.get("total_goals_probabilities") or {}

    bundle: dict[str, Any] = {
        "fixture_id": ml_match.get("fixture_id"),
        "home": home,
        "away": away,
        "group": group,
        "knockout": knockout,
        "live": live,
        "current_score": f"{score_h}-{score_a}",
        "model_versions": ml_match.get("model_version") or {},
        "ensemble": ml_match.get("ensemble") or {},
        "lambdas": {
            "home": round(lambda_h, 4),
            "away": round(lambda_a, 4),
        },
        "match_winner": {
            "home_win_90": outcomes.get("home_win"),
            "draw_90": outcomes.get("draw"),
            "away_win_90": outcomes.get("away_win"),
            "model": MARKET_MODEL_REGISTRY["regulation_1x2"],
        },
        "double_chance": {
            "home_or_draw": outcomes.get("home_double_chance"),
            "home_or_away": outcomes.get("no_draw"),
            "away_or_draw": outcomes.get("away_double_chance"),
            "model": MARKET_MODEL_REGISTRY["double_chance"],
        },
        "goals": {
            "over_under": {
                k: goal_mkts[k]
                for k in goal_mkts
                if k.startswith("over_") or k.startswith("under_")
            },
            "exact_total_distribution": _goal_count_distribution(total_goals),
            "model": MARKET_MODEL_REGISTRY["goals_totals"],
        },
        "both_teams_score": {
            "yes": goal_mkts.get("btts_yes"),
            "no": goal_mkts.get("btts_no"),
            "model": MARKET_MODEL_REGISTRY["btts"],
        },
        "correct_score": {
            "top_10": [{"score": s, "probability": round(p, 6)} for s, p in top_scores],
            "full_matrix_sample": matrix.get("score_matrix"),
            "model": MARKET_MODEL_REGISTRY["correct_score"],
        },
        "first_goal": {
            **first_goal,
            "model": MARKET_MODEL_REGISTRY["first_goal"],
        },
        "last_goal": {
            "home": first_goal["home_first"],
            "away": first_goal["away_first"],
            "no_goal": first_goal["no_goal"],
            "model": MARKET_MODEL_REGISTRY["first_goal"],
        },
        "first_half": {**ht, "model": MARKET_MODEL_REGISTRY["half_time"]},
        "second_half": {**sh, "model": MARKET_MODEL_REGISTRY["half_time"]},
        "team_total_goals": {
            "home": {
                "expected": round(lambda_h, 3),
                "over_0_5": goal_mkts.get("home_over_0_5"),
                "over_1_5": goal_mkts.get("home_over_1_5"),
            },
            "away": {
                "expected": round(lambda_a, 3),
                "over_0_5": goal_mkts.get("away_over_0_5"),
                "over_1_5": goal_mkts.get("away_over_1_5"),
            },
            "model": MARKET_MODEL_REGISTRY["team_totals"],
        },
        "clean_sheet": {**clean, "model": MARKET_MODEL_REGISTRY["goals_totals"]},
        "expected_stats": {
            "expected_goals_home": round(lambda_h, 3),
            "expected_goals_away": round(lambda_a, 3),
            "expected_total_goals": round(lambda_h + lambda_a, 3),
        },
    }

    if knockout:
        progression = build_knockout_progression(
            lambda_h, lambda_a, score_h=score_h, score_a=score_a,
            home=home, away=away,
            fixture_id=ml_match.get("fixture_id"),
        )
        bundle["knockout_progression"] = progression
        bundle["qualification_probability"] = progression["qualification"]
        bundle["extra_time_probability"] = {
            "reach": progression["extra_time"]["reach_probability"],
            "model": MARKET_MODEL_REGISTRY["extra_time"],
        }
        bundle["penalty_probability"] = {
            "reach": progression["penalties"]["reach_probability"],
            "home_win_skill": progression["penalties"]["home_win_skill"],
            "away_win_skill": progression["penalties"]["away_win_skill"],
            "model": MARKET_MODEL_REGISTRY["penalties"],
        }
        bundle["winner_probability"] = {
            "home_qualifies": progression["qualification"]["home"],
            "away_qualifies": progression["qualification"]["away"],
        }
        if run_simulation:
            sim_t0 = time.perf_counter()
            sim = simulate_knockout_match(
                lambda_h, lambda_a, n_simulations=simulation_count, seed=42,
                home_pen_skill=progression["penalties"]["home_win_skill"],
            )
            bundle["simulation_summary"] = {
                **sim,
                "runtime_ms": round((time.perf_counter() - sim_t0) * 1000, 1),
                "model": MARKET_MODEL_REGISTRY["simulation"],
            }
    else:
        bundle["qualification_probability"] = None
        bundle["winner_probability"] = {
            "home_win_90": outcomes.get("home_win"),
            "draw_90": outcomes.get("draw"),
            "away_win_90": outcomes.get("away_win"),
        }

    conf_src = ml_match.get("confidence") or {}
    conf_score = conf_src.get("score", 0.5) if isinstance(conf_src, dict) else float(conf_src or 0.5)
    agreement = (ml_match.get("ensemble") or {}).get("model_agreement", 0.5)
    bundle["confidence"] = compute_confidence(
        model_agreement=float(agreement),
        data_quality=0.65,
        calibration_uncertainty=0.15 if knockout else 0.1,
    )
    bundle["confidence"]["score"] = round(
        (conf_score + bundle["confidence"]["score"]) / 2, 3,
    )

    # Flat Kalshi-ready market map
    kalshi_map: dict[str, float] = {
        "home_win_90": float(outcomes.get("home_win") or 0),
        "draw_90": float(outcomes.get("draw") or 0),
        "away_win_90": float(outcomes.get("away_win") or 0),
        "btts_yes": float(goal_mkts.get("btts_yes") or 0),
        "over_2_5": float(goal_mkts.get("over_2_5") or 0),
        "over_3_5": float(goal_mkts.get("over_3_5") or 0),
    }
    if knockout and bundle.get("knockout_progression"):
        qual = bundle["knockout_progression"]["qualification"]
        kalshi_map["home_qualifies"] = float(qual["home"])
        kalshi_map["away_qualifies"] = float(qual["away"])
        kalshi_map["reach_extra_time"] = float(
            bundle["knockout_progression"]["extra_time"]["reach_probability"],
        )
        kalshi_map["reach_penalties"] = float(
            bundle["knockout_progression"]["penalties"]["reach_probability"],
        )

    bundle["kalshi_markets"] = kalshi_map
    bundle["trading_metrics"] = build_kalshi_market_metrics(
        kalshi_map,
        kalshi_prices,
        confidence=bundle["confidence"]["score"],
    )

    bundle["generated_at"] = datetime.now(timezone.utc).isoformat()
    bundle["prediction_latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    bundle["market_model_registry"] = MARKET_MODEL_REGISTRY

    log.debug(
        "Multi-market bundle %s vs %s (knockout=%s) in %.1fms",
        home, away, knockout, bundle["prediction_latency_ms"],
    )
    return bundle


def flatten_for_api(bundle: dict[str, Any]) -> dict[str, Any]:
    """Compact API view matching requested response shape."""
    return {
        "fixture_id": bundle.get("fixture_id"),
        "home": bundle.get("home"),
        "away": bundle.get("away"),
        "knockout": bundle.get("knockout"),
        "winner_probability": bundle.get("winner_probability"),
        "qualification_probability": bundle.get("qualification_probability"),
        "extra_time_probability": bundle.get("extra_time_probability"),
        "penalty_probability": bundle.get("penalty_probability"),
        "match_winner": bundle.get("match_winner"),
        "first_goal_probability": bundle.get("first_goal"),
        "correct_score_distribution": bundle.get("correct_score"),
        "goal_distribution": bundle.get("goals"),
        "team_goal_distribution": bundle.get("team_total_goals"),
        "BTTS_probability": bundle.get("both_teams_score"),
        "clean_sheet_probability": bundle.get("clean_sheet"),
        "simulation_summary": bundle.get("simulation_summary"),
        "knockout_progression": bundle.get("knockout_progression"),
        "confidence": bundle.get("confidence"),
        "model_versions": bundle.get("model_versions"),
        "trading_metrics": bundle.get("trading_metrics"),
        "kalshi_markets": bundle.get("kalshi_markets"),
        "generated_at": bundle.get("generated_at"),
    }
