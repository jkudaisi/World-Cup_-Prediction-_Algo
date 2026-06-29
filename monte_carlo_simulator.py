"""Monte Carlo match simulation for multi-market probability validation."""

from __future__ import annotations

import math
import random
from typing import Any

from knockout_progression import ET_INTENSITY, ET_MINUTES, REG_MINUTES, DEFAULT_HOME_PEN_SKILL


def _poisson_sample(lam: float, rng: random.Random) -> int:
    if lam <= 0:
        return 0
    limit = math.exp(-lam)
    k = 0
    p = 1.0
    while p > limit:
        k += 1
        p *= rng.random()
    return k - 1


def _simulate_segment(
    lambda_h: float,
    lambda_a: float,
    start_h: int,
    start_a: int,
    rng: random.Random,
) -> tuple[int, int]:
    gh = start_h + _poisson_sample(lambda_h, rng)
    ga = start_a + _poisson_sample(lambda_a, rng)
    return gh, ga


def simulate_knockout_match(
    lambda_h: float,
    lambda_a: float,
    *,
    n_simulations: int = 100_000,
    home_pen_skill: float = DEFAULT_HOME_PEN_SKILL,
    seed: int | None = None,
) -> dict[str, Any]:
    """Simulate knockout progression and aggregate market frequencies."""
    rng = random.Random(seed)
    counts = {
        "home_win_90": 0,
        "draw_90": 0,
        "away_win_90": 0,
        "home_win_et": 0,
        "away_win_et": 0,
        "draw_et": 0,
        "home_win_pens": 0,
        "away_win_pens": 0,
        "home_qualifies": 0,
        "away_qualifies": 0,
        "reach_et": 0,
        "reach_pens": 0,
        "total_goals": {},
        "btts": 0,
        "exact_score": {},
    }

    et_scale = (ET_MINUTES / REG_MINUTES) * ET_INTENSITY
    lh_et, la_et = lambda_h * et_scale, lambda_a * et_scale

    for _ in range(n_simulations):
        h, a = _simulate_segment(lambda_h, lambda_a, 0, 0, rng)
        tg = h + a
        counts["total_goals"][tg] = counts["total_goals"].get(tg, 0) + 1
        key = f"{h}-{a}"
        counts["exact_score"][key] = counts["exact_score"].get(key, 0) + 1
        if h > 0 and a > 0:
            counts["btts"] += 1

        if h > a:
            counts["home_win_90"] += 1
            counts["home_qualifies"] += 1
            continue
        if h < a:
            counts["away_win_90"] += 1
            counts["away_qualifies"] += 1
            continue

        counts["draw_90"] += 1
        counts["reach_et"] += 1
        h2, a2 = _simulate_segment(lh_et, la_et, h, a, rng)

        if h2 > a2:
            counts["home_win_et"] += 1
            counts["home_qualifies"] += 1
            continue
        if h2 < a2:
            counts["away_win_et"] += 1
            counts["away_qualifies"] += 1
            continue

        counts["draw_et"] += 1
        counts["reach_pens"] += 1
        if rng.random() < home_pen_skill:
            counts["home_win_pens"] += 1
            counts["home_qualifies"] += 1
        else:
            counts["away_win_pens"] += 1
            counts["away_qualifies"] += 1

    n = float(n_simulations)

    def rate(k: str) -> float:
        return round(counts[k] / n, 6)

    top_scores = sorted(
        counts["exact_score"].items(), key=lambda x: x[1], reverse=True,
    )[:10]

    return {
        "n_simulations": n_simulations,
        "regulation": {
            "home_win": rate("home_win_90"),
            "draw": rate("draw_90"),
            "away_win": rate("away_win_90"),
        },
        "extra_time": {
            "reach_probability": rate("reach_et"),
            "home_win_via_et": rate("home_win_et"),
            "away_win_via_et": rate("away_win_et"),
        },
        "penalties": {
            "reach_probability": rate("reach_pens"),
            "home_win_via_pens": rate("home_win_pens"),
            "away_win_via_pens": rate("away_win_pens"),
        },
        "qualification": {
            "home": rate("home_qualifies"),
            "away": rate("away_qualifies"),
        },
        "total_goals_distribution": {
            str(k): round(v / n, 6)
            for k, v in sorted(counts["total_goals"].items())
        },
        "btts_yes": rate("btts"),
        "top_correct_scores": [
            {"score": s, "probability": round(c / n, 6)}
            for s, c in top_scores
        ],
    }
