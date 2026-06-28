"""Exact score probability matrix from Poisson lambdas."""

from __future__ import annotations

import math
from typing import Any

MAX_MATRIX = 8


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def build_score_matrix(
    lambda_h: float,
    lambda_a: float,
    *,
    score_h: int = 0,
    score_a: int = 0,
    max_goals: int = MAX_MATRIX,
) -> dict[str, Any]:
    """Full score probability table and derived outcome probabilities."""
    matrix: dict[str, float] = {}
    home_win = draw = away_win = 0.0
    total_goals_probs: dict[int, float] = {}
    home_goal_probs: dict[int, float] = {}
    away_goal_probs: dict[int, float] = {}

    for add_h in range(max_goals + 1):
        p_h = _poisson_pmf(add_h, lambda_h)
        for add_a in range(max_goals + 1):
            p_a = _poisson_pmf(add_a, lambda_a)
            p = p_h * p_a
            fh, fa = score_h + add_h, score_a + add_a
            key = f"{fh}-{fa}"
            matrix[key] = matrix.get(key, 0) + p

            total = fh + fa
            total_goals_probs[total] = total_goals_probs.get(total, 0) + p

            if fh > fa:
                home_win += p
            elif fh == fa:
                draw += p
            else:
                away_win += p

    home_goal_probs = {str(k): round(_poisson_pmf(k, lambda_h), 6) for k in range(max_goals + 1)}
    away_goal_probs = {str(k): round(_poisson_pmf(k, lambda_a), 6) for k in range(max_goals + 1)}

    total = home_win + draw + away_win
    if total > 0:
        home_win /= total
        draw /= total
        away_win /= total

    sorted_scores = sorted(matrix.items(), key=lambda x: x[1], reverse=True)
    top_exact = [
        {"score": k, "probability": round(v, 6)}
        for k, v in sorted_scores[:10]
    ]

    table = {
        f"{h}-{a}": round(matrix.get(f"{h}-{a}", 0), 6)
        for h in range(max_goals + 1)
        for a in range(max_goals + 1)
    }

    return {
        "top_exact_scores": top_exact[:5],
        "score_matrix": table,
        "home_win": round(home_win, 4),
        "draw": round(draw, 4),
        "away_win": round(away_win, 4),
        "total_goals_probabilities": {
            str(k): round(v, 6) for k, v in sorted(total_goals_probs.items())
        },
        "home_goal_probabilities": home_goal_probs,
        "away_goal_probabilities": away_goal_probs,
        "lambda_home": round(lambda_h, 4),
        "lambda_away": round(lambda_a, 4),
        "current_score": f"{score_h}-{score_a}",
    }
