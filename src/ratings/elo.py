"""Simple Elo rating for international teams."""
from __future__ import annotations

import math


def expected_score(rating_a: float, rating_b: float, home_adv: float = 65.0) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b + home_adv - rating_a) / 400.0))


def update_elo(
    rating_a: float,
    rating_b: float,
    score_a: float,
    *,
    k: float = 32.0,
    home_adv: float = 65.0,
) -> tuple[float, float]:
    """score_a in {1.0 win, 0.5 draw, 0.0 loss}. Returns (new_a, new_b)."""
    exp_a = expected_score(rating_a, rating_b, home_adv)
    exp_b = 1.0 - exp_a
    new_a = rating_a + k * (score_a - exp_a)
    new_b = rating_b + k * ((1.0 - score_a) - exp_b)
    return new_a, new_b


def elo_win_probability(home_rating: float, away_rating: float, home_adv: float = 65.0) -> float:
    return expected_score(home_rating, away_rating, home_adv)
