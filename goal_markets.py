"""Goal-related market probabilities for pre-match and live predictions."""

from __future__ import annotations

from typing import Any

from ensemble import MAX_GOALS, both_teams_score_prob, over_under_prob
from score_matrix import build_score_matrix


def clamp_prob(p: float) -> float:
    return round(min(1.0, max(0.0, float(p))), 4)


def over_under_pair(
    lambda_h: float,
    lambda_a: float,
    line: float,
    score_h: int = 0,
    score_a: int = 0,
) -> dict[str, float]:
    over = over_under_prob(lambda_h, lambda_a, score_h, score_a, line=line)
    return {"over": over, "under": clamp_prob(1.0 - over), "line": line}


def team_over_prob(lambda_team: float, line: float, current_goals: int = 0) -> float:
    """P(team scores more than line given current goals)."""
    import math
    remaining = line - current_goals
    if remaining < 0:
        return 1.0
    if remaining == 0:
        return clamp_prob(1.0 - math.exp(-max(lambda_team, 0)))
    # P(additional goals > remaining) for half-goal lines
    over = 0.0
    for k in range(MAX_GOALS + 1):
        p = math.exp(-lambda_team) * (lambda_team ** k) / math.factorial(k) if lambda_team > 0 else (1.0 if k == 0 else 0.0)
        if k > remaining:
            over += p
        elif remaining == int(remaining) and k == int(remaining):
            pass  # strict over for x.5 lines handled by float remaining
    return clamp_prob(over)


def build_goal_markets(
    lambda_h: float,
    lambda_a: float,
    *,
    score_h: int = 0,
    score_a: int = 0,
    live: bool = False,
) -> dict[str, Any]:
    """All supported goal markets from model lambdas."""
    lines = [0.5, 1.5, 2.5, 3.5, 4.5]
    ou = {}
    for line in lines:
        key = str(line).replace(".", "_")
        pair = over_under_pair(lambda_h, lambda_a, line, score_h, score_a)
        ou[f"over_{key}"] = pair["over"]
        ou[f"under_{key}"] = pair["under"]

    btts_yes = both_teams_score_prob(lambda_h, lambda_a)
    btts_no = clamp_prob(1.0 - btts_yes)

    # Adjust BTTS for live current score
    if live and (score_h > 0 and score_a > 0):
        btts_yes = 1.0
        btts_no = 0.0
    elif live and score_h > 0:
        import math
        btts_yes = clamp_prob(1.0 - math.exp(-max(lambda_a, 0)))
        btts_no = clamp_prob(1.0 - btts_yes)
    elif live and score_a > 0:
        import math
        btts_yes = clamp_prob(1.0 - math.exp(-max(lambda_h, 0)))
        btts_no = clamp_prob(1.0 - btts_yes)

    matrix = build_score_matrix(lambda_h, lambda_a, score_h=score_h, score_a=score_a)

    home_o05 = team_over_prob(lambda_h, 0.5, score_h)
    home_o15 = team_over_prob(lambda_h, 1.5, score_h)
    away_o05 = team_over_prob(lambda_a, 0.5, score_a)
    away_o15 = team_over_prob(lambda_a, 1.5, score_a)

    return {
        **ou,
        "btts_yes": btts_yes,
        "btts_no": btts_no,
        "home_over_0_5": home_o05,
        "home_over_1_5": home_o15,
        "away_over_0_5": away_o05,
        "away_over_1_5": away_o15,
        "exact_score_top_5": matrix["top_exact_scores"],
        "outcomes": {
            "home_win": matrix["home_win"],
            "draw": matrix["draw"],
            "away_win": matrix["away_win"],
            "home_double_chance": clamp_prob(matrix["home_win"] + matrix["draw"]),
            "away_double_chance": clamp_prob(matrix["away_win"] + matrix["draw"]),
            "no_draw": clamp_prob(matrix["home_win"] + matrix["away_win"]),
        },
        "score_matrix_summary": {
            "top_exact_scores": matrix["top_exact_scores"],
            "total_goals": matrix["total_goals_probabilities"],
        },
        "live": live,
        "current_score": f"{score_h}-{score_a}",
    }


def model_prob_for_market_type(goal_markets: dict, market_type: str) -> float | None:
    """Lookup model probability for a market type key."""
    mt = market_type.lower()
    if mt.startswith("exact_score_"):
        score = mt.replace("exact_score_", "").replace("_", "-")
        for item in goal_markets.get("exact_score_top_5") or []:
            if item.get("score") == score:
                return item.get("probability")
        summary = (goal_markets.get("score_matrix_summary") or {}).get("top_exact_scores") or []
        for item in summary:
            if item.get("score") == score:
                return item.get("probability")
        return None
    if mt in goal_markets:
        return goal_markets[mt]
    outcomes = goal_markets.get("outcomes") or {}
    if mt in outcomes:
        return outcomes[mt]
    return None


def exact_score_market_type(score: str) -> str:
    """Kalshi mapping key for an exact scoreline, e.g. '3-0' -> 'exact_score_3_0'."""
    return "exact_score_" + score.replace("-", "_")
