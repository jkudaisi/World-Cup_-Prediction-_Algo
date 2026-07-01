"""Block live entries when score already decides a market or quotes look stale."""

from __future__ import annotations

from position_outcomes import yes_leg_outcome
from trading_config import get_config


def block_live_entry(
    *,
    market_type: str,
    score_home: int = 0,
    score_away: int = 0,
    match_final: bool = False,
    model_yes: float | None = None,
    kalshi_yes: float | None = None,
    spread: float | None = None,
    is_live: bool = False,
) -> str | None:
    """
    Return a human-readable block reason, or None if entry is allowed.

    Applies during live play or whenever the match already has goals.
    """
    cfg = get_config()
    has_score = int(score_home) > 0 or int(score_away) > 0
    if not is_live and not has_score and not match_final:
        return None

    mt = (market_type or "").lower()
    yes_out = yes_leg_outcome(
        mt,
        score_home=int(score_home),
        score_away=int(score_away),
        match_final=bool(match_final),
    )
    if yes_out is not None:
        return "Market outcome already decided from live score"

    if mt.startswith(("btts_", "over_")) and spread is not None:
        if float(spread) >= cfg.wide_goal_spread_cents:
            return f"Spread too wide for live goal market ({float(spread):.0f}¢)"

    if model_yes is not None and kalshi_yes is not None:
        model_yes = float(model_yes)
        kalshi_yes = float(kalshi_yes)
        cutoff = cfg.model_disagree_cutoff
        if kalshi_yes >= cfg.kalshi_near_settled_high and model_yes <= cutoff:
            return "Kalshi YES near settlement but model disagrees (stale quote)"
        if kalshi_yes <= cfg.kalshi_near_settled_low and model_yes >= (1.0 - cutoff):
            return "Kalshi YES near zero but model disagrees (stale quote)"

    return None
