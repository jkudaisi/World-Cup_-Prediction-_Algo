"""Determine when a held Kalshi position's outcome is decided from match state."""

from __future__ import annotations

FINAL_STATUSES = frozenset({"FT", "AET", "PEN"})


def fixture_score_status(fixture: dict | None) -> tuple[int, int, str, bool]:
    """Return (score_home, score_away, match_status, is_final) from a trading fixture row."""
    if not fixture:
        return 0, 0, "NS", False
    sh = fixture.get("score_home")
    sa = fixture.get("score_away")
    status = (fixture.get("match_status") or "NS").upper()
    is_final = bool(fixture.get("match_final")) or status in FINAL_STATUSES
    return int(sh or 0), int(sa or 0), status, is_final


def yes_leg_outcome(
    market_type: str,
    *,
    score_home: int,
    score_away: int,
    match_final: bool,
) -> bool | None:
    """
    Whether the YES leg of the market won.

    Returns True/False when decided, None when still open.
    """
    mt = (market_type or "").lower()
    total = score_home + score_away

    if mt == "btts_yes":
        if score_home > 0 and score_away > 0:
            return True
        if match_final:
            return False
        return None

    if mt.startswith("over_"):
        try:
            line = float(mt.replace("over_", "").replace("_", "."))
        except ValueError:
            return None
        if total > line:
            return True
        if match_final:
            return False
        return None

    if mt in ("home_win", "draw", "away_win"):
        if not match_final:
            return None
        if score_home > score_away:
            winner = "home_win"
        elif score_away > score_home:
            winner = "away_win"
        else:
            winner = "draw"
        return mt == winner

    return None


def side_won(*, side: str, yes_won: bool) -> bool:
    side = (side or "yes").lower()
    return yes_won if side == "yes" else not yes_won


def evaluate_position_outcome(position: dict, fixture: dict | None) -> dict | None:
    """
    If the position outcome is decided, return {won, yes_won, reason, score}.
    Otherwise return None.
    """
    if not fixture:
        return None

    sh, sa, _status, is_final = fixture_score_status(fixture)
    yes_won = yes_leg_outcome(
        position.get("market_type", ""),
        score_home=sh,
        score_away=sa,
        match_final=is_final,
    )
    if yes_won is None:
        return None

    side = (position.get("side") or "yes").lower()
    won = side_won(side=side, yes_won=yes_won)
    mt = (position.get("market_type") or "").lower()
    if mt == "btts_yes" and yes_won:
        reason = "btts_both_scored"
    elif is_final:
        reason = "match_final"
    else:
        reason = "event_outcome"

    return {
        "won": won,
        "yes_won": yes_won,
        "reason": reason,
        "score": f"{sh}-{sa}",
    }
