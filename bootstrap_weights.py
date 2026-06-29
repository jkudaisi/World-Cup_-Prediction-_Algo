"""Recency and competition sample weights for historical bootstrap training."""

from __future__ import annotations

from datetime import date, datetime

COMPETITION_WEIGHTS: dict[int, float] = {
    1: 2.0,    # World Cup
    4: 1.6,    # Euro Championship
    6: 1.6,    # AFCON
    9: 1.6,    # Copa America
    10: 1.6,   # CONCACAF Gold Cup
    29: 1.4,   # UEFA Euro Qualification
    30: 1.4,   # CONMEBOL Qualification
    31: 1.4,   # CONCACAF Qualification
    32: 1.4,   # AFC Asian Qualification
    34: 1.4,   # CAF Qualification
    848: 1.2,  # UEFA Nations League
    15: 0.5,   # FIFA Friendlies
}

DEFAULT_COMPETITION_WEIGHT = 0.8
MIN_WEIGHT = 0.1
MAX_WEIGHT = 2.0
INVALID_RECENCY_WEIGHT = 0.5


def _parse_date(date_str: str) -> date | None:
    if not date_str:
        return None
    text = str(date_str).strip()[:10]
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None


def recency_weight(date_str: str, *, today: date | None = None) -> float:
    """Days since match date → recency multiplier."""
    match_day = _parse_date(date_str)
    if match_day is None:
        return INVALID_RECENCY_WEIGHT
    ref = today or date.today()
    days = max(0, (ref - match_day).days)
    if days <= 365:
        return 1.00
    if days <= 730:
        return 0.85
    if days <= 1460:
        return 0.65
    if days <= 2190:
        return 0.45
    return 0.30


def competition_weight(league_id: int) -> float:
    try:
        lid = int(league_id)
    except (TypeError, ValueError):
        return DEFAULT_COMPETITION_WEIGHT
    return COMPETITION_WEIGHTS.get(lid, DEFAULT_COMPETITION_WEIGHT)


def compute_sample_weight(date_str: str, league_id: int) -> float:
    """Final bootstrap weight = recency × competition, capped."""
    raw = recency_weight(date_str) * competition_weight(league_id)
    return round(max(MIN_WEIGHT, min(MAX_WEIGHT, raw)), 3)
