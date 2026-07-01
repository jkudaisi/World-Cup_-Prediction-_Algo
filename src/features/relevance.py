"""Historical relevance scoring for sample weights."""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from src.config import pipeline_config as cfg


def _days_between(match_date: datetime | str, reference_date: datetime | str) -> int:
    if isinstance(match_date, str):
        match_date = datetime.fromisoformat(match_date.replace("Z", "+00:00")[:19])
    if isinstance(reference_date, str):
        reference_date = datetime.fromisoformat(reference_date.replace("Z", "+00:00")[:19])
    if hasattr(match_date, "tzinfo") and match_date.tzinfo:
        match_date = match_date.replace(tzinfo=None)
    if hasattr(reference_date, "tzinfo") and reference_date.tzinfo:
        reference_date = reference_date.replace(tzinfo=None)
    return max(0, (reference_date - match_date).days)


def recency_weight(days_since_match: int, method: str | None = None, decay_days: float | None = None) -> float:
    method = method or cfg.RECENCY_METHOD
    if method == "exponential":
        d = decay_days if decay_days is not None else cfg.RECENCY_DECAY_DAYS
        return math.exp(-days_since_match / d)
    for lo, hi, w in cfg.RECENCY_WEIGHT_TABLE:
        if lo <= days_since_match < hi:
            return w
    return 0.20


def world_cup_cycle_weight(match_year: int, reference_year: int = 2026) -> float:
    """Weight by World Cup cycle relative to reference tournament."""
    for start, end, w in cfg.WC_CYCLE_WEIGHTS:
        if start <= match_year <= end:
            return w
    return 0.20


def competition_weight(competition_type: str | None = None, league_id: int | None = None) -> float:
    if competition_type:
        key = competition_type.lower().replace(" ", "_")
        if key in cfg.COMPETITION_WEIGHTS:
            return cfg.COMPETITION_WEIGHTS[key]
    # API-Football league_id heuristics
    if league_id == 1:
        return cfg.COMPETITION_WEIGHTS["world_cup_group"]
    return cfg.COMPETITION_WEIGHTS["default"]


def coach_similarity_weight(same_coach: bool | None) -> float:
    if same_coach is True:
        return cfg.COACH_SAME
    if same_coach is False:
        return cfg.COACH_DIFFERENT
    return cfg.COACH_UNKNOWN


def goalkeeper_continuity_weight(same_gk: bool | None) -> float:
    if same_gk is True:
        return cfg.GK_SAME
    if same_gk is False:
        return cfg.GK_DIFFERENT
    return cfg.GK_UNKNOWN


def lineup_similarity_weight(starting_xi_overlap: float | None) -> float:
    if starting_xi_overlap is None:
        return 0.90
    return max(0.0, min(1.0, starting_xi_overlap / 11.0))


def data_quality_weight(
    *,
    has_lineups: bool = True,
    has_player_stats: bool = True,
    has_team_stats: bool = True,
    score_only: bool = False,
) -> tuple[float, dict[str, bool]]:
    flags = {
        "missing_lineups_flag": not has_lineups,
        "missing_player_stats_flag": not has_player_stats,
        "missing_team_stats_flag": not has_team_stats,
        "missing_injuries_flag": True,  # default unknown until injury feed wired
    }
    if score_only:
        return cfg.DATA_QUALITY_SCORE_ONLY, flags
    score = cfg.DATA_QUALITY_FULL
    if not has_player_stats:
        score = min(score, cfg.DATA_QUALITY_MISSING_PLAYER_STATS)
    if not has_lineups:
        score = min(score, cfg.DATA_QUALITY_MISSING_LINEUPS)
    if not has_team_stats:
        score = min(score, cfg.DATA_QUALITY_MISSING_TEAM_STATS)
    return score, flags


def opponent_strength_weight(opponent_elo: float | None, baseline: float = 1500.0) -> float:
    if opponent_elo is None:
        return 1.0
    # Slightly upweight matches vs stronger opponents
    delta = (opponent_elo - baseline) / 400.0
    return max(0.7, min(1.3, 1.0 + 0.15 * delta))


def compute_historical_relevance(
    match: dict[str, Any],
    reference_date: datetime | str,
    *,
    current_lineup_overlap: float | None = None,
    same_coach: bool | None = None,
    same_gk: bool | None = None,
    opponent_elo: float | None = None,
    recency_method: str | None = None,
) -> dict[str, float]:
    """Return component weights and final historical_relevance_score."""
    match_date = match.get("date") or match.get("fixture_date")
    days = _days_between(match_date, reference_date)
    match_year = datetime.fromisoformat(str(match_date)[:10]).year if match_date else 2000

    comp_w = competition_weight(match.get("competition_type"), match.get("league_id"))
    rec_w = recency_weight(days, method=recency_method)
    cycle_w = world_cup_cycle_weight(match_year)
    lineup_w = lineup_similarity_weight(current_lineup_overlap)
    coach_w = coach_similarity_weight(same_coach)
    gk_w = goalkeeper_continuity_weight(same_gk)
    opp_w = opponent_strength_weight(opponent_elo)

    dq, _flags = data_quality_weight(
        has_lineups=match.get("has_lineups", True),
        has_player_stats=match.get("has_player_stats", True),
        has_team_stats=match.get("has_team_stats", True),
        score_only=match.get("score_only", False),
    )

    final = comp_w * rec_w * cycle_w * lineup_w * coach_w * gk_w * opp_w * dq
    return {
        "competition_weight": comp_w,
        "recency_weight": rec_w,
        "world_cup_cycle_weight": cycle_w,
        "lineup_similarity_weight": lineup_w,
        "coach_similarity_weight": coach_w,
        "goalkeeper_continuity_weight": gk_w,
        "opponent_strength_weight": opp_w,
        "data_quality_weight": dq,
        "historical_relevance_score": final,
    }
