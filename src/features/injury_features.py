"""Injury impact features from raw API-Football injury data."""
from __future__ import annotations

from typing import Any


def count_team_injuries(injuries: list[dict[str, Any]] | None, team_id: int | None) -> int:
    if not injuries or team_id is None:
        return 0
    n = 0
    for item in injuries:
        tid = (item.get("team") or {}).get("id")
        if tid == team_id:
            n += 1
    return n


def injury_impact_score(
    injuries: list[dict[str, Any]] | None,
    home_team_id: int | None,
    away_team_id: int | None,
) -> dict[str, Any]:
    """
    Higher impact when more players listed injured.
    Returns per-side counts and normalized impact in [0, 1].
    """
    home_n = count_team_injuries(injuries, home_team_id)
    away_n = count_team_injuries(injuries, away_team_id)
    # Cap at 5 starters worth of impact
    home_impact = min(1.0, home_n / 5.0)
    away_impact = min(1.0, away_n / 5.0)
    return {
        "has_injuries": bool(injuries),
        "home_injury_count": home_n,
        "away_injury_count": away_n,
        "home_injury_impact": home_impact,
        "away_injury_impact": away_impact,
        "injury_impact": max(home_impact, away_impact),
        "missing_injuries_flag": injuries is None,
    }
