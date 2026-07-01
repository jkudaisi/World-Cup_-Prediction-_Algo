"""Unified feature store for training and live prediction (no separate paths)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from feature_builder import build_features
from src.features.injury_features import injury_impact_score
from src.features.lineup_features import merge_lineup_context_for_match
from src.data.api_football_backfill import load_raw
from src.features.relevance import compute_historical_relevance
from src.ratings.dynamic_team_state import DynamicTeamStateStore, TeamState


@dataclass
class MatchFeatureResult:
    feature_values: dict[str, float]
    missing_indicators: dict[str, bool] = field(default_factory=dict)
    data_quality_score: float = 1.0
    rating_state: dict[str, Any] = field(default_factory=dict)
    historical_relevance: dict[str, float] = field(default_factory=dict)
    competition_context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_values": self.feature_values,
            "missing_indicators": self.missing_indicators,
            "data_quality_score": self.data_quality_score,
            "rating_state": self.rating_state,
            "historical_relevance": self.historical_relevance,
            "competition_context": self.competition_context,
        }


# Module-level team state (rebuilt during backfill / incremental updates)
_team_state_store = DynamicTeamStateStore()


def get_team_state_store() -> DynamicTeamStateStore:
    return _team_state_store


def _resolve_team_names(home_team_id: int, away_team_id: int) -> tuple[str, str]:
    """Map API-Football team IDs to canonical names used by TEAM_STATS."""
    from apifootball_client import get_wc_team_id_map

    id_map = get_wc_team_id_map()  # name -> id
    rev = {v: k for k, v in id_map.items()}
    home = rev.get(home_team_id, str(home_team_id))
    away = rev.get(away_team_id, str(away_team_id))
    from team_names import resolve_team_name

    return resolve_team_name(home), resolve_team_name(away)


def enrich_context_from_raw(
    fixture_id: int | None,
    home_team_id: int,
    away_team_id: int,
    *,
    reference_fixture_id: int | None = None,
    base_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge lineup/injury signals from persisted raw API payloads."""
    ctx = dict(base_context or {})
    if fixture_id is None:
        return ctx

    cur_lineups = (load_raw("lineups", fixture_id) or {}).get("data")
    ref_lineups = None
    if reference_fixture_id is not None:
        ref_lineups = (load_raw("lineups", reference_fixture_id) or {}).get("data")

    lineup_ctx = merge_lineup_context_for_match(cur_lineups, ref_lineups)
    injuries_raw = (load_raw("injuries", fixture_id) or {}).get("data")
    injury_ctx = injury_impact_score(injuries_raw, home_team_id, away_team_id)

    stats_raw = (load_raw("statistics", fixture_id) or {}).get("data")
    players_raw = (load_raw("players", fixture_id) or {}).get("data")

    ctx.update({
        "has_lineups": lineup_ctx.get("has_lineups", False),
        "has_player_stats": bool(players_raw),
        "has_team_stats": bool(stats_raw),
        "has_injuries": injury_ctx.get("has_injuries", False),
        "starting_xi_overlap": lineup_ctx.get("starting_xi_overlap"),
        "same_gk": lineup_ctx.get("same_gk"),
        "injury_impact": injury_ctx.get("injury_impact", 0.0),
        "missing_injuries_flag": injury_ctx.get("missing_injuries_flag", True),
    })
    return ctx


def build_features_for_match(
    home_team_id: int,
    away_team_id: int,
    match_date: str | datetime,
    competition_context: dict[str, Any] | None = None,
    *,
    prior_matches: list[dict[str, Any]] | None = None,
    reference_match: dict[str, Any] | None = None,
) -> MatchFeatureResult:
    """
    Single entry point for historical training rows and future predictions.

    Features use only information available before kickoff when prior_matches
    is filtered to matches strictly before match_date (caller responsibility).
    """
    ctx = dict(competition_context or {})
    if prior_matches is None and ctx.get("fixture_id"):
        ctx = enrich_context_from_raw(
            int(ctx["fixture_id"]),
            home_team_id,
            away_team_id,
            reference_fixture_id=ctx.get("reference_fixture_id"),
            base_context=ctx,
        )
    home, away = _resolve_team_names(home_team_id, away_team_id)
    ref_date = match_date.isoformat() if isinstance(match_date, datetime) else str(match_date)

    feature_values = build_features(home, away, context=ctx)

    home_state = _team_state_store.get_or_create(home_team_id, home)
    away_state = _team_state_store.get_or_create(away_team_id, away)

    missing = {
        "missing_lineups_flag": not ctx.get("has_lineups", True),
        "missing_player_stats_flag": not ctx.get("has_player_stats", True),
        "missing_team_stats_flag": not ctx.get("has_team_stats", True),
        "missing_injuries_flag": ctx.get("missing_injuries_flag", not ctx.get("has_injuries", False)),
    }
    dq = min(
        1.0,
        0.55 if ctx.get("score_only") else 1.0,
        0.85 if missing["missing_player_stats_flag"] else 1.0,
        0.90 if missing["missing_lineups_flag"] else 1.0,
        0.75 if missing["missing_team_stats_flag"] else 1.0,
    )

    relevance: dict[str, float] = {}
    if reference_match:
        relevance = compute_historical_relevance(
            reference_match,
            ref_date,
            current_lineup_overlap=ctx.get("starting_xi_overlap"),
            same_coach=ctx.get("same_coach"),
            same_gk=ctx.get("same_gk"),
            opponent_elo=away_state.overall_rating,
        )

    return MatchFeatureResult(
        feature_values=feature_values,
        missing_indicators=missing,
        data_quality_score=dq,
        rating_state={
            "home": home_state.to_dict(),
            "away": away_state.to_dict(),
        },
        historical_relevance=relevance,
        competition_context=ctx,
    )


def filter_matches_before(
    matches: list[dict[str, Any]],
    cutoff: str | datetime,
    *,
    date_key: str = "date",
) -> list[dict[str, Any]]:
    """Leakage guard: keep only matches strictly before cutoff."""
    if isinstance(cutoff, datetime):
        cutoff_s = cutoff.isoformat()[:19]
    else:
        cutoff_s = str(cutoff)[:19]
    out = []
    for m in matches:
        d = str(m.get(date_key) or m.get("fixture_date") or "")[:19]
        if d and d < cutoff_s:
            out.append(m)
    return out
