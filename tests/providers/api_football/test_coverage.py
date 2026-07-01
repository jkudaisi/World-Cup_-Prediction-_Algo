from __future__ import annotations

from src.data.providers.api_football.models import CoverageFlags, compute_endpoint_coverage_score


def test_coverage_score_full():
    flags = CoverageFlags(
        has_fixtures=True,
        has_events=True,
        has_team_statistics=True,
        has_player_statistics=True,
        has_lineups=True,
        has_injuries=True,
    )
    assert compute_endpoint_coverage_score(flags) == 1.0


def test_coverage_score_score_only():
    flags = CoverageFlags(has_fixtures=True)
    assert compute_endpoint_coverage_score(flags) == 0.40


def test_missing_indicators():
    flags = CoverageFlags(has_fixtures=True, has_events=True)
    missing = flags.missing_indicators()
    assert missing["missing_lineups"] is True
    assert missing["missing_team_statistics"] is True
