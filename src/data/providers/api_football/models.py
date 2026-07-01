"""Data models for API-Football provider."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


PROVIDER_NAME = "api_football"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RawResponseWrapper:
    provider: str
    endpoint: str
    params: dict[str, Any]
    fetched_at: str
    cache_key: str
    success: bool
    error: str | None
    response_count: int
    api_response: Any

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def success_wrapper(
        cls,
        *,
        endpoint: str,
        params: dict[str, Any],
        cache_key: str,
        api_response: Any,
    ) -> RawResponseWrapper:
        count = len(api_response) if isinstance(api_response, list) else (1 if api_response else 0)
        return cls(
            provider=PROVIDER_NAME,
            endpoint=endpoint,
            params=params,
            fetched_at=utc_now_iso(),
            cache_key=cache_key,
            success=True,
            error=None,
            response_count=count,
            api_response=api_response,
        )

    @classmethod
    def failure_wrapper(
        cls,
        *,
        endpoint: str,
        params: dict[str, Any],
        cache_key: str,
        error: str,
    ) -> RawResponseWrapper:
        return cls(
            provider=PROVIDER_NAME,
            endpoint=endpoint,
            params=params,
            fetched_at=utc_now_iso(),
            cache_key=cache_key,
            success=False,
            error=error,
            response_count=0,
            api_response=None,
        )


@dataclass
class CoverageFlags:
    has_fixtures: bool = False
    has_events: bool = False
    has_team_statistics: bool = False
    has_player_statistics: bool = False
    has_lineups: bool = False
    has_injuries: bool = False
    has_standings: bool = False
    has_team_season_statistics: bool = False
    has_coach_data: bool = False
    has_squad_data: bool = False

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)

    def missing_indicators(self) -> dict[str, bool]:
        return {
            "missing_events": not self.has_events,
            "missing_team_statistics": not self.has_team_statistics,
            "missing_player_statistics": not self.has_player_statistics,
            "missing_lineups": not self.has_lineups,
            "missing_injuries": not self.has_injuries,
            "missing_goalkeeper_data": not self.has_lineups,
        }


def compute_endpoint_coverage_score(flags: dict[str, bool] | CoverageFlags) -> float:
    f = flags.to_dict() if isinstance(flags, CoverageFlags) else flags
    score = 0.40 if f.get("has_fixtures", True) else 0.0
    if f.get("has_events"):
        score += 0.10
    if f.get("has_team_statistics"):
        score += 0.20
    if f.get("has_player_statistics"):
        score += 0.15
    if f.get("has_lineups"):
        score += 0.10
    if f.get("has_injuries"):
        score += 0.05
    return min(score, 1.0)


@dataclass
class TeamCoverageReport:
    team_id: int
    team_name: str | None = None
    from_year: int = 2000
    to_year: int = 2026
    fixture_count: int = 0
    completed_fixture_count: int = 0
    sampled_fixture_ids: list[int] = field(default_factory=list)
    first_available_date: str | None = None
    last_available_date: str | None = None
    flags: CoverageFlags = field(default_factory=CoverageFlags)
    coverage_score: float = 0.0
    coverage_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["flags"] = self.flags.to_dict()
        d["missing_indicators"] = self.flags.missing_indicators()
        return d
