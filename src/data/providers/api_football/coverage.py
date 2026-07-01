"""Coverage discovery for API-Football."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from src.data.providers.api_football.client import APIFootballClient, APIFootballClientError
from src.data.providers.api_football.models import (
    CoverageFlags,
    TeamCoverageReport,
    compute_endpoint_coverage_score,
)
from src.data.providers.api_football.paths import COMPLETED_STATUSES
from src.data.providers.api_football.raw_store import APIFootballRawStore

log = logging.getLogger(__name__)


def _has_data(response: Any) -> bool:
    if response is None:
        return False
    if isinstance(response, list):
        return len(response) > 0
    if isinstance(response, dict):
        return bool(response)
    return True


class APIFootballCoverageDiscoverer:
    def __init__(
        self,
        client: APIFootballClient | None = None,
        raw_store: APIFootballRawStore | None = None,
        *,
        dry_run: bool = False,
    ):
        self.client = client or APIFootballClient()
        self.raw_store = raw_store or APIFootballRawStore()
        self.dry_run = dry_run

    def discover_fixture_endpoint_coverage(self, fixture_id: int) -> dict[str, Any]:
        flags = CoverageFlags(has_fixtures=True)
        notes: list[str] = []
        endpoints = {
            "has_events": ("/fixtures/events", lambda: self.client.get_fixture_events(fixture_id)),
            "has_team_statistics": ("/fixtures/statistics", lambda: self.client.get_fixture_statistics(fixture_id)),
            "has_player_statistics": ("/fixtures/players", lambda: self.client.get_fixture_players(fixture_id)),
            "has_lineups": ("/fixtures/lineups", lambda: self.client.get_fixture_lineups(fixture_id)),
            "has_injuries": ("/injuries", lambda: self.client.get_injuries(fixture=fixture_id)),
        }

        for flag_name, (endpoint, fetcher) in endpoints.items():
            try:
                if self.dry_run:
                    setattr(flags, flag_name, False)
                    notes.append(f"dry-run skip {endpoint}")
                    continue
                data = fetcher()
                ok = _has_data(data)
                setattr(flags, flag_name, ok)
                if not ok:
                    notes.append(f"{endpoint} returned no data for fixture {fixture_id}")
                    if not self.dry_run:
                        self.raw_store.save_endpoint_response(
                            self._kind_for_flag(flag_name),
                            str(fixture_id),
                            endpoint=endpoint,
                            params={"fixture": fixture_id},
                            api_response=data,
                            success=False,
                            error="endpoint returned no data",
                        )
                        self.raw_store.append_missing_endpoint_log({
                            "fixture_id": fixture_id,
                            "endpoint": endpoint,
                            "reason": "no data",
                        })
                elif not self.dry_run:
                    self.raw_store.save_endpoint_response(
                        self._kind_for_flag(flag_name),
                        str(fixture_id),
                        endpoint=endpoint,
                        params={"fixture": fixture_id},
                        api_response=data,
                        success=True,
                    )
            except APIFootballClientError as exc:
                notes.append(f"{endpoint} failed: {exc}")
                setattr(flags, flag_name, False)
                if not self.dry_run:
                    self.raw_store.append_missing_endpoint_log({
                        "fixture_id": fixture_id,
                        "endpoint": endpoint,
                        "reason": str(exc),
                    })

        score = compute_endpoint_coverage_score(flags)
        report = {
            "fixture_id": fixture_id,
            "flags": flags.to_dict(),
            "missing_indicators": flags.missing_indicators(),
            "coverage_score": score,
            "coverage_notes": notes,
        }
        return report

    @staticmethod
    def _kind_for_flag(flag_name: str) -> str:
        mapping = {
            "has_events": "events",
            "has_team_statistics": "statistics",
            "has_player_statistics": "players",
            "has_lineups": "lineups",
            "has_injuries": "injuries",
        }
        return mapping.get(flag_name, "events")

    def discover_team_coverage(
        self,
        team_id: int,
        from_year: int,
        to_year: int,
        *,
        team_name: str | None = None,
        sample_fixtures: int = 5,
    ) -> TeamCoverageReport:
        report = TeamCoverageReport(
            team_id=team_id,
            team_name=team_name,
            from_year=from_year,
            to_year=to_year,
        )
        fixtures_by_id: dict[int, dict] = {}
        dates: list[str] = []

        for year in range(from_year, to_year + 1):
            try:
                batch = self.client.get_fixtures(team=team_id, season=year) or []
            except APIFootballClientError as exc:
                report.coverage_notes.append(f"fixtures {year}: {exc}")
                continue
            if not isinstance(batch, list):
                continue
            for fx in batch:
                fid = (fx.get("fixture") or {}).get("id")
                if fid is None:
                    continue
                fixtures_by_id[int(fid)] = fx
                d = (fx.get("fixture") or {}).get("date")
                if d:
                    dates.append(str(d)[:10])

        report.fixture_count = len(fixtures_by_id)
        completed = [
            fid for fid, fx in fixtures_by_id.items()
            if (fx.get("fixture") or {}).get("status", {}).get("short") in COMPLETED_STATUSES
        ]
        report.completed_fixture_count = len(completed)
        if dates:
            report.first_available_date = min(dates)
            report.last_available_date = max(dates)

        report.flags.has_fixtures = report.fixture_count > 0
        sample_ids = completed[:sample_fixtures] if completed else list(fixtures_by_id.keys())[:sample_fixtures]
        report.sampled_fixture_ids = sample_ids

        if sample_ids and not self.dry_run:
            agg = CoverageFlags(has_fixtures=True)
            for fid in sample_ids:
                fx_cov = self.discover_fixture_endpoint_coverage(fid)
                fflags = fx_cov["flags"]
                for k, v in fflags.items():
                    if v and hasattr(agg, k):
                        setattr(agg, k, True)
            report.flags = agg
        elif sample_ids and self.dry_run:
            report.coverage_notes.append(f"would sample {len(sample_ids)} fixtures")

        report.coverage_score = compute_endpoint_coverage_score(report.flags)
        return report

    def discover_competition_coverage(self, league_id: int, season: int) -> dict[str, Any]:
        flags = CoverageFlags()
        notes: list[str] = []
        fixture_ids: list[int] = []

        try:
            fixtures = self.client.get_fixtures(league=league_id, season=season) or []
            if isinstance(fixtures, list):
                flags.has_fixtures = len(fixtures) > 0
                for fx in fixtures:
                    fid = (fx.get("fixture") or {}).get("id")
                    if fid is not None:
                        fixture_ids.append(int(fid))
        except APIFootballClientError as exc:
            notes.append(f"fixtures: {exc}")

        try:
            standings = self.client.get_standings(league=league_id, season=season)
            flags.has_standings = _has_data(standings)
        except APIFootballClientError as exc:
            notes.append(f"standings: {exc}")

        completed_sample = [
            fid for fid in fixture_ids[:5]
        ]
        if completed_sample and not self.dry_run:
            for fid in completed_sample[:3]:
                fx_cov = self.discover_fixture_endpoint_coverage(fid)
                for k, v in fx_cov["flags"].items():
                    if v and hasattr(flags, k):
                        setattr(flags, k, True)

        report = {
            "league_id": league_id,
            "season": season,
            "fixture_count": len(fixture_ids),
            "sampled_fixture_ids": completed_sample[:5],
            "flags": flags.to_dict(),
            "missing_indicators": flags.missing_indicators(),
            "coverage_score": compute_endpoint_coverage_score(flags),
            "coverage_notes": notes,
            "discovered_at": datetime.now(timezone.utc).isoformat(),
        }
        if not self.dry_run:
            self.raw_store.save_coverage_report("competition_coverage.json", {
                "reports": {f"{league_id}_{season}": report},
            })
        return report
