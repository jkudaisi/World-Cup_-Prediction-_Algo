#!/usr/bin/env python3
"""Discover API-Football endpoint coverage before full backfill."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.providers.api_football.backfill import load_teams_config, resolve_team_ids  # noqa: E402
from src.data.providers.api_football.client import APIFootballClient  # noqa: E402
from src.data.providers.api_football.coverage import APIFootballCoverageDiscoverer  # noqa: E402
from src.data.providers.api_football.manifest import build_backfill_manifest, write_manifest  # noqa: E402
from src.data.providers.api_football.paths import MANIFEST_ROOT  # noqa: E402
from src.data.providers.api_football.raw_store import APIFootballRawStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="API-Football coverage discovery")
    parser.add_argument("--from-year", type=int, default=2000)
    parser.add_argument("--to-year", type=int, default=2026)
    parser.add_argument("--teams", default="all_world_cup_teams")
    parser.add_argument("--sample-fixtures", type=int, default=5)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    client = APIFootballClient(force_refresh=args.force_refresh)
    store = APIFootballRawStore()
    discoverer = APIFootballCoverageDiscoverer(client, store, dry_run=args.dry_run)

    teams = load_teams_config()
    resolved, unresolved = resolve_team_ids(teams, client, dry_run=args.dry_run)

    team_reports = {}
    for team in resolved:
        tid = team.get("api_football_team_id")
        if tid is None:
            continue
        report = discoverer.discover_team_coverage(
            int(tid), args.from_year, args.to_year,
            team_name=team.get("name"),
            sample_fixtures=args.sample_fixtures,
        )
        team_reports[str(tid)] = report.to_dict()

    if not args.dry_run:
        store.save_coverage_report("endpoint_coverage.json", {"teams": team_reports})
        manifest = build_backfill_manifest(
            status="completed",
            teams_requested=len(teams),
            teams_resolved=len(resolved),
            endpoint_requests=client.endpoint_stats(),
            cache=client.cache_stats(),
            notes=["coverage discovery"],
        )
        write_manifest("coverage_manifest.json", manifest)

    print("Provider: api_football")
    print("Mode: coverage discovery" + (" (dry-run)" if args.dry_run else ""))
    print(f"Date range: {args.from_year} - {args.to_year}")
    print(f"Teams requested: {len(teams)}")
    print(f"Teams resolved: {len(resolved)}")
    print(f"Unresolved: {len(unresolved)}")
    print(f"Cache hits: {client.cache_stats()['hits']}")
    print(f"Cache misses: {client.cache_stats()['misses']}")
    print(f"Endpoint requests: {client.endpoint_stats()}")
    print(f"Output paths: {MANIFEST_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
