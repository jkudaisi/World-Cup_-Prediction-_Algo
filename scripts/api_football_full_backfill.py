#!/usr/bin/env python3
"""Full API-Football historical backfill (coverage-aware, resumable)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.providers.api_football.backfill import APIFootballBackfiller, load_backfill_config  # noqa: E402
from src.data.providers.api_football.client import APIFootballClient  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="API-Football full historical backfill")
    parser.add_argument("--from", dest="date_from", default=None)
    parser.add_argument("--to", dest="date_to", default="today")
    parser.add_argument("--teams", default="all_world_cup_teams")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--skip-coverage-discovery", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    config = load_backfill_config()
    if args.force_refresh:
        (config.setdefault("cache", {}))["force_refresh"] = True

    client = APIFootballClient(force_refresh=args.force_refresh)
    backfiller = APIFootballBackfiller(client=client, config=config, dry_run=args.dry_run)

    backfiller.run(
        date_from=args.date_from,
        date_to=args.date_to,
        run_discovery_first=not args.skip_coverage_discovery,
        resume=args.resume and not args.no_resume,
    )
    backfiller.print_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
