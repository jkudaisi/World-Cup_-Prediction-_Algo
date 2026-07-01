#!/usr/bin/env python3
"""Extend API-Football backfill with pre-2018 seasons and backfill new fixtures."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.providers.api_football.extend_backfill import APIFootballBackfillExtender  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Extend API-Football backfill to earlier seasons")
    parser.add_argument("--from-year", type=int, default=2000)
    parser.add_argument("--to-year", type=int, default=2017)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.from_year > args.to_year:
        print("from-year must be <= to-year", file=sys.stderr)
        return 1

    extender = APIFootballBackfillExtender(dry_run=args.dry_run)
    extender.run(from_year=args.from_year, to_year=args.to_year)
    extender.print_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
