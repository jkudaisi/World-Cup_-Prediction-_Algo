#!/usr/bin/env python3
"""Backfill raw API-Football layers for fixtures already in the training store."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.api_football_backfill import fetch_and_persist_fixture, load_raw  # noqa: E402
from training_store import load_wc_matches  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill raw fixture payloads from API-Football")
    parser.add_argument("--limit", type=int, default=50, help="Max fixtures to fetch")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if raw exists")
    args = parser.parse_args()

    rows = load_wc_matches()
    fetched = 0
    skipped = 0
    for row in rows:
        if fetched >= args.limit:
            break
        fid = row.get("fixture_id")
        if fid is None:
            continue
        fid = int(fid)
        if not args.force and load_raw("statistics", fid):
            skipped += 1
            continue
        home_id = row.get("home_team_id")
        away_id = row.get("away_team_id")
        paths = fetch_and_persist_fixture(fid, home_team_id=home_id, away_team_id=away_id)
        print(f"fixture {fid}: {len(paths)} files")
        fetched += 1

    print(f"done: fetched={fetched} skipped={skipped}")


if __name__ == "__main__":
    main()
