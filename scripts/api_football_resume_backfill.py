#!/usr/bin/env python3
"""Resume a partial API-Football backfill from resume_state.json."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.providers.api_football.backfill import APIFootballBackfiller, load_backfill_config  # noqa: E402
from src.data.providers.api_football.client import APIFootballClient  # noqa: E402
from src.data.providers.api_football.manifest import load_resume_state  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume API-Football backfill")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    state = load_resume_state()
    pending = len(state.get("pending_fixture_ids") or [])
    completed = len(state.get("completed_fixture_ids") or [])

    print("Provider: api_football")
    print("Mode: resume backfill" + (" (dry-run)" if args.dry_run else ""))
    print(f"Resume checkpoint: {state.get('last_successful_checkpoint_at')}")
    print(f"Completed fixtures: {completed}")
    print(f"Pending fixtures: {pending}")

    config = load_backfill_config()
    if args.force_refresh:
        (config.setdefault("cache", {}))["force_refresh"] = True

    client = APIFootballClient(force_refresh=args.force_refresh)
    backfiller = APIFootballBackfiller(client=client, config=config, dry_run=args.dry_run)
    backfiller.run(resume=True)
    backfiller.print_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
