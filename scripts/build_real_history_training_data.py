#!/usr/bin/env python3
"""Materialize training rows from the API-Football raw backfill lake."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.raw_backfill_training import (  # noqa: E402
    TRAINING_DATA_MANIFEST,
    materialize_training_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build training data from raw API-Football backfill")
    parser.add_argument("--no-merge", action="store_true", help="Replace dataset with backfill rows only")
    parser.add_argument(
        "--wc-only",
        action="store_true",
        help="Only include matches where both teams are in the WC 2026 pool",
    )
    parser.add_argument("--no-chronological", action="store_true", help="Use static team priors instead of rolling state")
    parser.add_argument("--dry-run", action="store_true", help="Build rows but do not write matches JSON")
    args = parser.parse_args()

    rows = materialize_training_dataset(
        merge_existing=not args.no_merge,
        write_matches=not args.dry_run,
        allow_external_opponents=not args.wc_only,
        chronological_features=not args.no_chronological,
    )
    print(json.dumps({
        "rows": len(rows),
        "manifest": str(TRAINING_DATA_MANIFEST),
        "dry_run": args.dry_run,
        "merge_existing": not args.no_merge,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
