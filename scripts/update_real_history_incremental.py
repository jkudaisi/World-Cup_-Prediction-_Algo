#!/usr/bin/env python3
"""Incremental update of real historical match store."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from incremental_trainer import run_incremental_training  # noqa: E402
from src.data.manifest import write_feature_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    result = run_incremental_training(force=args.force, fetch_from_api=True, verbose=args.verbose)
    write_feature_manifest(notes=["Incremental real-history update"])
    print(result)


if __name__ == "__main__":
    main()
