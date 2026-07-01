#!/usr/bin/env python3
"""Train production models on real historical data only."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.train_real_history import train_real_history_models  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    result = train_real_history_models(verbose=args.verbose)
    print(result)


if __name__ == "__main__":
    main()
