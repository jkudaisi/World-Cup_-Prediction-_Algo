#!/usr/bin/env python3
"""Build goalkeeper registry from raw lineups, stats, and knockout pen history."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.goalkeeper_penalties import REGISTRY_PATH, build_registry_from_sources  # noqa: E402


def main() -> int:
    reg = build_registry_from_sources()
    print(f"Goalkeepers indexed: {len(reg.goalkeepers)}")
    print(f"Teams with pen history: {len(reg.team_pen_record)}")
    print(f"Registry: {REGISTRY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
