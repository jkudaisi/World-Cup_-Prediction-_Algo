#!/usr/bin/env python3
"""
Backward-compatible wrapper — delegates to test_everything.py.

Usage:
    python test_all_apis.py              # all unit tests (by section)
    python test_all_apis.py --fast       # single pytest pass (quicker)
    python test_all_apis.py --live-only  # live API integration tests only
    python test_all_apis.py -v           # verbose
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Test all APIs and endpoints")
    parser.add_argument("--unit-only", action="store_true", help="Unit tests only (default)")
    parser.add_argument("--live-only", action="store_true", help="Live integration tests only")
    parser.add_argument("--fast", action="store_true", help="Single pytest run")
    parser.add_argument("-v", "--verbose", action="store_true")
    args, _unknown = parser.parse_known_args()

    if args.live_only:
        cmd = [sys.executable, "-m", "pytest", "tests/test_api_integration.py", "-m", "integration"]
        cmd += ["-v"] if args.verbose else ["-q"]
        print("Running:", " ".join(cmd))
        return subprocess.run(cmd, cwd=ROOT).returncode

    cmd = [sys.executable, str(ROOT / "test_everything.py")]
    if args.verbose:
        cmd.append("-v")
    if args.fast:
        cmd.append("--fast")

    print("Running:", " ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
