#!/usr/bin/env python3
"""
Run a full report of every API-Football client function and Flask endpoint.

Usage:
    python test_all_apis.py              # unit tests + live tests if key set
    python test_all_apis.py --unit-only  # mocked tests only (fast)
    python test_all_apis.py --live-only  # real API + server only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Test all APIs and endpoints")
    parser.add_argument("--unit-only", action="store_true", help="Run mocked unit tests only")
    parser.add_argument("--live-only", action="store_true", help="Run live integration tests only")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    cmd = [sys.executable, "-m", "pytest"]

    if args.live_only:
        cmd += ["tests/test_api_integration.py", "-m", "integration"]
    elif args.unit_only:
        cmd += [
            "tests/test_apifootball_client.py",
            "tests/test_server_endpoints.py",
            "tests/test_scheduler.py",
            "tests/test_live_trainer.py",
        ]
    else:
        cmd += ["tests/"]

    if args.verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")

    print("Running:", " ".join(cmd))
    print("-" * 60)
    result = subprocess.run(cmd, cwd=ROOT)
    print("-" * 60)

    if result.returncode == 0:
        print("All API tests passed.")
    else:
        print("Some tests failed.", file=sys.stderr)

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
