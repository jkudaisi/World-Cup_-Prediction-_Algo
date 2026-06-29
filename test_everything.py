#!/usr/bin/env python3
"""
Full test runner for the World Cup prediction project.

Runs import checks, pytest suites by area, optional live API tests, and
optional smoke checks against a running local server.

Usage:
    python test_everything.py                 # all unit tests (mocked, no API key needed)
    python test_everything.py -v              # verbose pytest output
    python test_everything.py --integration   # also run live API-Football tests (needs key)
    python test_everything.py --smoke         # also ping http://127.0.0.1:5000 if server up
    python test_everything.py --preflight     # environment checks only, no pytest
    python test_everything.py --list          # list test sections
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).parent.resolve()

# ── Test sections (pytest paths relative to project root) ─────────────────────

SECTIONS: list[tuple[str, list[str]]] = [
    ("Team names & aliases", ["tests/test_team_names.py"]),
    ("Feature / model improvements", ["tests/test_model_improvements.py"]),
    ("API-Football client (mocked)", ["tests/test_apifootball_client.py"]),
    ("Local schedule", ["tests/test_local_schedule.py"]),
    ("API poll windows", ["tests/test_api_polling_window.py"]),
    ("Scheduler", ["tests/test_scheduler.py"]),
    ("Live predictor", ["tests/test_live_predictor.py"]),
    ("Live trainer", ["tests/test_live_trainer.py"]),
    ("Incremental trainer", ["tests/test_incremental_trainer.py"]),
    ("Future fixture cache", ["tests/test_future_fixture_predictions.py"]),
    ("Kalshi discovery", ["tests/test_kalshi_discovery.py"]),
    ("Kalshi account", ["tests/test_kalshi_account.py"]),
    ("Trading, guards & P/L", ["tests/test_trading.py"]),
    ("Flask server endpoints", ["tests/test_server_endpoints.py"]),
]

INTEGRATION_PATHS = ["tests/test_api_integration.py"]


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class RunSummary:
    preflight: list[CheckResult] = field(default_factory=list)
    sections: list[tuple[str, int, str]] = field(default_factory=list)  # name, exit_code, note
    smoke: list[CheckResult] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        if any(not c.ok for c in self.preflight):
            return True
        if any(code != 0 for _, code, _ in self.sections):
            return True
        if any(not c.ok for c in self.smoke):
            return True
        return False


def _print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def _run_pytest(
    paths: list[str],
    *,
    verbose: bool,
    extra_args: list[str] | None = None,
) -> int:
    cmd = [sys.executable, "-m", "pytest", *paths]
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode


def run_preflight() -> list[CheckResult]:
    """Quick environment and import checks (no network)."""
    results: list[CheckResult] = []

    def check(name: str, fn: Callable[[], tuple[bool, str]]) -> None:
        try:
            ok, detail = fn()
        except Exception as exc:
            ok, detail = False, str(exc)
        results.append(CheckResult(name, ok, detail))
        mark = "OK" if ok else "FAIL"
        line = f"  [{mark}] {name}"
        if detail:
            line += f" — {detail}"
        print(line)

    _print_header("Preflight checks")

    def import_core() -> tuple[bool, str]:
        modules = [
            "config",
            "server",
            "scheduler",
            "apifootball_client",
            "future_fixture_predictions",
            "wc2026_ml_pipeline",
            "incremental_trainer",
            "live_updater",
            "trading_service",
        ]
        for mod in modules:
            importlib.import_module(mod)
        return True, f"{len(modules)} core modules import cleanly"

    def config_key() -> tuple[bool, str]:
        import config

        has_key = bool(config.APIFOOTBALL_KEY)
        return True, "APIFOOTBALL_KEY set" if has_key else "APIFOOTBALL_KEY not set (unit tests still run)"

    def predictions_file() -> tuple[bool, str]:
        path = ROOT / "predictions.json"
        if not path.exists():
            return True, "predictions.json missing (optional until first train)"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        n = len(data.get("ml_data") or [])
        return True, f"{n} matches in predictions.json"

    def model_artifacts() -> tuple[bool, str]:
        from model_store import models_exist

        if models_exist():
            return True, "trained models present in models/"
        return True, "no models/ yet (run training or POST /api/run)"

    def future_cache() -> tuple[bool, str]:
        path = ROOT / "data" / "future_fixture_prediction_cache.json"
        if not path.exists():
            return True, "future cache not created yet (normal before first startup)"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        n = len(data.get("fixtures") or {})
        return True, f"{n} cached future fixture(s)"

    def pytest_available() -> tuple[bool, str]:
        import pytest  # noqa: F401

        return True, "pytest installed"

    check("Core imports", import_core)
    check("Config", config_key)
    check("predictions.json", predictions_file)
    check("Model artifacts", model_artifacts)
    check("Future fixture cache", future_cache)
    check("pytest", pytest_available)

    return results


def run_unit_sections(verbose: bool, summary: RunSummary) -> None:
    _print_header("Unit test suites (mocked — no live API required)")
    for name, paths in SECTIONS:
        print(f"\n--- {name} ---")
        code = _run_pytest(paths, verbose=verbose)
        note = "passed" if code == 0 else "FAILED"
        summary.sections.append((name, code, note))
        if code != 0:
            print(f"*** Section failed: {name} (exit {code})")


def run_integration(verbose: bool, summary: RunSummary) -> None:
    _print_header("Integration tests (live API-Football — requires APIFOOTBALL_KEY)")
    import config

    if not config.APIFOOTBALL_KEY:
        print("  [SKIP] APIFOOTBALL_KEY not set")
        summary.sections.append(("Integration (live API)", 0, "skipped — no key"))
        return

    code = _run_pytest(
        INTEGRATION_PATHS,
        verbose=verbose,
        extra_args=["-m", "integration"],
    )
    note = "passed" if code == 0 else "FAILED"
    summary.sections.append(("Integration (live API)", code, note))


def run_live_future_fixture_check(summary: RunSummary) -> None:
    """Optional live check: fetch season fixtures count (1 API call)."""
    import config

    if not config.APIFOOTBALL_KEY:
        return

    _print_header("Live future-fixture discovery (1 API call)")
    try:
        from future_fixture_predictions import fetch_future_world_cup_fixtures, is_fixture_predictable

        fixtures = fetch_future_world_cup_fixtures()
        predictable = sum(1 for fx in fixtures if is_fixture_predictable(fx)[0])
        detail = f"{len(fixtures)} upcoming WC fixture(s), {predictable} predictable now"
        ok = True
        print(f"  [OK] {detail}")
    except Exception as exc:
        ok = False
        detail = str(exc)
        print(f"  [FAIL] {detail}")

    summary.smoke.append(CheckResult("Future fixture fetch", ok, detail))


def run_smoke(base_url: str, summary: RunSummary) -> None:
    _print_header(f"Smoke tests — {base_url}")

    routes = [
        "/api/status",
        "/api/predictions",
        "/api/future-fixture-cache",
        "/api/today",
        "/api/scheduler",
        "/api/live",
        "/api/training-state",
    ]

    for path in routes:
        url = base_url.rstrip("/") + path
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                ok = resp.status == 200
                body = resp.read(500)
                detail = f"HTTP {resp.status}, {len(body)}+ bytes"
        except urllib.error.HTTPError as exc:
            ok = False
            detail = f"HTTP {exc.code}"
        except urllib.error.URLError as exc:
            ok = False
            detail = f"unreachable ({exc.reason})"
        except Exception as exc:
            ok = False
            detail = str(exc)

        mark = "OK" if ok else "FAIL"
        print(f"  [{mark}] GET {path} — {detail}")
        summary.smoke.append(CheckResult(f"GET {path}", ok, detail))


def print_summary(summary: RunSummary) -> None:
    _print_header("Summary")

    preflight_fail = sum(1 for c in summary.preflight if not c.ok)
    section_fail = sum(1 for _, code, _ in summary.sections if code != 0)
    smoke_fail = sum(1 for c in summary.smoke if not c.ok)
    section_pass = sum(1 for _, code, _ in summary.sections if code == 0)

    print(f"  Preflight : {len(summary.preflight) - preflight_fail}/{len(summary.preflight)} OK")
    print(f"  Sections  : {section_pass}/{len(summary.sections)} passed")
    for name, code, note in summary.sections:
        if code != 0:
            print(f"    - FAILED: {name} ({note})")
    if summary.smoke:
        print(f"  Smoke     : {len(summary.smoke) - smoke_fail}/{len(summary.smoke)} OK")

    total_tests = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if total_tests.returncode == 0 and total_tests.stdout.strip():
        last = total_tests.stdout.strip().splitlines()[-1]
        print(f"  Collected : {last}")

    print()
    if summary.failed:
        print("RESULT: SOME CHECKS FAILED")
    else:
        print("RESULT: ALL CHECKS PASSED")


def list_sections() -> None:
    print("Test sections:")
    for i, (name, paths) in enumerate(SECTIONS, 1):
        print(f"  {i:2}. {name}")
        for p in paths:
            print(f"      {p}")
    print("\nIntegration (with --integration):")
    for p in INTEGRATION_PATHS:
        print(f"      {p}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the full World Cup prediction test suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose pytest output")
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run live API-Football integration tests (uses API quota)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="HTTP smoke tests against a running server (default: http://127.0.0.1:5000)",
    )
    parser.add_argument(
        "--smoke-url",
        default="http://127.0.0.1:5000",
        help="Base URL for smoke tests (default: http://127.0.0.1:5000)",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run preflight checks only (no pytest)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List test sections and exit",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Single pytest run over tests/ (faster than section-by-section)",
    )
    args = parser.parse_args()

    if args.list:
        list_sections()
        return 0

    print(f"Project root: {ROOT}")
    print(f"Python:       {sys.executable}")

    summary = RunSummary()
    summary.preflight = run_preflight()

    if args.preflight:
        print_summary(summary)
        return 1 if summary.failed else 0

    if args.fast:
        _print_header("All unit tests (single pytest run)")
        code = _run_pytest(["tests/"], verbose=args.verbose, extra_args=["-m", "not integration"])
        summary.sections.append(("All unit tests", code, "passed" if code == 0 else "FAILED"))
    else:
        run_unit_sections(args.verbose, summary)

    if args.integration:
        run_integration(args.verbose, summary)
        run_live_future_fixture_check(summary)

    if args.smoke:
        run_smoke(args.smoke_url, summary)

    print_summary(summary)
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
