#!/usr/bin/env python3
"""End-to-end production refresh: data → train → predictions → Kalshi/API wiring."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.raw_backfill_training import materialize_training_dataset  # noqa: E402
from src.models.production_refresh import (  # noqa: E402
    refresh_production_predictions,
    verify_production_connectivity,
)
from src.models.train_real_history import train_real_history_models  # noqa: E402


def _run_script(rel_path: str) -> dict:
    proc = subprocess.run(
        [sys.executable, str(ROOT / rel_path)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    return {
        "script": rel_path,
        "exit_code": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh production ML + trading pipeline")
    parser.add_argument("--skip-materialize", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-gk", action="store_true")
    parser.add_argument("--skip-knockout", action="store_true")
    parser.add_argument("--skip-predictions", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    summary: dict = {"steps": []}

    if args.verify_only:
        report = verify_production_connectivity()
        print(json.dumps(report, indent=2))
        return 0 if report.get("ok") else 1

    if not args.skip_materialize:
        rows = materialize_training_dataset(merge_existing=True, write_matches=True)
        step = {"step": "materialize_training_data", "rows": len(rows)}
        summary["steps"].append(step)
        print(json.dumps(step))

    if not args.skip_train:
        result = train_real_history_models(verbose=args.verbose)
        summary["steps"].append({"step": "train_real_history", **result})
        print(json.dumps(result))

    if not args.skip_gk:
        gk = _run_script("scripts/build_goalkeeper_registry.py")
        summary["steps"].append({"step": "goalkeeper_registry", **gk})
        print(json.dumps(gk))

    if not args.skip_knockout:
        from knockout_models import train_knockout_models
        ko = train_knockout_models(use_api=True, force=True)
        summary["steps"].append({"step": "knockout_models", **ko})
        print(json.dumps(ko, default=str))

    if not args.skip_predictions:
        pred = refresh_production_predictions(refresh_future=True)
        summary["steps"].append({"step": "production_predictions", **pred})
        print(json.dumps(pred, default=str))

    report = verify_production_connectivity()
    summary["connectivity"] = report
    print(json.dumps(report, indent=2))

    out = ROOT / "data" / "manifests" / "production_refresh_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"Summary written to {out}")

    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
