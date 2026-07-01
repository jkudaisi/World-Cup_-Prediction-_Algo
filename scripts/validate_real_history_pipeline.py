#!/usr/bin/env python3
"""Validate real-history pipeline guards and manifests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.pipeline_config import DATA_MANIFESTS, LEGACY_WC_MATCHES, MODELS_REAL  # noqa: E402
from src.data.guards import assert_no_synthetic_rows, is_production_training  # noqa: E402
from src.models.train_real_history import build_real_training_frame  # noqa: E402


def main() -> int:
    errors: list[str] = []
    print(f"production_mode={is_production_training()}")

    if not LEGACY_WC_MATCHES.exists():
        errors.append(f"Missing real match store: {LEGACY_WC_MATCHES}")
    else:
        try:
            df, syn = build_real_training_frame()
            assert_no_synthetic_rows(df)
            print(f"training_rows={len(df)} synthetic_rows={syn}")
        except Exception as exc:
            errors.append(str(exc))

    for name in ("backfill_manifest.json", "feature_manifest.json", "training_manifest.json"):
        p = DATA_MANIFESTS / name
        if p.exists():
            json.loads(p.read_text(encoding="utf-8"))
            print(f"ok manifest: {p}")
        else:
            print(f"missing manifest (optional): {p}")

    if MODELS_REAL.exists() and (MODELS_REAL / "meta.json").exists():
        print(f"ok models: {MODELS_REAL}")
    else:
        print(f"models not yet trained: {MODELS_REAL}")

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("VALIDATION OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
