#!/usr/bin/env python3
"""
Safe reset for real-history migration.

Archives synthetic training artifacts and clears stale caches.
Preserves .env, credentials, source code, and user docs.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.pipeline_config import (  # noqa: E402
    DATA_ARCHIVE,
    DATA_CACHE,
    DATA_PROCESSED,
    DATA_RAW,
    LEGACY_BASE_CACHE,
    LEGACY_MODELS,
    MODELS_REAL,
)
from src.data.cache_manager import clear_cache, ensure_raw_dirs  # noqa: E402
from src.data.manifest import write_backfill_manifest  # noqa: E402

# Paths known to hold synthetic or pre-migration training data
SYNTHETIC_CANDIDATES = [
    LEGACY_BASE_CACHE,
]

STALE_PROCESSED_GLOBS = [
    DATA_PROCESSED / "features",
    DATA_PROCESSED / "training",
    DATA_PROCESSED / "ratings",
]

LEGACY_MODEL_GLOBS = ["*.pkl", "meta.json"]


def _archive_file(src: Path, archive_root: Path, actions: list[str]) -> None:
    if not src.exists():
        return
    rel = src.relative_to(ROOT) if src.is_relative_to(ROOT) else Path(src.name)
    dest = archive_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    actions.append(f"archived: {src} -> {dest}")


def _remove_tree(src: Path, archive_root: Path, actions: list[str]) -> None:
    if not src.exists():
        return
    if src.is_file():
        _archive_file(src, archive_root, actions)
        return
    for f in sorted(src.rglob("*")):
        if f.is_file():
            _archive_file(f, archive_root, actions)
    shutil.rmtree(src, ignore_errors=True)
    actions.append(f"removed directory: {src}")


def run_reset(*, dry_run: bool = False, skip_backfill: bool = False) -> dict:
    archive_root = DATA_ARCHIVE / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    actions: list[str] = []

    # 1. Archive synthetic training cache
    for p in SYNTHETIC_CANDIDATES:
        if p.exists():
            if dry_run:
                actions.append(f"would archive: {p}")
            else:
                _archive_file(p, archive_root, actions)

    # 2. Clear stale API cache (archive first)
    if dry_run:
        if DATA_CACHE.exists():
            actions.append(f"would clear cache: {DATA_CACHE}")
    else:
        for f in DATA_CACHE.rglob("*") if DATA_CACHE.exists() else []:
            if f.is_file():
                _archive_file(f, archive_root, actions)
        removed = clear_cache()
        actions.extend(f"cleared cache file: {x}" for x in removed)

    # 3. Clear stale processed features
    for d in STALE_PROCESSED_GLOBS:
        if dry_run:
            if d.exists():
                actions.append(f"would archive processed: {d}")
        else:
            _remove_tree(d, archive_root, actions)

    # 4. Archive legacy root models (trained on synthetic-era pipeline)
    if LEGACY_MODELS.exists():
        for pattern in LEGACY_MODEL_GLOBS:
            for f in LEGACY_MODELS.glob(pattern):
                if dry_run:
                    actions.append(f"would archive model: {f}")
                else:
                    _archive_file(f, archive_root, actions)

    if not dry_run:
        ensure_raw_dirs()
        (DATA_PROCESSED / "features").mkdir(parents=True, exist_ok=True)
        (DATA_PROCESSED / "training").mkdir(parents=True, exist_ok=True)
        (DATA_PROCESSED / "ratings").mkdir(parents=True, exist_ok=True)
        MODELS_REAL.mkdir(parents=True, exist_ok=True)
        (ROOT / "data" / "manifests").mkdir(parents=True, exist_ok=True)

        reset_manifest = {
            "run_id": archive_root.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "archive_path": str(archive_root),
            "actions": actions,
            "preserved": [".env", "source code", "Kalshi credentials", "user settings"],
        }
        manifest_path = ROOT / "data" / "manifests" / "reset_manifest.json"
        manifest_path.write_text(json.dumps(reset_manifest, indent=2), encoding="utf-8")
        actions.append(f"wrote: {manifest_path}")

        if not skip_backfill:
            try:
                from historical_bootstrap import STATE_PATH, run_bootstrap  # type: ignore

                if STATE_PATH.exists():
                    STATE_PATH.unlink()
                    actions.append(f"reset bootstrap state: {STATE_PATH}")
                actions.append("starting historical bootstrap backfill...")
                summary = run_bootstrap()
                write_backfill_manifest(notes=["Post-reset bootstrap"], fixtures_count=summary.get("rows_added", 0))
                actions.append(f"backfill complete: {summary}")
            except Exception as exc:
                actions.append(f"backfill skipped/failed: {exc}")

    for line in actions:
        print(line)

    return {"archive_root": str(archive_root), "actions": actions, "dry_run": dry_run}


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset and backfill real historical data")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without changing files")
    parser.add_argument("--skip-backfill", action="store_true", help="Archive/clear only; do not run bootstrap")
    args = parser.parse_args()
    run_reset(dry_run=args.dry_run, skip_backfill=args.skip_backfill)


if __name__ == "__main__":
    main()
