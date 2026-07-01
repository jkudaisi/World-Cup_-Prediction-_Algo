"""Manifest writers for pipeline runs."""
from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config.pipeline_config import DATA_MANIFESTS, MODELS_REAL, ROOT


def _code_version_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()[:12]
    except (OSError, subprocess.SubprocessError):
        pass
    # Fallback: hash key pipeline files
    h = hashlib.sha256()
    for rel in ("src/data/manifest.py", "src/features/feature_store.py", "incremental_trainer.py"):
        p = ROOT / rel
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


def build_manifest(
    *,
    source: str = "api-football",
    date_from: str | None = None,
    date_to: str | None = None,
    teams_count: int = 0,
    fixtures_count: int = 0,
    training_rows_count: int = 0,
    synthetic_rows_count: int = 0,
    features_count: int = 0,
    missing_data_summary: dict[str, Any] | None = None,
    notes: list[str] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id or new_run_id(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "date_range": {"from": date_from, "to": date_to},
        "teams_count": teams_count,
        "fixtures_count": fixtures_count,
        "training_rows_count": training_rows_count,
        "synthetic_rows_count": synthetic_rows_count,
        "features_count": features_count,
        "missing_data_summary": missing_data_summary or {},
        "code_version_hash": _code_version_hash(),
        "notes": notes or [],
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def write_backfill_manifest(**kwargs: Any) -> Path:
    return write_manifest(DATA_MANIFESTS / "backfill_manifest.json", build_manifest(**kwargs))


def write_feature_manifest(**kwargs: Any) -> Path:
    return write_manifest(DATA_MANIFESTS / "feature_manifest.json", build_manifest(**kwargs))


def write_training_manifest(**kwargs: Any) -> Path:
    return write_manifest(DATA_MANIFESTS / "training_manifest.json", build_manifest(**kwargs))


def write_model_manifest(**kwargs: Any) -> Path:
    return write_manifest(MODELS_REAL / "model_manifest.json", build_manifest(**kwargs))
