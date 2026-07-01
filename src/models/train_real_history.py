"""Production model training on real historical data only."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from incremental_trainer import wc_rows_to_frame
from model_store import save_artifacts
from src.data.guards import assert_no_synthetic_rows
from src.data.historical_store import load_real_training_frame, load_wc_completed_matches
from src.data.manifest import write_model_manifest, write_training_manifest
from src.config.pipeline_config import MODELS_REAL
from training_store import load_wc_matches
from wc2026_ml_pipeline import get_feature_cols, train_models_from_frame

log = logging.getLogger(__name__)


def build_real_training_frame(*, materialize_from_raw: bool = True) -> tuple[Any, int]:
    """Real API-Football rows only; raises if synthetic present."""
    if materialize_from_raw:
        from src.data.raw_backfill_training import materialize_training_dataset

        rows = materialize_training_dataset(merge_existing=True, write_matches=True)
    else:
        rows = load_wc_matches()
    if not rows:
        rows = load_wc_completed_matches()
    df = wc_rows_to_frame(rows)
    assert_no_synthetic_rows(df, context="build_real_training_frame")
    synthetic_count = int(df["is_synthetic"].sum()) if "is_synthetic" in df.columns else 0
    return df, synthetic_count


def train_real_history_models(*, verbose: bool = False) -> dict[str, Any]:
    df, synthetic_count = build_real_training_frame()
    if df.empty:
        raise ValueError("No real historical training rows found. Run bootstrap/backfill first.")

    feature_cols = get_feature_cols()
    assert_no_synthetic_rows(df, context="train_real_history_models")

    weights = None
    if "sample_weight" in df.columns:
        weights = df["sample_weight"].to_numpy(dtype=float)

    trained, scaler = train_models_from_frame(
        df, feature_cols, verbose=verbose, sample_weight=weights,
    )
    MODELS_REAL.mkdir(parents=True, exist_ok=True)
    versions = {name: "real-history-1" for name in trained}
    save_artifacts(trained, scaler, feature_cols, versions, models_dir=MODELS_REAL)

    dates = df["date"].dropna() if "date" in df.columns else []
    date_from = str(dates.min())[:10] if len(dates) else None
    date_to = str(dates.max())[:10] if len(dates) else None

    training_manifest = write_training_manifest(
        training_rows_count=len(df),
        synthetic_rows_count=synthetic_count,
        features_count=len(feature_cols),
        date_from=date_from,
        date_to=date_to,
        notes=["Production training on real API-Football history only"],
    )
    model_manifest = write_model_manifest(
        training_rows_count=len(df),
        synthetic_rows_count=0,
        features_count=len(feature_cols),
        notes=[f"Models saved to {MODELS_REAL}"],
    )

    return {
        "status": "success",
        "rows": len(df),
        "synthetic_rows": synthetic_count,
        "models_dir": str(MODELS_REAL),
        "training_manifest": str(training_manifest),
        "model_manifest": str(model_manifest),
    }
