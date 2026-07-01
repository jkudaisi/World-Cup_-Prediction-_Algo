"""Production guards against synthetic training data."""
from __future__ import annotations

import os

import pandas as pd

from src.config.pipeline_config import PRODUCTION_ALLOW_SYNTHETIC


def is_production_training() -> bool:
    """True when production training path is active (default)."""
    if PRODUCTION_ALLOW_SYNTHETIC:
        return False
    env = os.environ.get("WC_ALLOW_SYNTHETIC_TRAINING", "").strip().lower()
    return env not in ("1", "true", "yes")


def assert_no_synthetic_rows(df: pd.DataFrame, *, context: str = "production training") -> None:
    """Raise if any row is marked synthetic during production training."""
    if not is_production_training():
        return
    if df is None or df.empty:
        return
    if "is_synthetic" not in df.columns:
        return
    if df["is_synthetic"].fillna(False).astype(bool).any():
        n = int(df["is_synthetic"].fillna(False).astype(bool).sum())
        raise ValueError(
            f"{context} cannot include synthetic rows ({n} found). "
            "Set WC_ALLOW_SYNTHETIC_TRAINING=1 only for tests."
        )


def mark_synthetic_rows(df: pd.DataFrame, is_synthetic: bool = True) -> pd.DataFrame:
    """Tag a dataframe with is_synthetic column."""
    out = df.copy()
    out["is_synthetic"] = bool(is_synthetic)
    return out
