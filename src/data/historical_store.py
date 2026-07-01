"""Load real historical match rows from disk."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.pipeline_config import LEGACY_WC_MATCHES
from src.data.guards import mark_synthetic_rows


def load_wc_completed_matches(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or LEGACY_WC_MATCHES
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "matches" in data:
        return data["matches"]
    return []


def wc_rows_to_frame(rows: list[dict[str, Any]], *, is_synthetic: bool = False) -> pd.DataFrame:
    """Convert bootstrap/API match dicts to a training frame."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = mark_synthetic_rows(df, is_synthetic=is_synthetic)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def load_real_training_frame(path: Path | None = None) -> pd.DataFrame:
    """Production training frame: real API-Football matches only."""
    rows = load_wc_completed_matches(path)
    return wc_rows_to_frame(rows, is_synthetic=False)
