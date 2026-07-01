"""Resolve which model directory is active for predictions and trading."""
from __future__ import annotations

import os
from pathlib import Path

from model_store import MODELS_DIR, models_exist

from src.config.pipeline_config import MODELS_REAL, ROOT


def get_active_models_dir() -> Path:
    """
    Prefer models/real_history/ when trained; else legacy models/.
    Override with WC_MODELS_DIR env var.
    """
    override = os.environ.get("WC_MODELS_DIR", "").strip()
    if override:
        return Path(override)
    if models_exist(MODELS_REAL):
        return MODELS_REAL
    return MODELS_DIR


def active_model_source() -> str:
    d = get_active_models_dir()
    if d == MODELS_REAL:
        return "real_history"
    if d == MODELS_DIR:
        return "legacy"
    return str(d.relative_to(ROOT)) if d.is_relative_to(ROOT) else str(d)
