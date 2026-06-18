"""Save/load trained ML model artifacts."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"

MODEL_NAMES = [
    "poisson",
    "ridge",
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "lightgbm",
    "mlp",
]

INTERNAL_TO_FILE = {
    "Poisson Regression": "poisson",
    "Ridge Regression": "ridge",
    "Random Forest": "random_forest",
    "Gradient Boosting": "gradient_boosting",
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm",
    "Neural Network": "mlp",
}


def models_exist() -> bool:
    """True when scaler, meta.json (feature_cols), and at least one model pair exist."""
    meta_path = MODELS_DIR / "meta.json"
    if not meta_path.exists() or not (MODELS_DIR / "scaler.pkl").exists():
        return False
    import json

    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if not meta.get("feature_cols"):
            return False
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False
    return _model_path("poisson", "home").exists() and _model_path("poisson", "away").exists()


def _model_path(slug: str, side: str) -> Path:
    return MODELS_DIR / f"{slug}_{side}.pkl"


def save_artifacts(
    trained: dict[str, tuple[Any, Any]],
    scaler: Any,
    feature_cols: list[str],
    model_versions: dict[str, str] | None = None,
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    versions = model_versions or {}

    for internal_name, (mh, ma) in trained.items():
        slug = INTERNAL_TO_FILE[internal_name]
        joblib.dump(mh, _model_path(slug, "home"))
        joblib.dump(ma, _model_path(slug, "away"))
        versions[internal_name] = versions.get(internal_name, "1.0.0")

    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    import json
    meta = {"feature_cols": feature_cols, "model_versions": versions}
    fd, tmp = tempfile.mkstemp(dir=MODELS_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        os.replace(tmp, MODELS_DIR / "meta.json")
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    log.info("Saved model artifacts to %s", MODELS_DIR)


def load_artifacts() -> dict[str, Any] | None:
    if not models_exist():
        return None
    import json
    with open(MODELS_DIR / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    trained: dict[str, tuple[Any, Any]] = {}
    for internal_name, slug in INTERNAL_TO_FILE.items():
        hp = _model_path(slug, "home")
        ap = _model_path(slug, "away")
        if hp.exists() and ap.exists():
            trained[internal_name] = (joblib.load(hp), joblib.load(ap))
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    return {
        "trained": trained,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "model_versions": meta.get("model_versions", {}),
    }
