"""Refresh predictions and verify live API-Football + Kalshi connectivity."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from future_fixture_predictions import (
    merge_future_predictions_into_doc,
    refresh_future_fixture_predictions,
)
from incremental_trainer import _build_predictions_payload
from model_store import load_artifacts, models_exist
from src.config.pipeline_config import MODELS_REAL, ROOT
from src.models.model_registry import active_model_source, get_active_models_dir
from training_store import (
    atomic_write_json,
    dataset_checksum,
    load_training_state,
    load_wc_matches,
    save_training_state,
    utc_now_iso,
)
from wc2026_ml_pipeline import get_feature_cols, predict_all_fixtures

log = logging.getLogger(__name__)

PREDICTIONS_PATH = ROOT / "predictions.json"


def refresh_production_predictions(
    predictions_path: Path | None = None,
    *,
    refresh_future: bool = True,
) -> dict[str, Any]:
    """Write group-stage predictions from active real-history models."""
    path = predictions_path or PREDICTIONS_PATH
    artifacts = load_artifacts()
    if not artifacts:
        raise RuntimeError("No trained model artifacts found — run training first")

    trained = artifacts["trained"]
    scaler = artifacts["scaler"]
    feature_cols = artifacts["feature_cols"]
    ml_data = predict_all_fixtures(trained, scaler, feature_cols)

    wc_all = load_wc_matches()
    training_meta = {
        "mode": "real_history",
        "last_trained_at": utc_now_iso(),
        "training_rows_count": len(wc_all),
        "total_world_cup_matches_used": len(wc_all),
        "model_versions": artifacts.get("model_versions", {}),
        "models_dir": artifacts.get("models_dir", str(get_active_models_dir())),
        "model_source": active_model_source(),
        "new_matches_used": 0,
        "trained_fixture_ids": sorted(
            int(r["fixture_id"]) for r in wc_all if r.get("fixture_id") is not None
        ),
        "n_features": len(feature_cols),
    }
    payload = _build_predictions_payload(ml_data, training_meta)
    atomic_write_json(path, payload)

    future_summary: dict[str, Any] | None = None
    if refresh_future:
        future_summary = refresh_future_fixture_predictions(force=True)
        merged = merge_future_predictions_into_doc(payload)
        atomic_write_json(path, merged)
        payload = merged

    state = load_training_state()
    state.update({
        "last_trained_at": training_meta["last_trained_at"],
        "last_incremental_run_status": "success",
        "training_rows_count": len(wc_all),
        "total_world_cup_matches_used": len(wc_all),
        "model_versions": training_meta["model_versions"],
        "dataset_checksum": dataset_checksum(wc_all),
        "errors": [],
    })
    save_training_state(state)

    return {
        "status": "success",
        "predictions_path": str(path),
        "group_stage_matches": len(ml_data),
        "training_rows": len(wc_all),
        "models_dir": training_meta["models_dir"],
        "model_source": training_meta["model_source"],
        "future_refresh": future_summary,
    }


def verify_production_connectivity() -> dict[str, Any]:
    """Sanity-check model, API-Football, Kalshi, and prediction wiring."""
    report: dict[str, Any] = {
        "ok": True,
        "checks": {},
    }

    def _check(name: str, passed: bool, detail: Any = None) -> None:
        report["checks"][name] = {"ok": bool(passed), "detail": detail}
        if not passed:
            report["ok"] = False

    models_dir = get_active_models_dir()
    artifacts = load_artifacts(models_dir)
    _check(
        "real_history_models",
        models_exist(MODELS_REAL) and artifacts is not None,
        {
            "active_dir": str(models_dir),
            "source": active_model_source(),
            "model_count": len((artifacts or {}).get("trained", {})),
        },
    )

    try:
        from config import APIFOOTBALL_KEY
        _check("api_football_key", bool(APIFOOTBALL_KEY), "configured" if APIFOOTBALL_KEY else "missing")
    except Exception as exc:
        _check("api_football_key", False, str(exc))

    try:
        from apifootball_client import calls_remaining
        _check("api_football_budget", calls_remaining() > 0, calls_remaining())
    except Exception as exc:
        _check("api_football_budget", False, str(exc))

    try:
        from kalshi_auth import auth_status
        ks = auth_status()
        _check("kalshi_auth", bool(ks.get("configured")), ks)
    except Exception as exc:
        _check("kalshi_auth", False, str(exc))

    pred_ok = PREDICTIONS_PATH.exists()
    pred_detail: dict[str, Any] = {}
    if pred_ok:
        try:
            doc = json.loads(PREDICTIONS_PATH.read_text(encoding="utf-8"))
            pred_detail = {
                "group_matches": len(doc.get("ml_data") or []),
                "training_mode": (doc.get("training") or {}).get("mode"),
                "model_source": (doc.get("training") or {}).get("model_source"),
            }
        except (OSError, json.JSONDecodeError) as exc:
            pred_ok = False
            pred_detail = {"error": str(exc)}
    _check("predictions_json", pred_ok, pred_detail)

    cache_path = ROOT / "data" / "future_fixture_prediction_cache.json"
    cache_ok = cache_path.exists()
    cache_detail: dict[str, Any] = {}
    if cache_ok:
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            cache_detail = {"cached_fixtures": len(cache.get("fixtures") or {})}
        except (OSError, json.JSONDecodeError):
            cache_ok = False
    _check("future_prediction_cache", cache_ok, cache_detail)

    kalshi_map = ROOT / "data" / "kalshi_discovered_markets.json"
    _check(
        "kalshi_market_discovery",
        kalshi_map.exists(),
        {"path": str(kalshi_map), "exists": kalshi_map.exists()},
    )

    wc_rows = len(load_wc_matches())
    _check("training_dataset", wc_rows > 0, {"rows": wc_rows})

    return report
