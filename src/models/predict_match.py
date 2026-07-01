"""Unified match prediction using feature store + active model registry."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from ensemble import build_prediction_envelope, weighted_ensemble_goals
from feature_builder import build_features
from model_store import load_artifacts
from src.features.feature_store import build_features_for_match
from src.models.model_registry import active_model_source
from wc2026_ml_pipeline import (
    FLAGS,
    MODEL_HTML_NAMES,
    SCALED_MODELS,
    TEAM_STATS,
    get_feature_cols,
)


def predict_match(
    home: str,
    away: str,
    *,
    home_team_id: int | None = None,
    away_team_id: int | None = None,
    match_date: str | datetime | None = None,
    group: str = "KO",
    match_number: int | None = None,
    knockout: bool = True,
    competition_context: dict[str, Any] | None = None,
    trained=None,
    scaler=None,
    feature_cols: list[str] | None = None,
) -> dict[str, Any]:
    """
    Predict one match using the same feature path as historical training.
    Resolves team names from IDs when provided.
    """
    if home not in TEAM_STATS or away not in TEAM_STATS:
        raise ValueError(f"Unknown teams for prediction: {home} vs {away}")

    ctx = dict(competition_context or {})
    ctx.setdefault("knockout_stage", 1.0 if knockout else 0.0)
    ref_date = match_date or datetime.utcnow().isoformat()

    data_quality = 0.65
    lineup_completeness = 0.5
    if home_team_id is not None and away_team_id is not None:
        fs = build_features_for_match(
            home_team_id, away_team_id, ref_date, ctx,
        )
        data_quality = fs.data_quality_score
        lineup_completeness = 1.0 - (
            0.5 if fs.missing_indicators.get("missing_lineups_flag") else 0.0
        )
        feats = fs.feature_values
    else:
        feats = build_features(home, away, context=ctx)

    if trained is None or scaler is None or feature_cols is None:
        artifacts = load_artifacts()
        if not artifacts:
            raise RuntimeError("No trained model artifacts found — run training first")
        trained = artifacts["trained"]
        scaler = artifacts["scaler"]
        feature_cols = artifacts["feature_cols"]

    feat_vec = np.array([feats[c] for c in feature_cols]).reshape(1, -1)
    feat_vec_sc = scaler.transform(feat_vec)

    model_preds = {}
    for name, (mh, ma) in trained.items():
        Xin = feat_vec_sc if name in SCALED_MODELS else feat_vec
        raw_h = float(mh.predict(Xin)[0])
        raw_a = float(ma.predict(Xin)[0])
        gh = max(0, round(raw_h))
        ga = max(0, round(raw_a))
        model_preds[name] = (gh, ga, raw_h, raw_a)

    rh, ra, model_agreement = weighted_ensemble_goals(model_preds)
    ens_h = max(0, round(rh))
    ens_a = max(0, round(ra))

    html_models = {
        MODEL_HTML_NAMES[name]: {"gh": gh, "ga": ga, "rh": rh, "ra": ra}
        for name, (gh, ga, rh, ra) in model_preds.items()
    }

    envelope = build_prediction_envelope(
        home, away, model_preds,
        data_quality=data_quality,
        lineup_completeness=lineup_completeness,
    )
    envelope["ensemble"]["model_agreement"] = round(model_agreement, 3)

    return {
        "mn": match_number,
        "group": group,
        "home": home,
        "away": away,
        "home_flag": FLAGS.get(home, "🏳️"),
        "away_flag": FLAGS.get(away, "🏳️"),
        "models": html_models,
        "ens_h": ens_h,
        "ens_a": ens_a,
        "ens": f"{ens_h}-{ens_a}",
        "prediction": envelope["prediction"],
        "confidence": envelope["confidence"],
        "explanation": envelope["explanation"],
        "ensemble": envelope["ensemble"],
        "model_source": active_model_source(),
        "data_quality_score": data_quality,
    }
