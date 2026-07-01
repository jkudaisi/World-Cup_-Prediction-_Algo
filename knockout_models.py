"""Train and apply ML models for knockout ET, penalties, and qualification."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from calibration import brier_score, select_best_calibrator, apply_calibrator
from knockout_outcomes import (
    fetch_knockout_outcomes_from_api,
    load_seed_knockout_outcomes,
    outcome_row_to_features,
)
from knockout_progression import (
    DEFAULT_HOME_PEN_SKILL,
    extra_time_outcomes_given_draw,
    regulation_outcomes,
)
from training_store import atomic_write_json

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
KNOCKOUT_MODELS_DIR = ROOT / "models" / "knockout"
META_PATH = KNOCKOUT_MODELS_DIR / "meta.json"

KNOCKOUT_FEATURE_COLS = [
    "elo_diff",
    "elo_prob_h",
    "xg_diff",
    "form_diff",
    "lambda_h",
    "lambda_a",
    "knockout_stage",
    "poisson_home_90",
    "poisson_draw_90",
    "poisson_away_90",
]


def _models_exist() -> bool:
    return META_PATH.exists() and (KNOCKOUT_MODELS_DIR / "pen_model.pkl").exists()


def _blend_weight(n_samples: int) -> float:
    """More training data → trust ML more; cap at 0.75."""
    return min(0.75, n_samples / (n_samples + 25.0))


def build_knockout_feature_matrix(
    outcomes: list[dict[str, Any]],
    *,
    lambda_h: float | None = None,
    lambda_a: float | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Feature matrix for knockout specialist models."""
    rows: list[list[float]] = []
    valid_cols = KNOCKOUT_FEATURE_COLS[:]
    for oc in outcomes:
        feats = outcome_row_to_features(oc)
        lh = lambda_h if lambda_h is not None else feats["lambda_h"]
        la = lambda_a if lambda_a is not None else feats["lambda_a"]
        reg = regulation_outcomes(lh, la)
        row = [
            feats.get(c, 0.0) if c in feats else 0.0
            for c in KNOCKOUT_FEATURE_COLS
            if c not in ("poisson_home_90", "poisson_draw_90", "poisson_away_90")
        ]
        row.extend([reg["home_win"], reg["draw"], reg["away_win"]])
        rows.append(row)
    return np.array(rows, dtype=float), valid_cols


def build_prediction_features(
    home: str,
    away: str,
    lambda_h: float,
    lambda_a: float,
) -> np.ndarray:
    feats = outcome_row_to_features({"home": home, "away": away})
    reg = regulation_outcomes(lambda_h, lambda_a)
    row = [
        feats.get(c, 0.0)
        for c in KNOCKOUT_FEATURE_COLS
        if c not in ("poisson_home_90", "poisson_draw_90", "poisson_away_90")
    ]
    row.extend([reg["home_win"], reg["draw"], reg["away_win"]])
    return np.array(row, dtype=float).reshape(1, -1)


def _train_logistic(X: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    if len(set(y.tolist())) < 2:
        return None
    model = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")
    model.fit(X, y)
    return model


def build_knockout_dataset(*, use_api: bool = True) -> list[dict[str, Any]]:
    rows = load_seed_knockout_outcomes()
    if use_api:
        try:
            api_rows = fetch_knockout_outcomes_from_api()
            seen = {(r["home"], r["away"], r.get("reg_goals_h"), r.get("reg_goals_a")) for r in rows}
            for r in api_rows:
                key = (r["home"], r["away"], r.get("reg_goals_h"), r.get("reg_goals_a"))
                if key not in seen:
                    rows.append(r)
                    seen.add(key)
        except Exception as exc:
            log.debug("API knockout fetch skipped: %s", exc)
    return rows


def train_knockout_models(*, use_api: bool = True, force: bool = False) -> dict[str, Any]:
    """Train ET, penalty, and qualification models; persist to models/knockout/."""
    if _models_exist() and not force:
        return load_knockout_model_meta()

    outcomes = build_knockout_dataset(use_api=use_api)
    if len(outcomes) < 8:
        log.warning("Insufficient knockout outcomes (%s) — keeping Poisson defaults", len(outcomes))
        return {"status": "skipped", "reason": "insufficient_data", "count": len(outcomes)}

    X, feature_cols = build_knockout_feature_matrix(outcomes)
    KNOCKOUT_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    meta: dict[str, Any] = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_samples": len(outcomes),
        "feature_cols": feature_cols,
        "blend_weight": _blend_weight(len(outcomes)),
        "models": {},
    }

    # --- Penalty winner (conditional on pens) ---
    pen_rows = [o for o in outcomes if o.get("went_to_pens")]
    if len(pen_rows) >= 5:
        Xp, _ = build_knockout_feature_matrix(pen_rows)
        yp = np.array([1 if o.get("home_won_pens") or o.get("home_qualifies") else 0 for o in pen_rows])
        pen_model = _train_logistic(Xp, yp)
        if pen_model:
            joblib.dump(pen_model, KNOCKOUT_MODELS_DIR / "pen_model.pkl")
            preds = pen_model.predict_proba(Xp)[:, 1]
            cal = select_best_calibrator(yp.tolist(), preds.tolist())
            meta["models"]["penalty_winner"] = {
                "samples": len(pen_rows),
                "brier_score": cal.get("brier_score"),
                "calibration": cal.get("method"),
            }
            if cal.get("calibrator") is not None:
                joblib.dump(
                    {"method": cal["method"], "calibrator": cal["calibrator"]},
                    KNOCKOUT_MODELS_DIR / "pen_calibrator.pkl",
                )

    # --- ET winner given draw at 90 ---
    et_rows = [o for o in outcomes if o.get("draw_at_90") and o.get("went_to_et")]
    if len(et_rows) >= 5:
        Xe, _ = build_knockout_feature_matrix(et_rows)
        ye_simple = np.array([
            1 if o.get("home_won_et") else (0 if o.get("away_won_et") else 2)
            for o in et_rows
        ])
        et_model = _train_logistic(Xe, ye_simple)
        if et_model:
            joblib.dump(et_model, KNOCKOUT_MODELS_DIR / "et_model.pkl")
            meta["models"]["extra_time_winner"] = {"samples": len(et_rows)}

    # --- Direct qualification ---
    yq = np.array([1 if o.get("home_qualifies") else 0 for o in outcomes])
    qual_model = _train_logistic(X, yq)
    if qual_model:
        joblib.dump(qual_model, KNOCKOUT_MODELS_DIR / "qual_model.pkl")
        preds = qual_model.predict_proba(X)[:, 1]
        cal = select_best_calibrator(yq.tolist(), preds.tolist())
        meta["models"]["qualification"] = {
            "samples": len(outcomes),
            "brier_score": cal.get("brier_score"),
            "calibration": cal.get("method"),
        }
        if cal.get("calibrator") is not None:
            joblib.dump(
                {"method": cal["method"], "calibrator": cal["calibrator"]},
                KNOCKOUT_MODELS_DIR / "qual_calibrator.pkl",
            )

    meta["status"] = "trained"
    atomic_write_json(META_PATH, meta)
    log.info(
        "Knockout models trained on %s outcomes (blend=%.2f)",
        len(outcomes), meta["blend_weight"],
    )
    return meta


def load_knockout_model_meta() -> dict[str, Any]:
    if not META_PATH.exists():
        return {"status": "untrained"}
    with open(META_PATH, encoding="utf-8") as f:
        return json.load(f)


def _load_model(name: str):
    path = KNOCKOUT_MODELS_DIR / name
    if not path.exists():
        return None
    return joblib.load(path)


def _load_calibrator(name: str) -> dict[str, Any]:
    path = KNOCKOUT_MODELS_DIR / name
    if not path.exists():
        return {"method": "shrink_uniform", "calibrator": None}
    loaded = joblib.load(path)
    if isinstance(loaded, dict):
        return loaded
    return {"method": "isotonic", "calibrator": loaded}


def predict_knockout_adjustments(
    home: str,
    away: str,
    lambda_h: float,
    lambda_a: float,
    fixture_id: int | None = None,
) -> dict[str, Any]:
    """
    ML adjustments for ET conditional outcomes and penalty skill.

    Returns blend weights and adjusted probabilities to merge with Poisson cascade.
    """
    meta = load_knockout_model_meta()
    blend = float(meta.get("blend_weight") or 0.0) if meta.get("status") == "trained" else 0.0
    X = build_prediction_features(home, away, lambda_h, lambda_a)
    poisson_et = extra_time_outcomes_given_draw(lambda_h, lambda_a)
    out: dict[str, Any] = {
        "available": meta.get("status") == "trained",
        "blend_weight": blend,
        "meta": meta if meta.get("status") == "trained" else {},
        "poisson_et": poisson_et,
    }

    base_pen = DEFAULT_HOME_PEN_SKILL
    if meta.get("status") == "trained":
        pen_model = _load_model("pen_model.pkl")
        if pen_model is not None:
            raw = float(pen_model.predict_proba(X)[0, 1])
            cal_info = _load_calibrator("pen_calibrator.pkl")
            home_pen = apply_calibrator(raw, cal_info) if cal_info.get("calibrator") else raw
            base_pen = round(blend * home_pen + (1 - blend) * DEFAULT_HOME_PEN_SKILL, 4)

    try:
        from src.features.goalkeeper_penalties import blend_pen_skill_with_goalkeepers
        blended, gk_meta = blend_pen_skill_with_goalkeepers(
            home, away, base_pen, fixture_id=fixture_id,
        )
        out["home_pen_skill"] = blended
        out["goalkeeper_penalties"] = gk_meta
    except Exception as exc:
        log.debug("GK pen blend skipped: %s", exc)
        out["home_pen_skill"] = base_pen

    if meta.get("status") != "trained":
        return out

    et_model = _load_model("et_model.pkl")
    if et_model is not None:
        probs = et_model.predict_proba(X)[0]
        classes = list(et_model.classes_)
        p_h = p_d = p_a = 0.0
        for cls, p in zip(classes, probs):
            if cls == 1:
                p_h = float(p)
            elif cls == 0:
                p_a = float(p)
            else:
                p_d = float(p)
        total = p_h + p_d + p_a or 1.0
        ml_et = {"home_win": p_h / total, "draw": p_d / total, "away_win": p_a / total}
        out["et_conditional"] = {
            k: round(blend * ml_et[k] + (1 - blend) * poisson_et[k], 4)
            for k in ("home_win", "draw", "away_win")
        }
    else:
        out["et_conditional"] = poisson_et

    qual_model = _load_model("qual_model.pkl")
    if qual_model is not None:
        raw_q = float(qual_model.predict_proba(X)[0, 1])
        cal_info = _load_calibrator("qual_calibrator.pkl")
        out["home_qualifies_ml"] = apply_calibrator(raw_q, cal_info) if cal_info.get("calibrator") else raw_q

    return out


def train_knockout_models_on_startup() -> dict[str, Any]:
    try:
        if _models_exist():
            meta = load_knockout_model_meta()
            log.info("Knockout models loaded (%s samples)", meta.get("training_samples", 0))
            return meta
        return train_knockout_models(use_api=True, force=True)
    except Exception as exc:
        log.exception("Knockout model training failed: %s", exc)
        return {"status": "error", "error": str(exc)}
