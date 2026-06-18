"""Backtest models on held-out synthetic validation data and recommend ensemble weights."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from calibration import brier_score, log_loss_multiclass, normalize_outcome_probs
from ensemble import DEFAULT_WEIGHTS, outcome_probs_from_lambdas, save_model_weights
from wc2026_ml_pipeline import SCALED_MODELS, generate_synthetic_dataset, get_feature_cols, train_models_from_frame

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
BACKTEST_RESULTS_PATH = ROOT / "backtest_results.json"


def _actual_outcome(gh: int, ga: int) -> str:
    if gh > ga:
        return "home_win"
    if gh == ga:
        return "draw"
    return "away_win"


def _goal_mae(preds_h: list[float], preds_a: list[float], actual_h: list[int], actual_a: list[int]) -> float:
    err = [abs(ph - ah) + abs(pa - aa) for ph, pa, ah, aa in zip(preds_h, preds_a, actual_h, actual_a)]
    return sum(err) / max(len(err), 1) / 2.0


def _accuracy(prob_rows: list[dict], outcomes: list[str]) -> float:
    correct = sum(
        1 for row, o in zip(prob_rows, outcomes)
        if max(row, key=row.get) == o
    )
    return correct / max(len(outcomes), 1)


def evaluate_model_on_holdout(
    df: pd.DataFrame,
    holdout_frac: float = 0.2,
    seed: int = 42,
) -> dict:
    """Train on first (1-holdout) fraction, test on rest."""
    n = len(df)
    split = int(n * (1 - holdout_frac))
    train_df = df.iloc[:split].reset_index(drop=True)
    test_df = df.iloc[split:].reset_index(drop=True)
    feature_cols = get_feature_cols()

    trained, scaler = train_models_from_frame(train_df, feature_cols, verbose=False)
    model_metrics: dict[str, dict] = {}

    for name, (mh, ma) in trained.items():
        preds_h, preds_a = [], []
        prob_rows = []
        outcomes = []
        for _, row in test_df.iterrows():
            X = np.array([row[c] for c in feature_cols]).reshape(1, -1)
            Xin = scaler.transform(X) if name in SCALED_MODELS else X
            rh = max(0.1, float(mh.predict(Xin)[0]))
            ra = max(0.1, float(ma.predict(Xin)[0]))
            preds_h.append(rh)
            preds_a.append(ra)
            probs = outcome_probs_from_lambdas(rh, ra, calibrate=True)
            prob_rows.append(probs)
            outcomes.append(_actual_outcome(int(row["goals_h"]), int(row["goals_a"])))

        model_metrics[name] = {
            "accuracy": round(_accuracy(prob_rows, outcomes), 4),
            "log_loss": round(log_loss_multiclass(prob_rows, outcomes), 4),
            "brier_score": round(
                brier_score([r["home_win"] for r in prob_rows], [1 if o == "home_win" else 0 for o in outcomes]),
                4,
            ),
            "goal_mae": round(_goal_mae(preds_h, preds_a, test_df["goals_h"].tolist(), test_df["goals_a"].tolist()), 4),
            "n_test": len(test_df),
        }

    return {"models": model_metrics, "n_train": len(train_df), "n_test": len(test_df)}


def recommend_weights(model_metrics: dict[str, dict]) -> dict[str, float]:
    """Inverse log-loss weighting with floor."""
    scores = {}
    for name, m in model_metrics.items():
        ll = max(m.get("log_loss", 1.0), 0.5)
        scores[name] = 1.0 / ll
    total = sum(scores.values()) or 1.0
    weights = {k: round(v / total, 4) for k, v in scores.items()}
    return weights


def run_backtest(seed: int = 42, n_synthetic: int = 5000, holdout_frac: float = 0.2) -> dict:
    df = generate_synthetic_dataset(n_synthetic=n_synthetic, seed=seed)
    result = evaluate_model_on_holdout(df, holdout_frac=holdout_frac, seed=seed)
    weights = recommend_weights(result["models"])
    payload = {
        "seed": seed,
        "n_synthetic": n_synthetic,
        "holdout_frac": holdout_frac,
        "results": result,
        "recommended_weights": weights,
    }
    with open(BACKTEST_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    save_model_weights(weights, metrics=result["models"])
    log.info("Backtest complete — saved %s and model_weights.json", BACKTEST_RESULTS_PATH)
    return payload


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    out = run_backtest()
    print(json.dumps(out["recommended_weights"], indent=2))
