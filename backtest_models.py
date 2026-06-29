"""Backtest models on temporal holdout and/or real WC completed matches."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from calibration import brier_score, log_loss_multiclass
from ensemble import load_model_weights, outcome_probs_from_lambdas, save_model_weights
from incremental_trainer import wc_rows_to_frame
from training_store import WC_MATCHES_PATH, load_wc_matches, utc_now_iso
from wc2026_ml_pipeline import SCALED_MODELS, generate_synthetic_dataset, get_feature_cols, train_models_from_frame

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
BACKTEST_RESULTS_PATH = ROOT / "backtest_results.json"
SPLIT_DATE = "2022-11-20"
MIN_TEST_ROWS = 10


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


def _fixture_ids(rows: list[dict[str, Any]]) -> list[int]:
    return [int(m["fixture_id"]) for m in rows if m.get("fixture_id") is not None]


def _ordered_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(matches, key=lambda m: (m.get("date", ""), int(m.get("fixture_id", 0))))


def split_matches_temporal(
    matches: list[dict[str, Any]],
    *,
    split_date: str = SPLIT_DATE,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Split matches by date; fall back to 80/20 random if test set too small."""
    ordered = _ordered_matches(matches)
    train_rows = [m for m in ordered if (m.get("date") or "") < split_date]
    test_rows = [m for m in ordered if (m.get("date") or "") >= split_date]

    if len(test_rows) < MIN_TEST_ROWS or len(train_rows) < 2:
        log.warning(
            "Temporal split produced train=%s test=%s (need test>=%s) — "
            "falling back to 80/20 random split on chronological order",
            len(train_rows),
            len(test_rows),
            MIN_TEST_ROWS,
        )
        n = len(ordered)
        if n < 2:
            return ordered[:1], ordered[1:], "random_80_20"
        split_idx = max(1, int(n * 0.8))
        if split_idx >= n:
            split_idx = n - 1
        train_rows = ordered[:split_idx]
        test_rows = ordered[split_idx:]
        return train_rows, test_rows, "random_80_20"

    log.info("Temporal split at %s: train=%s test=%s", split_date, len(train_rows), len(test_rows))
    return train_rows, test_rows, "temporal"


def evaluate_trained_on_frame(
    trained: dict,
    scaler,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, dict]:
    """Score already-fitted models on a labeled test frame."""
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
            prob_rows.append(outcome_probs_from_lambdas(rh, ra, calibrate=True))
            outcomes.append(_actual_outcome(int(row["goals_h"]), int(row["goals_a"])))

        model_metrics[name] = {
            "accuracy": round(_accuracy(prob_rows, outcomes), 4),
            "log_loss": round(log_loss_multiclass(prob_rows, outcomes), 4),
            "brier_score": round(
                brier_score(
                    [r["home_win"] for r in prob_rows],
                    [1 if o == "home_win" else 0 for o in outcomes],
                ),
                4,
            ),
            "goal_mae": round(
                _goal_mae(preds_h, preds_a, test_df["goals_h"].tolist(), test_df["goals_a"].tolist()),
                4,
            ),
            "n_test": len(test_df),
        }

    return model_metrics


def weights_from_brier(model_metrics: dict[str, dict]) -> dict[str, float]:
    """Derive ensemble weights from Brier scores (lower Brier → higher weight)."""
    raw = {
        name: max(0.01, 1.0 - float(m.get("brier_score", 1.0)))
        for name, m in model_metrics.items()
    }
    total = sum(raw.values()) or 1.0
    return {k: round(v / total, 4) for k, v in raw.items()}


def _print_metrics_table(model_metrics: dict[str, dict]) -> None:
    headers = ("Model", "Brier", "LogLoss", "GoalMAE", "Accuracy")
    rows = []
    for name, m in sorted(model_metrics.items()):
        rows.append((
            name,
            f"{m.get('brier_score', 0):.4f}",
            f"{m.get('log_loss', 0):.4f}",
            f"{m.get('goal_mae', 0):.4f}",
            f"{m.get('accuracy', 0):.4f}",
        ))
    widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for row in rows:
        print(fmt.format(*row))


def _print_weight_comparison(old_w: dict[str, float], new_w: dict[str, float]) -> None:
    names = sorted(set(old_w) | set(new_w))
    print(f"\n{'Model':<22} {'Old':>8} {'New':>8}")
    print("-" * 40)
    for name in names:
        print(f"{name:<22} {old_w.get(name, 0):>8.4f} {new_w.get(name, 0):>8.4f}")


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
    model_metrics = evaluate_trained_on_frame(trained, scaler, test_df, feature_cols)

    return {"models": model_metrics, "n_train": len(train_df), "n_test": len(test_df)}


def evaluate_on_wc_matches(
    train_df: pd.DataFrame,
    test_rows: list[dict[str, Any]],
    *,
    test_fixture_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Train on train_df, evaluate on completed WC test rows."""
    feature_cols = get_feature_cols()
    test_df = wc_rows_to_frame(test_rows)
    if test_df.empty:
        raise ValueError("No WC test rows available for evaluation")

    trained, scaler = train_models_from_frame(train_df, feature_cols, verbose=False)
    model_metrics = evaluate_trained_on_frame(trained, scaler, test_df, feature_cols)
    ids = test_fixture_ids if test_fixture_ids is not None else _fixture_ids(test_rows)

    return {
        "models": model_metrics,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "fixture_ids": ids,
    }


def recommend_weights(model_metrics: dict[str, dict]) -> dict[str, float]:
    """Backward-compatible alias — Brier-based weights."""
    return weights_from_brier(model_metrics)


def _save_payload(payload: dict[str, Any]) -> dict[str, Any]:
    with open(BACKTEST_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    new_weights = payload.get("new_weights") or payload.get("recommended_weights") or {}
    save_model_weights(new_weights, metrics=payload.get("per_model") or payload.get("results", {}).get("models"))
    log.info("Backtest complete — saved %s and model_weights.json", BACKTEST_RESULTS_PATH)
    return payload


def run_wc_only_backtest(
    *,
    seed: int = 42,
    holdout_frac: float = 0.2,
) -> dict[str, Any]:
    """Train and evaluate only on real WC completed matches (temporal split)."""
    matches = load_wc_matches()
    if len(matches) < 2:
        raise ValueError(
            f"Need at least 2 WC matches in {WC_MATCHES_PATH.name}; found {len(matches)}"
        )

    train_rows, test_rows, split_mode = split_matches_temporal(matches)
    train_df = wc_rows_to_frame(train_rows)
    wc_result = evaluate_on_wc_matches(
        train_df,
        test_rows,
        test_fixture_ids=_fixture_ids(test_rows),
    )
    old_weights = load_model_weights()
    new_weights = weights_from_brier(wc_result["models"])
    _print_metrics_table(wc_result["models"])
    _print_weight_comparison(old_weights, new_weights)

    payload = {
        "generated_at": utc_now_iso(),
        "train_rows": wc_result["n_train"],
        "test_rows": wc_result["n_test"],
        "split_date": SPLIT_DATE,
        "split_mode": split_mode,
        "per_model": wc_result["models"],
        "old_weights": old_weights,
        "new_weights": new_weights,
        "mode": "wc_only",
        "seed": seed,
        "holdout_frac": holdout_frac,
        "wc_fixture_ids": _fixture_ids(matches),
        "wc_train_fixture_ids": _fixture_ids(train_rows),
        "wc_eval_fixture_ids": wc_result["fixture_ids"],
        "results": wc_result,
        "recommended_weights": new_weights,
    }
    return _save_payload(payload)


def run_backtest(
    seed: int = 42,
    n_synthetic: int = 5000,
    holdout_frac: float = 0.2,
    *,
    wc_only: bool = False,
) -> dict[str, Any]:
    """Backtest ensemble weights; trains on synthetic + WC train split, evaluates on WC test split."""
    if wc_only:
        return run_wc_only_backtest(seed=seed, holdout_frac=holdout_frac)

    matches = load_wc_matches()
    train_df = generate_synthetic_dataset(n_synthetic=n_synthetic, seed=seed)
    synthetic_holdout = evaluate_model_on_holdout(train_df, holdout_frac=holdout_frac, seed=seed)

    if matches:
        train_rows, test_rows, split_mode = split_matches_temporal(matches)
        wc_train_df = wc_rows_to_frame(train_rows)
        if not wc_train_df.empty:
            combined_train = pd.concat([train_df, wc_train_df], ignore_index=True)
        else:
            combined_train = train_df
        wc_result = evaluate_on_wc_matches(
            combined_train,
            test_rows,
            test_fixture_ids=_fixture_ids(test_rows),
        )
        primary = wc_result
        mode = "synthetic_train_wc_eval"
        old_weights = load_model_weights()
        new_weights = weights_from_brier(wc_result["models"])
        _print_metrics_table(wc_result["models"])
        _print_weight_comparison(old_weights, new_weights)
        log.info(
            "WC evaluation on %s test fixtures (train=%s synthetic+%s wc, split=%s)",
            wc_result["n_test"],
            len(train_df),
            len(wc_train_df),
            split_mode,
        )
    else:
        primary = synthetic_holdout
        mode = "synthetic_holdout"
        old_weights = load_model_weights()
        new_weights = weights_from_brier(synthetic_holdout["models"])
        split_mode = "synthetic_holdout"
        _print_metrics_table(synthetic_holdout["models"])
        _print_weight_comparison(old_weights, new_weights)
        log.warning("No WC matches in %s — using synthetic holdout only", WC_MATCHES_PATH.name)

    payload = {
        "generated_at": utc_now_iso(),
        "train_rows": primary["n_train"],
        "test_rows": primary["n_test"],
        "split_date": SPLIT_DATE,
        "split_mode": split_mode,
        "per_model": primary["models"],
        "old_weights": old_weights,
        "new_weights": new_weights,
        "mode": mode,
        "seed": seed,
        "n_synthetic": n_synthetic,
        "holdout_frac": holdout_frac,
        "wc_fixture_ids": _fixture_ids(matches),
        "wc_eval_fixture_ids": primary.get("fixture_ids", []),
        "results": primary,
        "synthetic_holdout": synthetic_holdout,
        "recommended_weights": new_weights,
    }
    return _save_payload(payload)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest ML models and update ensemble weights")
    parser.add_argument(
        "--wc-only",
        action="store_true",
        help="Train and evaluate only on completed WC matches (temporal holdout)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--n-synthetic", type=int, default=5000)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    out = run_backtest(
        seed=args.seed,
        n_synthetic=args.n_synthetic,
        holdout_frac=args.holdout_frac,
        wc_only=args.wc_only,
    )
    print(json.dumps(out.get("new_weights") or out.get("recommended_weights"), indent=2))
