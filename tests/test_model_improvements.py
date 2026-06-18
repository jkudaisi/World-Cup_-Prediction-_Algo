"""Tests for feature_builder, calibration, ensemble, and backtest."""

import json

import numpy as np
import pytest

import calibration as cal
import feature_builder as fb
from ensemble import load_model_weights, outcome_probs_from_lambdas, weighted_ensemble_goals


def test_xg_proxy_uses_api_xg():
    stats = {"expected_goals": "1.45", "shots_on_goal": 2}
    assert fb.calc_xg_proxy(stats) == pytest.approx(1.45)


def test_xg_proxy_weighted_formula():
    stats = {
        "shots_on_goal": 4,
        "shots_off_goal": 3,
        "shots_blocked": 1,
        "corner_kicks": 2,
    }
    expected = 4 * 0.32 + 3 * 0.06 + 1 * 0.03 + 2 * 0.04
    assert fb.calc_xg_proxy(stats) == pytest.approx(expected, rel=1e-3)


def test_xg_pair_diff():
    hs = {"shots_on_goal": 5, "shots_total": 10, "corner_kicks": 3}
    aws = {"shots_on_goal": 2, "shots_total": 6, "corner_kicks": 1}
    pair = fb.calc_xg_pair(hs, aws)
    assert pair["diff"] == pytest.approx(pair["home"] - pair["away"])
    assert pair["home_per_shot"] == pytest.approx(pair["home"] / 10, rel=1e-3)


def test_live_features_from_snapshot():
    snap = {
        "minute": 55,
        "score": {"home": 1, "away": 0},
        "stats": {
            "home": {"shots_on_goal": 4, "shots_total": 8, "ball_possession": "58%", "corner_kicks": 3},
            "away": {"shots_on_goal": 1, "shots_total": 4, "ball_possession": "42%", "corner_kicks": 1},
        },
        "events": [],
    }
    feats = fb.build_live_features(snap)
    assert feats["live_score_diff"] == 1.0
    assert feats["live_xg_diff"] > 0


def test_data_quality_scoring():
    dq = fb.score_data_quality({"stats": {"home": {"shots_on_goal": 1}}, "events": []})
    assert 0 < dq["score"] < 1
    assert "stats" not in dq["missing"]


def test_sample_weights_capped():
    assert fb.sample_weight_for_row({"source": "world_cup", "wc_match_index": 99}) <= 1.5


def test_normalize_probs_sum_to_one():
    out = cal.normalize_outcome_probs({"home_win": 0.5, "draw": 0.3, "away_win": 0.1})
    assert pytest.approx(sum(out.values()), rel=1e-3) == 1.0


def test_calibration_reduces_peak():
    raw = {"home_win": 0.92, "draw": 0.05, "away_win": 0.03}
    calmed = cal.calibrate_outcome_probs(raw)
    assert calmed["home_win"] < raw["home_win"]


def test_weighted_ensemble_not_equal_average():
    preds = {
        "Poisson Regression": (2, 1, 2.0, 1.0),
        "Neural Network": (0, 0, 0.5, 0.5),
    }
    rh, ra, _ = weighted_ensemble_goals(preds)
    simple_h = 1.25
    assert rh != pytest.approx(simple_h, abs=0.01)


def test_outcome_probs_from_lambdas():
    probs = outcome_probs_from_lambdas(1.5, 1.0)
    assert pytest.approx(sum(probs.values()), rel=1e-3) == 1.0


def test_load_model_weights():
    w = load_model_weights()
    assert pytest.approx(sum(w.values()), rel=1e-2) == 1.0


def test_backtest_runs(tmp_path, monkeypatch):
    import backtest_models as bt

    monkeypatch.setattr(bt, "BACKTEST_RESULTS_PATH", tmp_path / "backtest_results.json")
    monkeypatch.setattr(bt, "ROOT", tmp_path)
    from ensemble import WEIGHTS_PATH
    monkeypatch.setattr("ensemble.WEIGHTS_PATH", tmp_path / "model_weights.json")

    result = bt.run_backtest(n_synthetic=400, holdout_frac=0.25)
    assert "recommended_weights" in result
    assert (tmp_path / "backtest_results.json").exists()
