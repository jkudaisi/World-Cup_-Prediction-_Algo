"""Tests for confidence scoring and per-market confidence."""

from __future__ import annotations

import pytest

from calibration import compute_confidence
from live_predictor import update_live_prediction_from_snapshot


class TestComputeConfidence:
    def test_early_live_with_stats_not_too_low(self):
        conf = compute_confidence(
            model_agreement=0.6,
            data_quality=0.45,
            lineup_completeness=0.3,
            live_stats_completeness=0.9,
            minute=7,
        )
        assert conf["score"] >= 0.47

    def test_prematch_baseline(self):
        conf = compute_confidence(
            model_agreement=0.6,
            data_quality=0.65,
            lineup_completeness=0.3,
            minute=0,
        )
        assert conf["score"] >= 0.35


class TestLivePredictorConfidence:
    def test_live_snapshot_boosts_confidence_with_stats(self):
        base = {
            "prediction": {
                "projected_home_goals": 2.1,
                "projected_away_goals": 0.9,
                "home_win": 0.82,
                "draw": 0.11,
                "away_win": 0.07,
            },
            "ensemble": {"model_agreement": 0.65},
            "confidence": {"score": 0.58},
        }
        snapshot = {
            "fixture_id": 1562344,
            "minute": 7,
            "status": "1H",
            "score": {"home": 0, "away": 0},
            "stats": {
                "home": {"ball_possession": "78%", "shots_total": 1, "corner_kicks": 1},
                "away": {"ball_possession": "22%", "shots_total": 0, "corner_kicks": 0},
            },
            "events": [],
        }
        out = update_live_prediction_from_snapshot(snapshot, base)
        assert out["confidence"]["score"] >= 0.48


class TestMarketConfidence:
    def test_advance_market_uses_knockout_blend(self):
        from trading_service import _confidence_for_market

        match = {
            "confidence": {"score": 0.58},
            "ensemble": {"model_agreement": 0.65},
        }
        live_row = {"confidence": {"score": 0.43}, "status": "1H"}
        conf, source = _confidence_for_market(
            match,
            live_row,
            "home_advance",
            qualification_probs={"home_qualifies": 0.85, "away_qualifies": 0.15},
        )
        assert source == "knockout_qualification"
        assert conf >= 0.52
        assert conf > live_row["confidence"]["score"]
