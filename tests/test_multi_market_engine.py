"""Tests for multi-market probability engine and knockout progression."""

from __future__ import annotations

import pytest

from knockout_progression import build_knockout_progression, is_knockout_round
from monte_carlo_simulator import simulate_knockout_match
from multi_market_engine import build_multi_market_bundle, flatten_for_api
from market_trading_metrics import compute_market_trading_metrics


class TestKnockoutProgression:
    def test_qualification_sums_to_one(self):
        prog = build_knockout_progression(1.4, 1.1)
        h = prog["qualification"]["home"]
        a = prog["qualification"]["away"]
        assert h + a == pytest.approx(1.0, abs=0.001)

    def test_qualification_formula_consistency(self):
        prog = build_knockout_progression(1.2, 0.9)
        reg = prog["regulation"]
        et = prog["extra_time"]
        pens = prog["penalties"]
        expected_home = (
            reg["home_win"]
            + et["home_win_via_et"]
            + pens["home_win_via_pens"]
        )
        assert prog["qualification"]["home"] == pytest.approx(expected_home, abs=0.001)

    def test_reach_et_equals_draw_90(self):
        prog = build_knockout_progression(1.0, 1.0)
        assert prog["extra_time"]["reach_probability"] == pytest.approx(
            prog["regulation"]["draw"], abs=0.001,
        )

    def test_is_knockout_round(self):
        assert is_knockout_round("R32")
        assert is_knockout_round("Round of 32")
        assert not is_knockout_round("A")


class TestMonteCarlo:
    def test_simulation_qualification_near_analytic(self):
        lh, la = 1.3, 1.0
        analytic = build_knockout_progression(lh, la)
        sim = simulate_knockout_match(lh, la, n_simulations=50_000, seed=7)
        assert sim["qualification"]["home"] == pytest.approx(
            analytic["qualification"]["home"], abs=0.02,
        )
        assert sim["qualification"]["away"] == pytest.approx(
            analytic["qualification"]["away"], abs=0.02,
        )


class TestMultiMarketEngine:
    def test_knockout_bundle_has_qualification(self):
        ml = {
            "fixture_id": 1561329,
            "home": "South Africa",
            "away": "Canada",
            "group": "R32",
            "ens_h": 0.8,
            "ens_a": 1.4,
            "prediction": {
                "projected_home_goals": 0.8,
                "projected_away_goals": 1.4,
                "home_win": 0.15,
                "draw": 0.22,
                "away_win": 0.63,
            },
            "confidence": {"score": 0.55},
            "ensemble": {"model_agreement": 0.6},
        }
        bundle = build_multi_market_bundle(ml, run_simulation=False)
        assert bundle["knockout"] is True
        assert bundle["qualification_probability"] is not None
        assert "home_qualifies" in bundle["kalshi_markets"]
        assert bundle["match_winner"]["draw_90"] > 0

    def test_group_stage_no_knockout_progression(self):
        ml = {
            "home": "Brazil",
            "away": "Serbia",
            "group": "G",
            "prediction": {"projected_home_goals": 2.0, "projected_away_goals": 0.8},
            "confidence": {"score": 0.7},
        }
        bundle = build_multi_market_bundle(ml, run_simulation=False)
        assert bundle["knockout"] is False
        assert bundle.get("knockout_progression") is None

    def test_flatten_api_shape(self):
        ml = {
            "fixture_id": 1,
            "home": "A",
            "away": "B",
            "group": "R32",
            "prediction": {"projected_home_goals": 1.1, "projected_away_goals": 1.0},
            "confidence": {"score": 0.5},
        }
        api = flatten_for_api(build_multi_market_bundle(ml, run_simulation=False))
        assert "winner_probability" in api
        assert "BTTS_probability" in api
        assert "trading_metrics" in api


class TestTradingMetrics:
    def test_no_auto_trade_without_edge(self):
        m = compute_market_trading_metrics(
            fair_probability=0.55,
            market_probability=0.54,
            confidence=0.6,
            min_edge=0.08,
        )
        assert m["recommendation"] == "PASS"

    def test_trade_when_edge_and_confidence_met(self):
        m = compute_market_trading_metrics(
            fair_probability=0.70,
            market_probability=0.55,
            confidence=0.65,
            min_edge=0.08,
        )
        assert m["recommendation"] == "TRADE"
        assert m["kelly_fraction"] > 0
