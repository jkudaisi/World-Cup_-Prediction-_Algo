"""Tests for knockout ET/penalty/qualification ML models."""

from __future__ import annotations

import pytest

from knockout_models import (
    build_knockout_dataset,
    load_knockout_model_meta,
    predict_knockout_adjustments,
    train_knockout_models,
)
from knockout_outcomes import load_seed_knockout_outcomes, parse_knockout_outcome


class TestKnockoutOutcomes:
    def test_parse_knockout_pen_fixture(self):
        fixture = {
            "fixture": {"id": 1, "status": {"short": "PEN"}},
            "league": {"round": "Round of 16"},
            "teams": {"home": {"name": "Spain"}, "away": {"name": "Russia"}},
            "goals": {"home": 1, "away": 1},
            "score": {
                "fulltime": {"home": 1, "away": 1},
                "extratime": {"home": 1, "away": 1},
                "penalty": {"home": 3, "away": 4},
            },
        }
        out = parse_knockout_outcome(fixture)
        assert out is not None
        assert out["draw_at_90"] is True
        assert out["went_to_pens"] is True
        assert out["home_won_pens"] is False
        assert out["home_qualifies"] is False

    def test_parse_group_stage_returns_none(self):
        fixture = {
            "fixture": {"id": 2, "status": {"short": "FT"}},
            "league": {"round": "Group A - 1"},
            "teams": {"home": {"name": "Brazil"}, "away": {"name": "Serbia"}},
            "goals": {"home": 2, "away": 0},
            "score": {"fulltime": {"home": 2, "away": 0}},
        }
        assert parse_knockout_outcome(fixture) is None

    def test_seed_has_enough_rows(self):
        rows = load_seed_knockout_outcomes()
        assert len(rows) >= 20
        pen_rows = [r for r in rows if r.get("went_to_pens")]
        assert len(pen_rows) >= 5


class TestKnockoutModels:
    def test_train_and_predict(self, tmp_path, monkeypatch):
        monkeypatch.setattr("knockout_models.KNOCKOUT_MODELS_DIR", tmp_path)
        monkeypatch.setattr("knockout_models.META_PATH", tmp_path / "meta.json")

        meta = train_knockout_models(use_api=False, force=True)
        assert meta.get("status") == "trained"
        assert meta.get("training_samples", 0) >= 20
        assert "qualification" in meta.get("models", {})

        loaded = load_knockout_model_meta()
        assert loaded.get("status") == "trained"

        adj = predict_knockout_adjustments("France", "Argentina", 1.5, 1.2)
        assert adj["available"] is True
        assert 0.0 < adj["home_pen_skill"] < 1.0
        et = adj["et_conditional"]
        assert et["home_win"] + et["draw"] + et["away_win"] == pytest.approx(1.0, abs=0.01)

    def test_build_dataset_merges_seed(self, monkeypatch):
        monkeypatch.setattr(
            "knockout_outcomes.fetch_knockout_outcomes_from_api",
            lambda **k: [],
        )
        rows = build_knockout_dataset(use_api=True)
        assert len(rows) >= 20
