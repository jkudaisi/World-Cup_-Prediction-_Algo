"""Tests for Part 2: model registry, raw storage, lineup/injury features."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.api_football_backfill import load_raw, persist_fixture_bundle
from src.features.feature_store import enrich_context_from_raw
from src.features.lineup_features import (
    extract_starter_ids,
    merge_lineup_context_for_match,
    starting_xi_overlap,
)
from src.models.model_registry import get_active_models_dir


class TestModelRegistry:
    def test_prefers_real_history_when_present(self, tmp_path, monkeypatch):
        import model_store as ms

        real = tmp_path / "real_history"
        legacy = tmp_path / "legacy"
        for d in (real, legacy):
            d.mkdir()
            (d / "scaler.pkl").write_bytes(b"")
            (d / "meta.json").write_text(json.dumps({"feature_cols": ["elo_h"]}))
            (d / "poisson_home.pkl").write_bytes(b"")
            (d / "poisson_away.pkl").write_bytes(b"")

        monkeypatch.setattr(ms, "MODELS_DIR", legacy)
        import src.models.model_registry as reg
        monkeypatch.setattr(reg, "MODELS_REAL", real)
        monkeypatch.delenv("WC_MODELS_DIR", raising=False)
        assert get_active_models_dir() == real


class TestRawStorage:
    def test_persist_and_load_fixture_bundle(self, tmp_path, monkeypatch):
        from src.config import pipeline_config as cfg
        monkeypatch.setattr(cfg, "DATA_RAW", tmp_path / "raw")

        lineups = {
            "home": {"formation": "4-3-3", "startXI": [{"player": {"id": 1, "pos": "G"}}]},
            "away": {"formation": "4-4-2", "startXI": [{"player": {"id": 2, "pos": "G"}}]},
        }
        written = persist_fixture_bundle(12345, fixture={"id": 12345}, lineups=lineups)
        assert written
        loaded = load_raw("lineups", 12345)
        assert loaded is not None
        assert loaded["data"]["home"]["formation"] == "4-3-3"


class TestLineupFeatures:
    def test_starting_xi_overlap(self):
        a = [{"player": {"id": i}} for i in range(1, 12)]
        b = [{"player": {"id": i}} for i in range(6, 17)]
        assert starting_xi_overlap(extract_starter_ids({"startXI": a}), extract_starter_ids({"startXI": b})) == 6.0

    def test_merge_lineup_context(self):
        cur = {
            "home": {"startXI": [{"player": {"id": 1, "pos": "G"}}], "formation": "4-3-3"},
            "away": {"startXI": [{"player": {"id": 2, "pos": "G"}}], "formation": "4-4-2"},
        }
        ref = {
            "home": {"startXI": [{"player": {"id": 1, "pos": "G"}}], "formation": "4-3-3"},
            "away": {"startXI": [{"player": {"id": 2, "pos": "G"}}], "formation": "4-4-2"},
        }
        ctx = merge_lineup_context_for_match(cur, ref)
        assert ctx["same_gk"] is True
        assert ctx["has_lineups"] is True


class TestFeatureStoreRawEnrichment:
    def test_enrich_context_from_raw(self, tmp_path, monkeypatch):
        from src.config import pipeline_config as cfg
        monkeypatch.setattr(cfg, "DATA_RAW", tmp_path / "raw")
        persist_fixture_bundle(
            99,
            lineups={"home": {"startXI": [{"player": {"id": 10, "pos": "G"}}], "formation": "3-5-2"}},
            injuries=[{"team": {"id": 1}}],
        )
        ctx = enrich_context_from_raw(99, home_team_id=1, away_team_id=2)
        assert ctx["has_lineups"] is True
        assert ctx["has_injuries"] is True
