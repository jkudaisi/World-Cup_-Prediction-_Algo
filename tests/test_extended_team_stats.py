"""Tests for extended team stats and external-opponent training rows."""
from __future__ import annotations

import json

import pytest

from feature_builder import build_team_features
from src.ratings.extended_team_stats import (
    ChronologicalTeamStateTracker,
    get_team_stats,
    is_wc_pool_team,
    reset_runtime_registry,
)


def test_get_team_stats_unknown_team_uses_defaults():
    reset_runtime_registry()
    stats = get_team_stats("Wales", team_id=999)
    assert stats["elo"] == 1500.0
    assert stats["xg"] == 1.15
    assert is_wc_pool_team("Wales") is False


def test_build_team_features_with_external_opponent():
    reset_runtime_registry()
    feats = build_team_features("Brazil", "Wales")
    assert feats["elo_h"] > feats["elo_a"]
    assert feats["lambda_h"] > 0


def test_chronological_tracker_updates_elo_after_match():
    reset_runtime_registry()
    tracker = ChronologicalTeamStateTracker()
    before_home = tracker.snapshot("Brazil")["elo"]
    before_away = tracker.snapshot("Wales")["elo"]
    tracker.apply_match("Brazil", "Wales", 3, 0)
    after_home = tracker.snapshot("Brazil")["elo"]
    after_away = tracker.snapshot("Wales")["elo"]
    assert after_home > before_home
    assert after_away < before_away


def test_materialize_includes_external_opponent_fixture(tmp_path, monkeypatch):
    from src.data import raw_backfill_training as mod

    wc_fixture = {
        "fixture": {"id": 999002, "date": "2024-06-01T18:00:00+00:00", "status": {"short": "FT"}},
        "teams": {
            "home": {"id": 26, "name": "Argentina"},
            "away": {"id": 8888, "name": "Wales"},
        },
        "goals": {"home": 2, "away": 0},
        "score": {"fulltime": {"home": 2, "away": 0}},
        "league": {"id": 15, "name": "Friendlies", "round": "Regular Season"},
    }
    team_path = tmp_path / "fixtures" / "by_team"
    team_path.mkdir(parents=True)
    (team_path / "team_26.json").write_text(
        json.dumps({"api_response": [wc_fixture]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "load_resume_state", lambda: {"completed_fixture_ids": [999002]})
    monkeypatch.setattr(mod, "load_wc_pool_team_ids", lambda: {26})
    monkeypatch.setattr(mod, "load_raw", lambda kind, fid: None)
    monkeypatch.setattr(mod, "save_wc_matches", lambda rows: None)
    monkeypatch.setattr(mod, "load_wc_matches", lambda: [])

    rows = mod.build_training_rows_from_raw_backfill(
        raw_root=tmp_path,
        allow_external_opponents=True,
        chronological_features=False,
    )
    assert len(rows) == 1
    assert rows[0]["home_team"] == "Argentina"
    assert rows[0]["away_team"] == "Wales"
    assert rows[0]["home_in_wc_pool"] is True
    assert rows[0]["away_in_wc_pool"] is False
