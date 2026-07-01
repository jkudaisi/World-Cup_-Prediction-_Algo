"""Tests for raw backfill → training row materialization."""
from __future__ import annotations

import json

from src.data.raw_backfill_training import (
    merge_training_rows,
    parse_fixture_statistics,
)


def test_parse_fixture_statistics_maps_home_and_away():
    raw = [
        {
            "team": {"id": 10, "name": "Home"},
            "statistics": [
                {"type": "Total Shots", "value": 12},
                {"type": "Ball Possession", "value": "55%"},
            ],
        },
        {
            "team": {"id": 20, "name": "Away"},
            "statistics": [
                {"type": "Total Shots", "value": 8},
                {"type": "Corner Kicks", "value": 3},
            ],
        },
    ]
    stats = parse_fixture_statistics(raw, 10, 20)
    assert stats["home"]["shots_total"] == 12
    assert stats["home"]["ball_possession"] == "55%"
    assert stats["away"]["shots_total"] == 8
    assert stats["away"]["corner_kicks"] == 3


def test_merge_training_rows_backfill_overwrites_existing():
    existing = [{"fixture_id": 1, "goals_h": 1, "source": "historical_bootstrap", "extra": "keep"}]
    backfill = [{"fixture_id": 1, "goals_h": 2, "source": "api_football_raw_backfill", "league_id": 1}]
    merged = merge_training_rows(existing, backfill)
    assert len(merged) == 1
    assert merged[0]["goals_h"] == 2
    assert merged[0]["extra"] == "keep"
    assert merged[0]["prior_source"] == "historical_bootstrap"


def test_materialize_training_dataset_from_fixture_cache(tmp_path, monkeypatch):
    from src.data import raw_backfill_training as mod

    fixture = {
        "fixture": {"id": 999001, "date": "2024-06-01T18:00:00+00:00", "status": {"short": "FT"}},
        "teams": {
            "home": {"id": 10, "name": "Brazil"},
            "away": {"id": 6, "name": "Argentina"},
        },
        "goals": {"home": 2, "away": 1},
        "score": {"fulltime": {"home": 2, "away": 1}},
        "league": {"id": 15, "name": "Friendlies", "round": "Regular Season"},
    }
    team_path = tmp_path / "fixtures" / "by_team"
    team_path.mkdir(parents=True)
    (team_path / "team_10.json").write_text(
        json.dumps({"api_response": [fixture]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "load_resume_state", lambda: {"completed_fixture_ids": [999001]})
    monkeypatch.setattr(mod, "load_raw", lambda kind, fid: None)
    monkeypatch.setattr(mod, "save_wc_matches", lambda rows: None)
    monkeypatch.setattr(mod, "load_wc_matches", lambda: [])

    rows = mod.build_training_rows_from_raw_backfill(
        raw_root=tmp_path,
        use_resume_completed_only=True,
        allow_external_opponents=False,
        chronological_features=False,
    )
    assert len(rows) == 1
    assert rows[0]["fixture_id"] == 999001
    assert rows[0]["home_team"] == "Brazil"
    assert rows[0]["source"] == "api_football_raw_backfill"
