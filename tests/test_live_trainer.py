"""Tests for live_trainer snapshot ingestion."""

import json
from pathlib import Path

import live_trainer as lt


def test_ingest_updates_lambdas(tmp_path, monkeypatch):
    pred_file = tmp_path / "predictions.json"
    pred_file.write_text(json.dumps({
        "ml_data": [{
            "home": "Mexico",
            "away": "South Africa",
            "models": {
                "Poisson": {"rh": 1.8, "ra": 1.0, "gh": 2, "ga": 1},
                "Ridge": {"rh": 1.6, "ra": 1.0, "gh": 2, "ga": 1},
            },
        }],
    }), encoding="utf-8")

    monkeypatch.setattr(lt, "PREDICTIONS_PATH", pred_file)
    lt._invalidate_predictions_cache()

    data = {
        "stats": {
            "home": {
                "shots_on_goal": 4,
                "corner_kicks": 3,
                "ball_possession": "55%",
                "yellow_cards": 1,
                "red_cards": 0,
            },
            "away": {
                "shots_on_goal": 2,
                "corner_kicks": 1,
                "ball_possession": "45%",
                "yellow_cards": 0,
                "red_cards": 0,
            },
        },
        "events": [
            {"time": {"elapsed": 30}, "team": {"id": 1}, "type": "Card", "detail": "Yellow Card"},
        ],
    }

    state = lt.ingest_live_snapshot(
        99, data, home_name="Mexico", away_name="South Africa",
        home_team_id=1, away_team_id=2,
    )

    assert state.elapsed == 30
    assert state.adj_lambda_home is not None
    assert state.adj_lambda_home > 0

    saved = json.loads(pred_file.read_text(encoding="utf-8"))
    match = saved["ml_data"][0]
    assert match["live_status"] == "live"
    assert "live_adj_lambda_h" in match
    assert match["live_stats"]["home_sot"] == 4


def test_get_base_lambdas_fallback(monkeypatch, tmp_path):
    pred_file = tmp_path / "predictions.json"
    pred_file.write_text(json.dumps({"ml_data": []}), encoding="utf-8")
    monkeypatch.setattr(lt, "PREDICTIONS_PATH", pred_file)
    lt._invalidate_predictions_cache()
    base = lt._get_base_lambdas("Unknown A", "Unknown B")
    assert base == {"lambda_h": 1.2, "lambda_a": 0.9}
