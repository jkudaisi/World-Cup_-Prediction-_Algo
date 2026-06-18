"""Tests for live prediction engine."""

import json

import pytest

import feature_builder as fb
import live_predictor as lp
import live_snapshot_store as lss
import live_trainer as lt


def test_calc_xg_proxy():
    stats = {"shots_on_goal": 4, "shots_off_goal": 3, "shots_blocked": 1, "corner_kicks": 3}
    assert fb.calc_xg_proxy(stats) == pytest.approx(4 * 0.32 + 3 * 0.06 + 1 * 0.03 + 3 * 0.04, rel=1e-3)


def test_momentum_sums_reasonably():
    hs = {"shots_on_goal": 5, "shots_total": 10, "ball_possession": "60%", "corner_kicks": 4}
    aws = {"shots_on_goal": 2, "shots_total": 5, "ball_possession": "40%", "corner_kicks": 1}
    mom = fb.calc_momentum(hs, aws)
    assert mom["home"] > mom["away"]
    assert abs(mom["home"] + mom["away"] - 100.0) < 0.2


def test_update_live_prediction_changes_with_minute():
    base = {
        "models": {
            "Poisson": {"rh": 1.5, "ra": 1.0, "gh": 2, "ga": 1},
        },
    }
    snap_early = {
        "minute": 12,
        "status": "1H",
        "score": {"home": 0, "away": 0},
        "stats": {
            "home": {"shots_on_goal": 2, "shots_total": 4, "corner_kicks": 2},
            "away": {"shots_on_goal": 1, "shots_total": 2, "corner_kicks": 0},
        },
        "events": [],
    }
    snap_late = dict(snap_early)
    snap_late["minute"] = 70
    snap_late["score"] = {"home": 1, "away": 0}
    snap_late["stats"] = {
        "home": {"shots_on_goal": 6, "shots_total": 12, "corner_kicks": 5, "red_cards": 0},
        "away": {"shots_on_goal": 2, "shots_total": 6, "corner_kicks": 2, "red_cards": 1},
    }
    early = lp.update_live_prediction_from_snapshot(snap_early, base)
    late = lp.update_live_prediction_from_snapshot(snap_late, base)
    assert early["probabilities"]["home_win"] != late["probabilities"]["home_win"]
    assert "over_under" in late
    assert "next_goal" in late
    assert late["confidence"]["score"] >= early["confidence"]["score"]


def test_snapshot_append_dedupes(tmp_path, monkeypatch):
    monkeypatch.setattr(lss, "SNAPSHOT_DIR", tmp_path)
    snap = {
        "minute": 10,
        "status": "1H",
        "score": {"home": 0, "away": 0},
        "stats": {"home": {}, "away": {}},
    }
    assert lss.append_snapshot(99, snap) is True
    assert lss.append_snapshot(99, snap) is False
    assert len(lss.load_snapshots(99)) == 1


def test_live_probs_normalize():
    base = {"models": {"Poisson": {"rh": 1.4, "ra": 1.1, "gh": 1, "ga": 1}}}
    snap = {
        "minute": 70,
        "status": "2H",
        "score": {"home": 1, "away": 0},
        "stats": {
            "home": {"shots_on_goal": 3, "shots_total": 7, "corner_kicks": 4, "red_cards": 0},
            "away": {"shots_on_goal": 2, "shots_total": 5, "corner_kicks": 2, "red_cards": 1},
        },
        "events": [],
    }
    out = lp.update_live_prediction_from_snapshot(snap, base)
    p = out["prediction"]
    assert pytest.approx(p["home_win"] + p["draw"] + p["away_win"], rel=1e-2) == 1.0
    assert isinstance(out["explanation"]["top_factors"], list)
    assert isinstance(out["confidence"], dict)


def test_red_card_impact():
    base = {"models": {"Poisson": {"rh": 1.5, "ra": 1.2, "gh": 2, "ga": 1}}}
    no_card = {
        "minute": 60, "score": {"home": 1, "away": 1},
        "stats": {"home": {"shots_on_goal": 3, "shots_total": 6}, "away": {"shots_on_goal": 3, "shots_total": 6}},
        "events": [],
    }
    with_card = dict(no_card)
    with_card["stats"] = {
        "home": {"shots_on_goal": 3, "shots_total": 6, "red_cards": 1},
        "away": {"shots_on_goal": 3, "shots_total": 6, "red_cards": 0},
    }
    p0 = lp.update_live_prediction_from_snapshot(no_card, base)
    p1 = lp.update_live_prediction_from_snapshot(with_card, base)
    assert p1["prediction"]["home_win"] < p0["prediction"]["home_win"]


def test_ingest_updates_probabilities(tmp_path, monkeypatch):
    pred_file = tmp_path / "predictions.json"
    pred_file.write_text(json.dumps({
        "ml_data": [{
            "home": "Mexico",
            "away": "South Africa",
            "models": {
                "Poisson": {"rh": 1.8, "ra": 1.0, "gh": 2, "ga": 1},
            },
        }],
    }), encoding="utf-8")
    monkeypatch.setattr(lt, "PREDICTIONS_PATH", pred_file)
    monkeypatch.setattr(lss, "SNAPSHOT_DIR", tmp_path / "snapshots")
    lt._invalidate_predictions_cache()

    data = {
        "stats": {
            "home": {"shots_on_goal": 4, "shots_total": 7, "corner_kicks": 3, "ball_possession": "55%"},
            "away": {"shots_on_goal": 2, "shots_total": 4, "corner_kicks": 1, "ball_possession": "45%"},
        },
        "events": [{"time": {"elapsed": 34}, "team": {"id": 1}, "type": "Goal", "detail": "Normal Goal"}],
    }
    state = lt.ingest_live_snapshot(
        99, data, home_name="Mexico", away_name="South Africa",
        home_team_id=1, away_team_id=2, score_home=1, score_away=0, status="1H",
    )
    assert state.probabilities is not None
    assert "home_win" in state.probabilities
    saved = json.loads(pred_file.read_text(encoding="utf-8"))
    assert saved["ml_data"][0]["live_probabilities"]["home_win"] > 0
    conf = saved["ml_data"][0]["live_confidence"]
    assert conf is not None
