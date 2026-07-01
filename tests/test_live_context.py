"""Tests for live context enrichment."""

import live_context as lc
import live_predictor as lp


def test_merge_event_cards_prefers_events():
    snap = {
        "events": [
            {"type": "Card", "detail": "Red Card", "team": {"id": 1}},
        ],
        "stats": {"home": {"red_cards": 0}, "away": {}},
        "home_team_id": 1,
        "away_team_id": 2,
    }
    merged = lc.merge_event_cards_into_stats(snap)
    assert merged["stats"]["home"]["red_cards"] == 1


def test_lineup_completeness_with_data():
    lineups = {
        "home": {"formation": "4-3-3", "startXI": [{"player": {}}] * 11},
        "away": {"formation": "4-4-2", "startXI": [{"player": {}}] * 11},
    }
    players = {
        "home": [{"players": [{"statistics": [{"games": {"rating": "7.2"}}]}]}],
        "away": [{"players": [{"statistics": [{"games": {"rating": "6.9"}}]}]}],
    }
    score = lc.lineup_completeness_score(lineups, players)
    assert score > 0.7


def test_live_prediction_uses_lineups_for_confidence():
    base = {"models": {"Poisson": {"rh": 1.5, "ra": 1.0, "gh": 2, "ga": 1}}}
    snap = {
        "minute": 30,
        "status": "1H",
        "score": {"home": 0, "away": 0},
        "stats": {
            "home": {"shots_on_goal": 2, "shots_total": 4, "dangerous_attacks": 20},
            "away": {"shots_on_goal": 1, "shots_total": 2, "dangerous_attacks": 8},
        },
        "events": [],
        "lineups": {
            "home": {"formation": "4-3-3", "startXI": [{"player": {}}] * 11},
            "away": {"formation": "4-4-2", "startXI": [{"player": {}}] * 11},
        },
        "players": {},
        "injuries": [],
    }
    out = lp.update_live_prediction_from_snapshot(snap, base)
    assert out["lineups_available"] is True
    assert out["confidence"]["score"] > 0.4
