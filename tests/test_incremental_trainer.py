"""Tests for incremental training system."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

import incremental_trainer as it
import training_store as ts
from incremental_trainer import (
    SKIP_REASON,
    completed_match_to_training_row,
    fetch_new_completed_world_cup_matches,
    run_incremental_training,
    scan_live_snapshots_for_completed_matches,
    snapshot_to_training_row,
)
from training_store import append_wc_matches, load_training_state, save_training_state


SAMPLE_FIXTURE = {
    "fixture": {
        "id": 99001,
        "date": "2026-06-17T18:00:00+00:00",
        "status": {"short": "FT", "elapsed": 90},
    },
    "league": {"id": 1, "name": "World Cup"},
    "teams": {
        "home": {"id": 10, "name": "Portugal"},
        "away": {"id": 20, "name": "Congo DR"},
    },
    "goals": {"home": 2, "away": 1},
    "score": {"fulltime": {"home": 2, "away": 1}},
}

SAMPLE_STATS = {
    "home": {
        "shots_on_goal": 5, "corner_kicks": 4, "ball_possession": "58%",
        "fouls": 10, "yellow_cards": 2, "red_cards": 0, "shots_total": 12,
    },
    "away": {
        "shots_on_goal": 3, "corner_kicks": 2, "ball_possession": "42%",
        "fouls": 14, "yellow_cards": 3, "red_cards": 0, "shots_total": 8,
    },
}


@pytest.fixture
def isolated_training(tmp_path, monkeypatch):
    state_path = tmp_path / "training_state.json"
    wc_path = tmp_path / "data" / "world_cup_completed_matches.json"
    base_path = tmp_path / "data" / "base_training_cache.json"
    models_dir = tmp_path / "models"
    pred_path = tmp_path / "predictions.json"

    monkeypatch.setattr(ts, "TRAINING_STATE_PATH", state_path)
    monkeypatch.setattr(ts, "WC_MATCHES_PATH", wc_path)
    monkeypatch.setattr(ts, "BASE_CACHE_PATH", base_path)
    monkeypatch.setattr(ts, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(it, "PREDICTIONS_PATH", pred_path)

    import model_store as ms
    monkeypatch.setattr(ms, "MODELS_DIR", models_dir)

    return tmp_path


class TestTrainingStore:
    def test_load_default_state(self, isolated_training):
        state = load_training_state()
        assert state["trained_fixture_ids"] == []
        assert state["last_incremental_run_status"] == "never"

    def test_save_and_load_state(self, isolated_training):
        state = load_training_state()
        state["trained_fixture_ids"] = [1, 2, 3]
        save_training_state(state)
        loaded = load_training_state()
        assert loaded["trained_fixture_ids"] == [1, 2, 3]

    def test_models_exist_uses_meta_json(self, isolated_training):
        import json
        import model_store as ms

        assert ms.models_exist() is False
        ms.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        (ms.MODELS_DIR / "scaler.pkl").write_bytes(b"")
        (ms.MODELS_DIR / "meta.json").write_text(
            json.dumps({"feature_cols": ["elo_h"], "model_versions": {}}),
            encoding="utf-8",
        )
        assert ms.models_exist() is False
        (ms.MODELS_DIR / "poisson_home.pkl").write_bytes(b"")
        (ms.MODELS_DIR / "poisson_away.pkl").write_bytes(b"")
        assert ms.models_exist() is True

    def test_append_wc_deduplicates(self, isolated_training):
        row = completed_match_to_training_row(SAMPLE_FIXTURE, SAMPLE_STATS, [])
        assert row is not None
        append_wc_matches([row])
        append_wc_matches([row])
        from training_store import load_wc_matches
        assert len(load_wc_matches()) == 1


class TestCompletedMatchRow:
    def test_converts_portugal_congo(self):
        row = completed_match_to_training_row(SAMPLE_FIXTURE, SAMPLE_STATS, [])
        assert row is not None
        assert row["home_team"] == "Portugal"
        assert row["away_team"] == "DRC"
        assert row["goals_h"] == 2
        assert row["goals_a"] == 1
        assert row["home_shots_on_target"] == 5

    def test_missing_stats_uses_defaults(self):
        row = completed_match_to_training_row(SAMPLE_FIXTURE, {}, [])
        assert row is not None
        assert row["home_possession"] == 0.5
        assert row["home_shots_on_target"] == 0

    def test_unknown_team_returns_none(self):
        fx = dict(SAMPLE_FIXTURE)
        fx["teams"] = {"home": {"name": "Atlantis"}, "away": {"name": "Portugal"}}
        assert completed_match_to_training_row(fx, SAMPLE_STATS, []) is None


class TestTeamStatsUpdate:
    def test_update_team_stats_from_match_ema(self):
        import copy
        from wc2026_ml_pipeline import TEAM_STATS, update_team_stats_from_match

        backup = copy.deepcopy(TEAM_STATS)
        try:
            before = {
                "portugal_form": TEAM_STATS["Portugal"]["form"],
                "portugal_xg": TEAM_STATS["Portugal"]["xg"],
                "portugal_xga": TEAM_STATS["Portugal"]["xga"],
                "drc_xga": TEAM_STATS["DRC"]["xga"],
            }
            match = {
                "home_team": "Portugal",
                "away_team": "DRC",
                "goals_h": 2,
                "goals_a": 1,
                "home_xg_proxy": 1.8,
                "away_xg_proxy": 0.9,
            }
            update_team_stats_from_match(match)

            assert TEAM_STATS["Portugal"]["form"] == pytest.approx(0.75 * before["portugal_form"] + 0.25 * 3.0, rel=1e-3)
            assert TEAM_STATS["Portugal"]["xg"] == pytest.approx(0.75 * before["portugal_xg"] + 0.25 * 1.8, rel=1e-3)
            assert TEAM_STATS["Portugal"]["xga"] == pytest.approx(0.75 * before["portugal_xga"] + 0.25 * 1.0, rel=1e-3)
            assert TEAM_STATS["DRC"]["xga"] == pytest.approx(0.75 * before["drc_xga"] + 0.25 * 2.0, rel=1e-3)
        finally:
            TEAM_STATS.clear()
            TEAM_STATS.update(backup)

    def test_apply_team_stats_on_scan(self, isolated_training, tmp_path, monkeypatch):
        import copy
        import live_snapshot_store as lss
        from wc2026_ml_pipeline import TEAM_STATS, update_team_stats_from_match

        backup = copy.deepcopy(TEAM_STATS)
        try:
            monkeypatch.setattr(it, "update_team_stats_from_match", update_team_stats_from_match)
            monkeypatch.setattr(lss, "SNAPSHOT_DIR", tmp_path / "snapshots")
            snap_dir = tmp_path / "snapshots"
            snap_dir.mkdir(parents=True)
            completed = {
                "fixture_id": 99002,
                "snapshot_time": "2026-06-17T20:00:00+00:00",
                "minute": 90,
                "status": "2H",
                "score": {"home": 2, "away": 1},
                "stats": SAMPLE_STATS,
                "events": [],
                "home_name": "Portugal",
                "away_name": "Congo DR",
                "home_team_id": 10,
                "away_team_id": 20,
            }
            (snap_dir / "99002.json").write_text(json.dumps([completed]), encoding="utf-8")

            before_xg = TEAM_STATS["Portugal"]["xg"]
            scan_live_snapshots_for_completed_matches(snapshot_dir=snap_dir)
            assert TEAM_STATS["Portugal"]["xg"] != before_xg
        finally:
            TEAM_STATS.clear()
            TEAM_STATS.update(backup)


class TestSnapshotCompletedMatches:
    SAMPLE_SNAPSHOT = {
        "fixture_id": 99002,
        "snapshot_time": "2026-06-17T20:00:00+00:00",
        "minute": 90,
        "status": "2H",
        "score": {"home": 2, "away": 1},
        "stats": SAMPLE_STATS,
        "events": [],
        "home_name": "Portugal",
        "away_name": "Congo DR",
        "home_team_id": 10,
        "away_team_id": 20,
    }

    def test_snapshot_to_training_row_2h_at_90(self):
        row = snapshot_to_training_row(self.SAMPLE_SNAPSHOT)
        assert row is not None
        assert row["fixture_id"] == 99002
        assert row["goals_h"] == 2
        assert row["goals_a"] == 1
        assert row["source"] == "live_snapshot"

    def test_scan_skips_existing_and_incomplete(self, isolated_training, tmp_path, monkeypatch):
        import live_snapshot_store as lss

        monkeypatch.setattr(lss, "SNAPSHOT_DIR", tmp_path / "snapshots")
        snap_dir = tmp_path / "snapshots"
        snap_dir.mkdir(parents=True)

        completed = dict(self.SAMPLE_SNAPSHOT)
        (snap_dir / "99002.json").write_text(
            json.dumps([completed]), encoding="utf-8",
        )
        incomplete = dict(completed)
        incomplete.update({"fixture_id": 99003, "minute": 70, "status": "2H"})
        (snap_dir / "99003.json").write_text(
            json.dumps([incomplete]), encoding="utf-8",
        )

        append_wc_matches([
            snapshot_to_training_row(completed),
        ])

        added = scan_live_snapshots_for_completed_matches(snapshot_dir=snap_dir)
        assert added == []

        from training_store import load_wc_matches
        assert len(load_wc_matches()) == 1

    def test_scan_ignores_placeholder_rows_without_goals(self, isolated_training, tmp_path, monkeypatch):
        import live_snapshot_store as lss

        monkeypatch.setattr(lss, "SNAPSHOT_DIR", tmp_path / "snapshots")
        snap_dir = tmp_path / "snapshots"
        snap_dir.mkdir(parents=True)

        append_wc_matches([{"fixture_id": 99002, "home_team": "Portugal", "away_team": "DRC"}])

        completed = dict(self.SAMPLE_SNAPSHOT)
        (snap_dir / "99002.json").write_text(
            json.dumps([completed]), encoding="utf-8",
        )

        added = scan_live_snapshots_for_completed_matches(snapshot_dir=snap_dir)
        assert len(added) == 1
        assert added[0]["goals_h"] == 2

    def test_scan_appends_new_completed(self, isolated_training, tmp_path, monkeypatch):
        import live_snapshot_store as lss

        monkeypatch.setattr(lss, "SNAPSHOT_DIR", tmp_path / "snapshots")
        snap_dir = tmp_path / "snapshots"
        snap_dir.mkdir(parents=True)
        completed = dict(self.SAMPLE_SNAPSHOT)
        (snap_dir / "99002.json").write_text(
            json.dumps([completed]), encoding="utf-8",
        )

        added = scan_live_snapshots_for_completed_matches(snapshot_dir=snap_dir)
        assert len(added) == 1
        assert added[0]["fixture_id"] == 99002

        from training_store import load_wc_matches
        assert len(load_wc_matches()) == 1


class TestFetchNewMatches:
    def test_ignores_already_trained(self, isolated_training, monkeypatch):
        monkeypatch.setattr(
            it, "fetch_completed_world_cup_fixtures",
            lambda: [SAMPLE_FIXTURE],
        )
        monkeypatch.setattr(
            it, "get_fixture_full",
            lambda *a, **k: {"stats": SAMPLE_STATS, "events": []},
        )
        rows = fetch_new_completed_world_cup_matches(trained_fixture_ids={99001})
        assert rows == []

    def test_returns_untrained(self, isolated_training, monkeypatch):
        monkeypatch.setattr(
            it, "fetch_completed_world_cup_fixtures",
            lambda: [SAMPLE_FIXTURE],
        )
        monkeypatch.setattr(
            it, "get_fixture_full",
            lambda *a, **k: {"stats": SAMPLE_STATS, "events": []},
        )
        rows = fetch_new_completed_world_cup_matches(trained_fixture_ids=set())
        assert len(rows) == 1
        assert rows[0]["fixture_id"] == 99001


class TestIncrementalTraining:
    def test_skips_when_no_new_matches(self, isolated_training, monkeypatch):
        monkeypatch.setattr(it, "models_exist", lambda *a, **k: True)
        monkeypatch.setattr(it, "fetch_new_completed_world_cup_matches", lambda *a, **k: [])
        result = run_incremental_training(force=False, fetch_from_api=False)
        assert result["status"] == "skipped"
        assert result["reason"] == SKIP_REASON

    @patch("incremental_trainer.train_models_from_frame")
    @patch("incremental_trainer.predict_all_fixtures")
    @patch("incremental_trainer.save_artifacts")
    def test_trains_on_new_matches(
        self, mock_save, mock_predict, mock_train, isolated_training, monkeypatch,
    ):
        row = completed_match_to_training_row(SAMPLE_FIXTURE, SAMPLE_STATS, [])
        monkeypatch.setattr(it, "models_exist", lambda *a, **k: True)
        monkeypatch.setattr(
            it, "fetch_new_completed_world_cup_matches",
            lambda *a, **k: [row],
        )
        mock_train.return_value = ({"Poisson Regression": (MagicMock(), MagicMock())}, MagicMock())
        mock_predict.return_value = [{"mn": 1, "home": "Portugal", "away": "DRC",
                                       "models": {}, "ens_h": 2, "ens_a": 1, "ens": "2-1",
                                       "group": "K", "home_flag": "", "away_flag": ""}]

        with patch("incremental_trainer.build_combined_training_frame") as mock_combined:
            import pandas as pd
            from wc2026_ml_pipeline import get_feature_cols
            cols = get_feature_cols()
            base = {c: [1.0] for c in cols}
            base["goals_h"] = [1]
            base["goals_a"] = [1]
            base["is_synthetic"] = [False]
            frame = pd.DataFrame(base)
            mock_combined.return_value = (frame, frame)

            result = run_incremental_training(force=False, fetch_from_api=True, verbose=False)

        assert result["status"] == "success"
        assert result["new_matches_used"] == 1
        state = load_training_state()
        assert 99001 in state["trained_fixture_ids"]

    def test_pending_skips_already_trained_fixtures(self, isolated_training, monkeypatch):
        from incremental_trainer import (
            get_pending_training_ids,
            mark_fixture_for_training,
            process_pending_training,
        )

        save_training_state({"trained_fixture_ids": [99001], "last_incremental_run_status": "never"})
        mark_fixture_for_training(99001)
        monkeypatch.setattr(it, "run_incremental_training", MagicMock())

        result = process_pending_training()

        assert result is None
        assert get_pending_training_ids() == set()
        it.run_incremental_training.assert_not_called()
    def test_predictions_route_readonly(self, flask_client, monkeypatch):
        monkeypatch.setattr("server.save_predictions", MagicMock())
        flask_client.get("/api/predictions")
        import server
        server.save_predictions.assert_not_called()

    def test_train_incremental_endpoint(self, flask_client, monkeypatch):
        monkeypatch.setattr(
            "server.run_incremental_training",
            lambda **k: {"status": "skipped", "reason": SKIP_REASON},
        )
        res = flask_client.post("/api/train-incremental")
        assert res.status_code == 200
        assert res.get_json()["status"] == "skipped"
