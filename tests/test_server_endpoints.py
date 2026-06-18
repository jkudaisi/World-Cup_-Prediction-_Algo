"""Tests for every Flask server route."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from tests.fixtures_data import MINIMAL_PREDICTIONS


ALL_ROUTES = [
    ("GET", "/", 200),
    ("GET", "/api/predictions", 200),
    ("GET", "/api/live", 200),
    ("GET", "/api/today", 200),
    ("GET", "/api/scheduler", 200),
    ("GET", "/api/live-states", 200),
    ("GET", "/api/live-snapshots/12345", 200),
    ("GET", "/api/training-state", 200),
    ("GET", "/api/status", 200),
]


class TestAllRoutesReachable:
    @pytest.mark.parametrize("method,path,expected_status", ALL_ROUTES)
    def test_route_returns_expected_status(self, flask_client, method, path, expected_status, monkeypatch):
        import scheduler as sched

        monkeypatch.setattr(sched, "schedule", None)
        monkeypatch.setattr(sched, "_today_view_fetched", "")

        if path == "/api/today":
            monkeypatch.setattr(
                "scheduler.get_today_view",
                lambda: {"date": "2026-06-17", "matches": [], "n_matches": 0, "live_count": 0},
            )
        if path == "/api/live":
            monkeypatch.setattr("server.run_live_cycle", lambda **k: {"live_count": 0})
            monkeypatch.setattr(
                "server.build_live_api_response",
                lambda: {**MINIMAL_PREDICTIONS, "live_meta": {}, "live_predictions": {}},
            )

        response = flask_client.open(path, method=method)
        assert response.status_code == expected_status


class TestIndex:
    def test_serves_html(self, flask_client):
        response = flask_client.get("/")
        assert response.status_code == 200
        assert b"WC 2026" in response.data


class TestPredictions:
    def test_returns_saved_json(self, flask_client):
        response = flask_client.get("/api/predictions")
        data = response.get_json()
        assert response.status_code == 200
        assert "ml_data" in data
        assert len(data["ml_data"]) == 1
        assert data["ml_data"][0]["home"] == "Mexico"


class TestLive:
    def test_returns_same_shape_as_predictions(self, flask_client, monkeypatch):
        monkeypatch.setattr("server.run_live_cycle", lambda **k: {"live_count": 0})
        monkeypatch.setattr(
            "server.build_live_api_response",
            lambda: {**MINIMAL_PREDICTIONS, "live_meta": {}, "live_predictions": {}},
        )
        response = flask_client.get("/api/live")
        data = response.get_json()
        assert "ml_data" in data
        assert data["ml_data"][0]["away"] == "South Africa"
        assert "live_meta" in data


class TestRun:
    def test_post_trains_and_returns_data(self, flask_client, monkeypatch):
        monkeypatch.setattr(
            "server.save_predictions",
            lambda path, verbose=False, incremental=True, **kw: {
                **MINIMAL_PREDICTIONS,
                "status": "success",
                "new_matches_used": 0,
            },
        )
        response = flask_client.post("/api/run")
        assert response.status_code == 200
        assert response.get_json()["ml_data"][0]["home"] == "Mexico"

    def test_post_error_returns_500(self, flask_client, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("pipeline failed")

        monkeypatch.setattr("server.save_predictions", boom)
        response = flask_client.post("/api/run")
        assert response.status_code == 500
        assert "pipeline failed" in response.get_json()["error"]


class TestToday:
    def test_today_view_structure(self, flask_client, monkeypatch):
        monkeypatch.setattr(
            "scheduler.get_today_view",
            lambda: {
                "date": "2026-06-17",
                "matches": [{
                    "fixture_id": 12345,
                    "kickoff": "2026-06-17T18:00:00+00:00",
                    "status": "1H",
                    "elapsed": 32,
                    "home": {"id": 10, "name": "Mexico"},
                    "away": {"id": 20, "name": "South Africa"},
                    "score": {"home": 1, "away": 0},
                    "is_live": True,
                }],
                "n_matches": 1,
                "live_count": 1,
                "api_budget_remaining": 7500,
                "daily_limit": 7500,
                "scheduler_active": True,
            },
        )
        data = flask_client.get("/api/today").get_json()
        assert data["n_matches"] == 1
        assert data["matches"][0]["home"]["name"] == "Mexico"
        assert data["live_count"] == 1


class TestScheduler:
    def test_scheduler_status_keys(self, flask_client, monkeypatch):
        import scheduler as sched
        from datetime import datetime, timezone

        monkeypatch.setattr(
            sched,
            "schedule",
            MagicMock(
                date="2026-06-17",
                n_matches=3,
                live_cycles=10,
                calls_used=5,
                fixtures=[],
            ),
        )
        monkeypatch.setattr(sched, "cached_status", {12345: "1H"})
        monkeypatch.setattr("live_updater.get_live_status", lambda: {"active_matches": 1})

        data = flask_client.get("/api/scheduler").get_json()
        for key in (
            "date", "n_matches", "live_poll_interval_seconds", "live_cycles",
            "calls_used_today", "api_budget_remaining", "daily_limit",
            "active_fixture_ids", "cached_statuses", "live_status",
        ):
            assert key in data, f"missing key: {key}"
        assert data["n_matches"] == 3
        assert data["cached_statuses"]["12345"] == "1H"


class TestLiveStates:
    def test_returns_dict(self, flask_client):
        data = flask_client.get("/api/live-states").get_json()
        assert isinstance(data, dict)


class TestStatus:
    def test_status_keys(self, flask_client):
        data = flask_client.get("/api/status").get_json()
        for key in (
            "predictions_exist",
            "predictions_age_seconds",
            "apifootball_key_configured",
            "apifootball_calls_remaining",
            "apifootball_daily_limit",
            "scheduler_running",
            "live_update",
            "server_time",
        ):
            assert key in data, f"missing key: {key}"
        assert data["predictions_exist"] is True
        assert data["apifootball_daily_limit"] == 7500


class TestTrainingRoutes:
    def test_training_state(self, flask_client):
        data = flask_client.get("/api/training-state").get_json()
        assert "trained_fixture_ids" in data
        assert "last_incremental_run_status" in data

    def test_train_incremental_skipped(self, flask_client, monkeypatch):
        monkeypatch.setattr(
            "server.run_incremental_training",
            lambda **k: {"status": "skipped", "reason": "No new completed World Cup matches since last training run"},
        )
        res = flask_client.post("/api/train-incremental")
        assert res.status_code == 200
        assert res.get_json()["status"] == "skipped"
