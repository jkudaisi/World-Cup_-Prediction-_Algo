"""Tests for future World Cup fixture prediction cache."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

import future_fixture_predictions as ffp
from tests.fixtures_data import SAMPLE_FIXTURE


def _future_fixture(
    *,
    fid: int = 900001,
    home: str = "Brazil",
    away: str = "Japan",
    home_id: int = 9,
    away_id: int = 22,
    kickoff: datetime | None = None,
    round_name: str = "Round of 16",
    status: str = "NS",
) -> dict:
    kickoff = kickoff or (datetime.now(timezone.utc) + timedelta(days=3))
    fx = dict(SAMPLE_FIXTURE)
    fx["fixture"] = dict(fx["fixture"])
    fx["fixture"]["id"] = fid
    fx["fixture"]["date"] = kickoff.isoformat()
    fx["fixture"]["status"] = {"short": status, "elapsed": None}
    fx["league"] = {"id": 1, "name": "World Cup", "season": 2026, "round": round_name}
    fx["teams"] = {
        "home": {"id": home_id, "name": home},
        "away": {"id": away_id, "name": away},
    }
    return fx


def _placeholder_fixture() -> dict:
    fx = _future_fixture(fid=900002, home="Winner Group A", away="Runner-up Group B", home_id=0, away_id=0)
    return fx


@pytest.fixture
def cache_path(tmp_path, monkeypatch):
    path = tmp_path / "future_fixture_prediction_cache.json"
    monkeypatch.setattr(ffp, "CACHE_PATH", path)
    return path


@pytest.fixture
def minimal_ml_match():
    return {
        "mn": 900001,
        "group": "R16",
        "home": "Brazil",
        "away": "Japan",
        "home_flag": "🇧🇷",
        "away_flag": "🇯🇵",
        "models": {},
        "ens_h": 2,
        "ens_a": 1,
        "ens": "2-1",
        "prediction": {"home_win": 0.55, "draw": 0.22, "away_win": 0.23},
        "confidence": {"score": 0.7},
        "explanation": {},
        "ensemble": {},
        "fixture_id": 900001,
        "source": "future_cache",
    }


class TestIsFixturePredictable:
    def test_confirmed_future_fixture_ok(self):
        ok, reason = ffp.is_fixture_predictable(_future_fixture())
        assert ok is True
        assert reason == "ok"

    def test_placeholder_teams_skipped(self):
        ok, reason = ffp.is_fixture_predictable(_placeholder_fixture())
        assert ok is False
        assert "placeholder" in reason or "missing" in reason

    def test_past_kickoff_skipped(self):
        past = datetime.now(timezone.utc) - timedelta(hours=2)
        ok, _ = ffp.is_fixture_predictable(_future_fixture(kickoff=past))
        assert ok is False


class TestCacheIO:
    def test_creates_cache_if_missing(self, cache_path):
        assert not cache_path.exists()
        cache = ffp.load_future_prediction_cache()
        assert cache["fixtures"] == {}
        ffp.save_future_prediction_cache(cache)
        assert cache_path.exists()

    def test_corrupt_cache_backed_up_and_reset(self, cache_path):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("{not json", encoding="utf-8")
        cache = ffp.load_future_prediction_cache()
        assert cache["fixtures"] == {}
        assert cache_path.with_suffix(".json.bak").exists()


class TestRefresh:
    @patch("future_fixture_predictions.models_exist", return_value=True)
    @patch("future_fixture_predictions.load_artifacts")
    @patch("future_fixture_predictions.fetch_future_world_cup_fixtures")
    @patch("future_fixture_predictions.predict_match")
    def test_predicts_new_fixture(
        self, mock_predict, mock_fetch, mock_artifacts, mock_models_exist, cache_path, minimal_ml_match,
    ):
        fx = _future_fixture()
        mock_fetch.return_value = [fx]
        mock_artifacts.return_value = {
            "trained": {},
            "scaler": None,
            "feature_cols": [],
            "model_versions": {"Poisson Regression": "1.0.0"},
        }
        mock_predict.return_value = minimal_ml_match

        result = ffp.refresh_future_fixture_predictions(force=False)
        assert result["found"] == 1
        assert result["predicted"] == 1
        assert result["already_cached"] == 0

        cache = ffp.load_future_prediction_cache()
        assert "900001" in cache["fixtures"]
        assert cache["fixtures"]["900001"]["home_team"] == "Brazil"

    @patch("future_fixture_predictions.models_exist", return_value=True)
    @patch("future_fixture_predictions.fetch_future_world_cup_fixtures")
    @patch("future_fixture_predictions.predict_match")
    def test_skips_already_cached(
        self, mock_predict, mock_fetch, mock_models_exist, cache_path, minimal_ml_match,
    ):
        fx = _future_fixture()
        mock_fetch.return_value = [fx]
        ffp.save_future_prediction_cache({
            "version": 1,
            "fixtures": {"900001": {"fixture_id": 900001, "ml_match": minimal_ml_match}},
        })

        result = ffp.refresh_future_fixture_predictions(force=False)
        assert result["already_cached"] == 1
        assert result["predicted"] == 0
        mock_predict.assert_not_called()

    @patch("future_fixture_predictions.models_exist", return_value=True)
    @patch("future_fixture_predictions.load_artifacts")
    @patch("future_fixture_predictions.fetch_future_world_cup_fixtures")
    @patch("future_fixture_predictions.predict_match")
    def test_force_refresh_repredicts(
        self, mock_predict, mock_fetch, mock_artifacts, mock_models_exist, cache_path, minimal_ml_match,
    ):
        fx = _future_fixture()
        mock_fetch.return_value = [fx]
        mock_artifacts.return_value = {
            "trained": {},
            "scaler": None,
            "feature_cols": [],
            "model_versions": {},
        }
        mock_predict.return_value = minimal_ml_match
        ffp.save_future_prediction_cache({
            "version": 1,
            "fixtures": {"900001": {"fixture_id": 900001, "ml_match": minimal_ml_match}},
        })

        result = ffp.refresh_future_fixture_predictions(force=True)
        assert result["predicted"] == 1
        mock_predict.assert_called_once()

    @patch("future_fixture_predictions.models_exist", return_value=True)
    @patch("future_fixture_predictions.fetch_future_world_cup_fixtures")
    def test_skips_tbd_placeholder(self, mock_fetch, mock_models_exist, cache_path):
        mock_fetch.return_value = [_placeholder_fixture()]
        result = ffp.refresh_future_fixture_predictions(force=False)
        assert result["predicted"] == 0
        assert len(result["skipped"]) == 1

    @patch("future_fixture_predictions.fetch_future_world_cup_fixtures")
    def test_api_failure_does_not_raise(self, mock_fetch, cache_path):
        from apifootball_client import APIFootballError

        mock_fetch.side_effect = APIFootballError(503, "unavailable")
        with patch("future_fixture_predictions.models_exist", return_value=True):
            result = ffp.refresh_future_fixture_predictions_on_startup()
        assert result["status"] == "error"


class TestMergeIntoPredictions:
    def test_merge_adds_cached_fixtures(self, cache_path, minimal_ml_match):
        ffp.save_future_prediction_cache({
            "version": 1,
            "fixtures": {
                "900001": {
                    "fixture_id": 900001,
                    "ml_match": minimal_ml_match,
                },
            },
        })
        doc = {"ml_data": [{"home": "Mexico", "away": "South Africa", "mn": 1}], "team_elo": {}}
        merged = ffp.merge_future_predictions_into_doc(doc)
        assert len(merged["ml_data"]) == 2
        assert merged["future_fixture_cache"]["added_to_ml_data"] == 1
        assert any(m.get("fixture_id") == 900001 for m in merged["ml_data"])

    def test_merge_skips_duplicate_pair(self, cache_path, minimal_ml_match):
        ffp.save_future_prediction_cache({
            "version": 1,
            "fixtures": {"900001": {"fixture_id": 900001, "ml_match": minimal_ml_match}},
        })
        doc = {"ml_data": [{"home": "Brazil", "away": "Japan", "mn": 1}]}
        merged = ffp.merge_future_predictions_into_doc(doc)
        assert len(merged["ml_data"]) == 1


class TestTodayPredictionAttach:
    @patch("future_fixture_predictions.ensure_predictions_for_fixtures", return_value=0)
    @patch("future_fixture_predictions.lookup_ml_prediction")
    def test_attach_ml_predictions(self, mock_lookup, _mock_ensure, minimal_ml_match):
        mock_lookup.return_value = minimal_ml_match
        matches = [{
            "fixture_id": 900001,
            "ml_home": "Brazil",
            "ml_away": "Japan",
            "home": {"name": "Brazil"},
            "away": {"name": "Japan"},
        }]
        raw = [_future_fixture()]
        ffp.attach_ml_predictions_to_today_matches(matches, raw)
        assert matches[0]["ml_prediction"]["ens"] == "2-1"


class TestServerEndpoints:
    def test_get_future_cache(self, flask_client, cache_path, monkeypatch):
        monkeypatch.setattr("server.CACHE_PATH", cache_path, raising=False)
        monkeypatch.setattr(ffp, "CACHE_PATH", cache_path)
        response = flask_client.get("/api/future-fixture-cache")
        assert response.status_code == 200
        assert "fixtures" in response.get_json()

    def test_predictions_includes_future_cache(self, flask_client, cache_path, minimal_ml_match, monkeypatch):
        monkeypatch.setattr(ffp, "CACHE_PATH", cache_path)
        ffp.save_future_prediction_cache({
            "version": 1,
            "fixtures": {"900001": {"fixture_id": 900001, "ml_match": minimal_ml_match}},
        })
        response = flask_client.get("/api/predictions")
        data = response.get_json()
        assert response.status_code == 200
        assert any(m.get("fixture_id") == 900001 for m in data["ml_data"])
        assert data["future_fixture_cache"]["count"] == 1

    def test_refresh_endpoint(self, flask_client, monkeypatch):
        monkeypatch.setattr(
            "server.refresh_future_fixture_predictions",
            lambda force=False: {"status": "ok", "predicted": 0, "found": 0, "already_cached": 0, "skipped": [], "errors": []},
        )
        monkeypatch.setattr("server.load_future_prediction_cache", lambda: {"fixtures": {}})
        response = flask_client.post("/api/future-fixture-cache/refresh")
        assert response.status_code == 200
        assert response.get_json()["refresh"]["status"] == "ok"
