"""Shared fixtures for API and endpoint tests."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from tests.fixtures_data import MINIMAL_PREDICTIONS


@pytest.fixture
def reset_api_budget(monkeypatch):
    import apifootball_client as afc

    monkeypatch.setattr(afc, "_calls_made_today", 0)
    monkeypatch.setattr(afc, "_calls_date", "")
    monkeypatch.setattr(afc, "_session", None)
    monkeypatch.setattr(afc, "APIFOOTBALL_KEY", "test-key")
    yield afc


@pytest.fixture
def mock_api_response(monkeypatch, reset_api_budget):
    """Patch session.get to return canned API-Football JSON wrappers."""
    afc = reset_api_budget

    def _make_mock(responses: dict):
        def fake_get(url, params=None, headers=None, timeout=None):
            path = url.replace("https://v3.football.api-sports.io", "")
            key = path
            if path == "/fixtures":
                if params and "live" in params:
                    key = "/fixtures?live"
                elif params and "date" in params:
                    key = "/fixtures?date"
            resp = MagicMock()
            payload = responses.get(key, {"response": [], "errors": {}})
            resp.ok = True
            resp.status_code = 200
            resp.json.return_value = payload
            resp.text = json.dumps(payload)
            return resp

        session = MagicMock()
        session.get.side_effect = fake_get
        monkeypatch.setattr(afc, "_session", session)
        return session

    return _make_mock


@pytest.fixture
def flask_client(tmp_path, monkeypatch):
    """Flask test client with scheduler disabled and temp predictions file."""
    import server as srv

    pred_file = tmp_path / "predictions.json"
    pred_file.write_text(json.dumps(MINIMAL_PREDICTIONS), encoding="utf-8")
    monkeypatch.setattr(srv, "PREDICTIONS_FILE", pred_file)
    monkeypatch.setattr(srv, "_scheduler_booted", True)
    monkeypatch.setattr(srv, "_ensure_scheduler", lambda: None)
    srv.app.config["TESTING"] = True
    with srv.app.test_client() as client:
        yield client
