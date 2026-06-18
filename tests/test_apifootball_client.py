"""Unit tests for every apifootball_client public function (HTTP mocked)."""

from __future__ import annotations

import pytest

from apifootball_client import (
    APIFootballError,
    DAILY_LIMIT,
    calls_remaining,
    get_fixture_events,
    get_fixture_full,
    get_fixture_lineups,
    get_fixture_stats,
    get_live_fixtures,
    get_today_fixtures,
)
from tests.fixtures_data import (
    SAMPLE_EVENTS_RESPONSE,
    SAMPLE_FIXTURE,
    SAMPLE_LINEUPS_RESPONSE,
    SAMPLE_STATS_RESPONSE,
)


class TestBudgetCounter:
    def test_calls_remaining_starts_at_daily_limit(self, reset_api_budget):
        assert calls_remaining() == DAILY_LIMIT

    def test_calls_remaining_decrements_after_request(self, mock_api_response):
        mock_api_response({
            "/fixtures?date": {"response": [SAMPLE_FIXTURE], "errors": {}},
        })
        get_today_fixtures("2026-06-17")
        assert calls_remaining() == DAILY_LIMIT - 1

    def test_budget_exhausted_raises(self, reset_api_budget, monkeypatch):
        import apifootball_client as afc
        from datetime import date

        monkeypatch.setattr(afc, "_calls_made_today", DAILY_LIMIT)
        monkeypatch.setattr(afc, "_calls_date", date.today().isoformat())
        with pytest.raises(APIFootballError, match="budget exhausted"):
            get_today_fixtures("2026-06-17")


class TestAPIFootballError:
    def test_message_format(self):
        err = APIFootballError(400, "bad request")
        assert err.status == 400
        assert "400" in str(err)


class TestGetErrors:
    def test_missing_api_key(self, reset_api_budget, monkeypatch):
        import apifootball_client as afc

        monkeypatch.setattr(afc, "APIFOOTBALL_KEY", "")
        with pytest.raises(APIFootballError, match="not configured"):
            get_today_fixtures("2026-06-17")

    def test_http_error(self, reset_api_budget, monkeypatch):
        from unittest.mock import MagicMock

        def fail_get(*args, **kwargs):
            r = MagicMock()
            r.ok = False
            r.status_code = 503
            r.text = "Service Unavailable"
            return r

        session = MagicMock()
        session.get.side_effect = fail_get
        monkeypatch.setattr(reset_api_budget, "_session", session)
        with pytest.raises(APIFootballError, match="503"):
            get_today_fixtures("2026-06-17")

    def test_api_errors_dict(self, mock_api_response):
        mock_api_response({
            "/fixtures?date": {
                "response": [],
                "errors": {"plan": "Free plans do not have access to this season"},
            },
        })
        with pytest.raises(APIFootballError, match="400"):
            get_today_fixtures("2026-06-17")


class TestGetTodayFixtures:
    def test_returns_fixture_list(self, mock_api_response):
        mock_api_response({
            "/fixtures?date": {"response": [SAMPLE_FIXTURE], "errors": {}},
        })
        result = get_today_fixtures("2026-06-17")
        assert len(result) == 1
        assert result[0]["fixture"]["id"] == 12345

    def test_empty_response(self, mock_api_response):
        mock_api_response({"/fixtures?date": {"response": [], "errors": {}}})
        assert get_today_fixtures("2026-06-17") == []


class TestGetLiveFixtures:
    def test_returns_live_list(self, mock_api_response):
        mock_api_response({
            "/fixtures?live": {"response": [SAMPLE_FIXTURE], "errors": {}},
        })
        result = get_live_fixtures(league_id=1)
        assert len(result) == 1


class TestGetFixtureStats:
    def test_parses_home_and_away_stats(self, mock_api_response):
        mock_api_response({
            "/fixtures/statistics": {"response": SAMPLE_STATS_RESPONSE, "errors": {}},
        })
        stats = get_fixture_stats(12345, home_team_id=10, away_team_id=20)
        assert stats["home"]["shots_on_goal"] == 4
        assert stats["home"]["ball_possession"] == "55%"
        assert stats["home"]["expected_goals"] == "1.23"
        assert stats["away"]["shots_on_goal"] == 2
        assert stats["away"]["corner_kicks"] == 1

    def test_empty_stats_response(self, mock_api_response):
        mock_api_response({"/fixtures/statistics": {"response": [], "errors": {}}})
        stats = get_fixture_stats(12345)
        assert stats["home"]["shots_on_goal"] is None
        assert stats["away"]["shots_on_goal"] is None


class TestGetFixtureEvents:
    def test_returns_events(self, mock_api_response):
        mock_api_response({
            "/fixtures/events": {"response": SAMPLE_EVENTS_RESPONSE, "errors": {}},
        })
        events = get_fixture_events(12345)
        assert len(events) == 2
        assert events[0]["type"] == "Goal"
        assert events[1]["detail"] == "Yellow Card"


class TestGetFixtureLineups:
    def test_returns_home_and_away(self, mock_api_response):
        mock_api_response({
            "/fixtures/lineups": {"response": SAMPLE_LINEUPS_RESPONSE, "errors": {}},
        })
        lineups = get_fixture_lineups(12345)
        assert lineups["home"]["formation"] == "4-3-3"
        assert lineups["away"]["formation"] == "4-4-2"
        assert len(lineups["home"]["startXI"]) == 1


class TestGetFixtureFull:
    def test_combines_stats_and_events(self, reset_api_budget, monkeypatch):
        from unittest.mock import MagicMock

        responses = {
            "/fixtures/statistics": {"response": SAMPLE_STATS_RESPONSE, "errors": {}},
            "/fixtures/events": {"response": SAMPLE_EVENTS_RESPONSE, "errors": {}},
        }

        def route_get(url, params=None, headers=None, timeout=None):
            path = url.replace("https://v3.football.api-sports.io", "")
            resp = MagicMock()
            resp.ok = True
            resp.status_code = 200
            payload = responses[path]
            resp.json.return_value = payload
            resp.text = str(payload)
            return resp

        session = MagicMock()
        session.get.side_effect = route_get
        monkeypatch.setattr(reset_api_budget, "_session", session)

        result = get_fixture_full(12345, home_team_id=10, away_team_id=20)
        assert "stats" in result
        assert "events" in result
        assert result["stats"]["home"]["shots_on_goal"] == 4
        assert len(result["events"]) == 2
        assert "partial" not in result

    def test_partial_on_stats_failure(self, reset_api_budget, monkeypatch):
        from unittest.mock import MagicMock

        def route_get(url, params=None, headers=None, timeout=None):
            path = url.replace("https://v3.football.api-sports.io", "")
            resp = MagicMock()
            if path == "/fixtures/statistics":
                resp.ok = False
                resp.status_code = 500
                resp.text = "error"
                return resp
            resp.ok = True
            resp.json.return_value = {"response": SAMPLE_EVENTS_RESPONSE, "errors": {}}
            return resp

        session = MagicMock()
        session.get.side_effect = route_get
        monkeypatch.setattr(reset_api_budget, "_session", session)

        result = get_fixture_full(12345)
        assert result["partial"] is True
        assert len(result["events"]) == 2
