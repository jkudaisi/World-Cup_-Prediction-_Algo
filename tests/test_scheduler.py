"""Tests for scheduler logic used by /api/today and /api/scheduler."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

import scheduler as sched
from tests.fixtures_data import SAMPLE_FIXTURE


@pytest.fixture(autouse=True)
def reset_scheduler_state(monkeypatch):
    monkeypatch.setattr(sched, "schedule", None)
    monkeypatch.setattr(sched, "cached_status", {})
    monkeypatch.setattr(sched, "_today_view_fetched", "")
    monkeypatch.setattr(sched, "_all_today_fixtures", [])
    monkeypatch.setattr(sched, "_scheduler_thread", None)
    monkeypatch.setattr(sched, "_last_status_refresh", None)


class TestMorningInit:
    def test_builds_schedule_from_fixtures(self, monkeypatch):
        monkeypatch.setattr(sched, "_fetch_wc_fixtures_for_local_day", lambda: [SAMPLE_FIXTURE])
        day = sched.morning_init()
        assert day is not None
        assert day.n_matches == 1
        assert len(sched._all_today_fixtures) == 1

    def test_skips_finished_in_active_count(self, monkeypatch):
        finished = dict(SAMPLE_FIXTURE)
        finished["fixture"] = dict(finished["fixture"])
        finished["fixture"]["status"] = {"short": "FT", "elapsed": 90}
        monkeypatch.setattr(sched, "_fetch_wc_fixtures_for_local_day", lambda: [finished])
        day = sched.morning_init()
        assert day.n_matches == 0


class TestAnyMatchWindow:
    def test_live_status_in_window(self):
        sched._all_today_fixtures = [SAMPLE_FIXTURE]
        sched.cached_status[12345] = "1H"
        assert sched._any_match_window() is True

    def test_future_match_outside_window(self):
        future = dict(SAMPLE_FIXTURE)
        kickoff = datetime.now(timezone.utc) + timedelta(hours=5)
        future["fixture"] = dict(future["fixture"])
        future["fixture"]["date"] = kickoff.isoformat()
        future["fixture"]["status"] = {"short": "NS", "elapsed": None}
        sched._all_today_fixtures = [future]
        sched.cached_status[12345] = "NS"
        assert sched._any_match_window() is False


class TestGetTodayView:
    def test_uses_schedule_fixtures(self, monkeypatch):
        from datetime import date as date_cls

        today = date_cls.today().strftime("%Y-%m-%d")
        sched.schedule = sched.DaySchedule(
            date=today, fixtures=[SAMPLE_FIXTURE], n_matches=1,
        )
        sched._all_today_fixtures = [SAMPLE_FIXTURE]
        sched.cached_status[12345] = "1H"
        view = sched.get_today_view()
        assert view["n_matches"] == 1
        assert view["matches"][0]["fixture_id"] == 12345
        assert view["live_count"] == 1

    def test_fetches_when_no_schedule(self, monkeypatch):
        monkeypatch.setattr(sched, "_fetch_wc_fixtures_for_local_day", lambda: [SAMPLE_FIXTURE])
        import config
        monkeypatch.setattr(config, "APIFOOTBALL_KEY", "test-key")
        view = sched.get_today_view()
        assert view["n_matches"] == 1
        assert "local_timezone" in view


class TestGetSchedulerStatus:
    def test_empty_schedule(self):
        status = sched.get_scheduler_status()
        assert status["n_matches"] == 0
        assert status["daily_limit"] == 7500
        assert "live_poll_interval_seconds" in status

    def test_with_active_schedule(self):
        sched.schedule = sched.DaySchedule(
            date="2026-06-17", fixtures=[SAMPLE_FIXTURE], n_matches=1, live_cycles=5,
        )
        status = sched.get_scheduler_status()
        assert status["n_matches"] == 1
        assert status["live_cycles"] == 5
