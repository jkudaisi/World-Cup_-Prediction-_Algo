"""Tests for local calendar day fixture loading."""

from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import local_schedule as ls
from tests.fixtures_data import SAMPLE_FIXTURE


class TestLocalSchedule:
    def test_utc_dates_span_eastern_evening(self, monkeypatch):
        tz = ZoneInfo("America/New_York")
        local_day = date(2026, 6, 27)
        dates = ls.utc_dates_spanning_local_day(local_day, tz)
        assert "2026-06-27" in dates
        assert "2026-06-28" in dates

    def test_fixture_belongs_to_local_day_by_kickoff(self, monkeypatch):
        tz = ZoneInfo("America/New_York")
        fx = dict(SAMPLE_FIXTURE)
        fx["fixture"] = dict(fx["fixture"])
        # 9 PM ET June 27 = June 28 01:00 UTC
        fx["fixture"]["date"] = "2026-06-28T01:00:00+00:00"
        fx["fixture"]["status"] = {"short": "NS", "elapsed": None}
        assert ls.fixture_belongs_to_local_day(fx, date(2026, 6, 27), tz) is True
        assert ls.fixture_belongs_to_local_day(fx, date(2026, 6, 28), tz) is False

    def test_live_fixture_always_included(self):
        tz = ZoneInfo("UTC")
        fx = dict(SAMPLE_FIXTURE)
        fx["fixture"]["status"] = {"short": "1H", "elapsed": 20}
        assert ls.fixture_belongs_to_local_day(fx, date(2026, 6, 17), tz) is True

    def test_fetch_filters_by_local_kickoff(self, monkeypatch):
        tz = ZoneInfo("America/New_York")
        monkeypatch.setattr(ls, "get_display_tz", lambda: tz)

        evening_arg = dict(SAMPLE_FIXTURE)
        evening_arg["fixture"] = dict(evening_arg["fixture"])
        evening_arg["fixture"]["id"] = 999
        evening_arg["fixture"]["date"] = "2026-06-28T01:00:00+00:00"
        evening_arg["fixture"]["status"] = {"short": "NS", "elapsed": None}
        evening_arg["teams"] = {
            "home": {"id": 1, "name": "Argentina"},
            "away": {"id": 2, "name": "Brazil"},
        }

        def fake_fetch(date_str):
            if date_str == "2026-06-28":
                return [evening_arg]
            return []

        rows = ls.fetch_wc_fixtures_for_local_day(
            date(2026, 6, 27),
            fetch_by_date=fake_fetch,
        )
        assert len(rows) == 1
        assert rows[0]["teams"]["home"]["name"] == "Argentina"

    def test_merge_live_adds_missing(self, monkeypatch):
        def fake_live():
            live_fx = dict(SAMPLE_FIXTURE)
            live_fx["fixture"] = dict(live_fx["fixture"])
            live_fx["fixture"]["id"] = 777
            live_fx["fixture"]["status"] = {"short": "2H", "elapsed": 60}
            return [live_fx]

        import apifootball_client as afc
        monkeypatch.setattr(afc, "get_all_live_fixtures", fake_live)
        merged = ls.merge_live_wc_fixtures([])
        assert len(merged) == 1
        assert merged[0]["fixture"]["id"] == 777
