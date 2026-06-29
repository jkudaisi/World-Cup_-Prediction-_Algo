"""Tests for API-Football poll windows (T-15min through FT+15min)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import api_polling_window as apw
from tests.fixtures_data import SAMPLE_FIXTURE


def _fixture(kickoff: datetime, status: str = "NS", fid: int = 12345) -> dict:
    fx = dict(SAMPLE_FIXTURE)
    fx["fixture"] = dict(fx["fixture"])
    fx["fixture"]["id"] = fid
    fx["fixture"]["date"] = kickoff.isoformat()
    fx["fixture"]["status"] = {"short": status, "elapsed": None}
    return fx


class TestPollWindow:
    def test_live_always_in_window(self):
        now = datetime.now(timezone.utc)
        fx = _fixture(now, status="1H")
        assert apw.is_fixture_in_poll_window(fx, "1H", now=now) is True

    def test_far_future_outside_window(self):
        now = datetime.now(timezone.utc)
        kickoff = now + timedelta(hours=5)
        fx = _fixture(kickoff, status="NS")
        assert apw.is_fixture_in_poll_window(fx, "NS", now=now) is False

    def test_ten_minutes_before_kickoff_in_window(self):
        now = datetime.now(timezone.utc)
        kickoff = now + timedelta(minutes=10)
        fx = _fixture(kickoff, status="NS")
        assert apw.is_fixture_in_poll_window(fx, "NS", now=now) is True

    def test_twenty_minutes_before_kickoff_outside_window(self):
        now = datetime.now(timezone.utc)
        kickoff = now + timedelta(minutes=20)
        fx = _fixture(kickoff, status="NS")
        assert apw.is_fixture_in_poll_window(fx, "NS", now=now) is False

    def test_ten_minutes_after_ft_still_in_window(self):
        now = datetime.now(timezone.utc)
        kickoff = now - timedelta(minutes=100)
        finalized = now - timedelta(minutes=10)
        fx = _fixture(kickoff, status="FT")
        assert apw.is_fixture_in_poll_window(
            fx, "FT", now=now, finalized_at=finalized,
        ) is True

    def test_twenty_minutes_after_ft_outside_window(self):
        now = datetime.now(timezone.utc)
        kickoff = now - timedelta(minutes=110)
        finalized = now - timedelta(minutes=20)
        fx = _fixture(kickoff, status="FT")
        assert apw.is_fixture_in_poll_window(
            fx, "FT", now=now, finalized_at=finalized,
        ) is False

    def test_seconds_until_next_window_capped(self):
        now = datetime.now(timezone.utc)
        kickoff = now + timedelta(hours=2)
        fx = _fixture(kickoff, status="NS")
        secs = apw.seconds_until_next_poll_window(
            [fx], {12345: "NS"}, {}, now=now,
        )
        # Capped at max_sleep (30 min) when next window is far away.
        assert secs == 1800

    def test_seconds_until_next_window_near(self):
        now = datetime.now(timezone.utc)
        kickoff = now + timedelta(minutes=20)
        fx = _fixture(kickoff, status="NS")
        secs = apw.seconds_until_next_poll_window(
            [fx], {12345: "NS"}, {}, now=now,
        )
        assert 4 * 60 <= secs <= 6 * 60
