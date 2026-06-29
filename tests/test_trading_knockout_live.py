"""Trading + live integration for knockout fixtures."""

from __future__ import annotations

from unittest.mock import patch

import trading_service as ts
from kalshi_market_mapper import map_fixture_to_tickers


class TestKalshiAutoMatch:
    def test_winner_title_does_not_map_as_draw(self):
        markets = [{
            "title": "South Africa vs Canada Winner?",
            "ticker": "KXWCGAME-26JUN28RSACAN-TIE",
        }]
        auto = map_fixture_to_tickers(
            "South Africa", "Canada", date="2026-06-28", kalshi_markets=markets,
        )["auto_matched"]
        assert auto.get("draw") == "KXWCGAME-26JUN28RSACAN-TIE"
        assert auto.get("home_win") != "KXWCGAME-26JUN28RSACAN-TIE"


class TestTradingLiveResolution:
    def test_should_fetch_prices_during_poll_window(self):
        mapping = {"tickers": {"away_win": "KXWCGAME-TEST-CAN"}, "match_confidence": 0.5}
        assert ts._should_fetch_prices(
            mapping=mapping, is_live=False, mn=None, in_poll_window=True,
        )

    def test_resolve_live_from_scheduler_snapshot(self):
        match = {"home": "South Africa", "away": "Canada", "fixture_id": 1561329}
        sched = {
            1561329: {
                "fixture_id": 1561329,
                "status": "2H",
                "score_home": 0,
                "score_away": 1,
                "is_live": True,
            },
        }
        row, is_live, sh, sa, status, final = ts._resolve_fixture_live_state(
            match,
            live_by_team={},
            live_by_fixture={},
            sched_snaps=sched,
        )
        assert is_live is True
        assert sh == 0 and sa == 1
        assert status == "2H"
        assert final is False
        assert row is not None
        assert row["score"] == {"home": 0, "away": 1}

    def test_opportunities_cache_stale_when_discovery_newer(self, tmp_path, monkeypatch):
        cache_path = tmp_path / "opportunities.json"
        cache_path.write_text(
            '{"updated_at": 100.0, "opportunities": [{"x": 1}]}',
            encoding="utf-8",
        )
        monkeypatch.setattr(ts, "OPPORTUNITIES_CACHE", cache_path)
        monkeypatch.setattr(ts, "_in_api_poll_window", lambda: True)
        with patch("kalshi_market_discovery.load_discovery_cache") as mock_disc:
            mock_disc.return_value = {"discovered_at_ts": 200.0}
            assert ts._opportunities_cache_stale() is True
