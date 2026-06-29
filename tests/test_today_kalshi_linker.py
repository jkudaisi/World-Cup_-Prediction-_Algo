"""Tests for today Kalshi link startup cache."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import today_kalshi_linker as tkl


class TestTodayKalshiLinker:
    def test_cache_key_uses_kickoff_date(self):
        entry = {
            "fixture_id": 1,
            "ml_home": "South Africa",
            "ml_away": "Canada",
            "kickoff": "2026-06-28T19:00:00+00:00",
            "home": {"name": "South Africa"},
            "away": {"name": "Canada"},
        }
        assert tkl._cache_key_for_match(entry) == "South Africa|Canada|2026-06-28"

    @patch.object(tkl, "_apply_row_to_manual_mapping")
    @patch("kalshi_market_discovery.discover_wc_markets")
    @patch("kalshi_auth.credentials_configured", return_value=True)
    @patch("scheduler.get_today_view")
    def test_startup_links_and_caches(
        self, mock_today, mock_creds, mock_discover, _mock_apply, tmp_path, monkeypatch,
    ):
        monkeypatch.setattr(tkl, "TODAY_LINKS_PATH", tmp_path / "links.json")
        mock_today.return_value = {
            "matches": [{
                "fixture_id": 999,
                "ml_home": "South Africa",
                "ml_away": "Canada",
                "kickoff": "2026-06-28T19:00:00+00:00",
                "home": {"name": "South Africa"},
                "away": {"name": "Canada"},
            }],
        }
        mock_discover.return_value = {
            "matched": [{
                "home": "South Africa",
                "away": "Canada",
                "date": "2026-06-28",
                "fixture_key": "South Africa|Canada|2026-06-28",
                "kalshi_url": "https://kalshi.com/markets/kxwcgame/world-cup-game/kxwcgame-26jun28rsacan",
                "kalshi_advance_url": "https://kalshi.com/markets/kxwcadvance/world-cup-advance/kxwcadvance-26jun28rsacan?op_market_ticker=KXWCADVANCE-26JUN28RSACAN-CAN",
                "kalshi_event_ticker": "KXWCGAME-26JUN28RSACAN",
                "tickers": {
                    "home_win": "KXWCADVANCE-26JUN28RSACAN-RSA",
                    "away_win": "KXWCADVANCE-26JUN28RSACAN-CAN",
                },
            }],
            "unmatched_kalshi": [],
        }

        summary = tkl.refresh_today_kalshi_links_on_startup(force=True)
        assert summary["newly_linked"] == 1

        links = tkl.load_today_kalshi_links()
        key = "South Africa|Canada|2026-06-28"
        assert key in links["fixtures"]
        assert "kxwcadvance" in links["fixtures"][key]["kalshi_url"]

    @patch("kalshi_auth.credentials_configured", return_value=True)
    @patch("scheduler.get_today_view")
    def test_startup_skips_kalshi_when_cached(self, mock_today, mock_creds, tmp_path, monkeypatch):
        monkeypatch.setattr(tkl, "TODAY_LINKS_PATH", tmp_path / "links.json")
        entry = {
            "fixture_id": 999,
            "ml_home": "South Africa",
            "ml_away": "Canada",
            "kickoff": "2026-06-28T19:00:00+00:00",
            "home": {"name": "South Africa"},
            "away": {"name": "Canada"},
        }
        mock_today.return_value = {"matches": [entry]}
        tkl.save_today_kalshi_links({
            "fixtures": {
                "South Africa|Canada|2026-06-28": {
                    "tickers": {"home_win": "KXWCADVANCE-26JUN28RSACAN-RSA"},
                    "kalshi_url": "https://kalshi.com/markets/kxwcadvance/world-cup-advance/kxwcadvance-26jun28rsacan",
                },
            },
        })

        with patch("kalshi_market_discovery.discover_wc_markets") as mock_discover:
            summary = tkl.refresh_today_kalshi_links_on_startup(force=False)
            mock_discover.assert_not_called()
        assert summary["already_linked"] == 1

    def test_build_linked_matches_view_merges_discovery_and_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tkl, "TODAY_LINKS_PATH", tmp_path / "links.json")
        tkl.save_today_kalshi_links({
            "updated_at": "2026-06-28T12:00:00+00:00",
            "fixtures": {
                "South Africa|Canada|2026-06-28": {
                    "home": "South Africa",
                    "away": "Canada",
                    "date": "2026-06-28",
                    "linked_at": "2026-06-28T10:00:00+00:00",
                    "kalshi_advance_url": "https://kalshi.com/markets/kxwcadvance/world-cup-advance/kxwcadvance-26jun28rsacan",
                    "tickers": {"away_win": "KXWCADVANCE-26JUN28RSACAN-CAN"},
                },
            },
        })

        with patch("kalshi_market_discovery.load_discovery_cache") as mock_cache:
            mock_cache.return_value = {
                "discovered_at": "2026-06-28T11:00:00+00:00",
                "status": "ok",
                "matched": [{
                    "fixture_key": "South Africa|Canada|2026-06-28",
                    "home": "South Africa",
                    "away": "Canada",
                    "date": "2026-06-28",
                    "kalshi_url": "https://kalshi.com/markets/kxwcgame/world-cup-game/kxwcgame-26jun28rsacan",
                    "kalshi_advance_url": "https://kalshi.com/markets/kxwcadvance/world-cup-advance/kxwcadvance-26jun28rsacan",
                    "tickers": {"home_win": "KXWCADVANCE-26JUN28RSACAN-RSA"},
                }],
                "unmatched_kalshi": [],
                "game_events": 1,
                "advance_events": 1,
            }
            view = tkl.build_kalshi_linked_matches_view()

        assert view["count"] == 1
        match = view["matches"][0]
        assert match["home"] == "South Africa"
        assert "discovery" in match["sources"]
        assert "today_cache" in match["sources"]
        assert match["mapped_markets"] == 2
        assert "kxwcadvance" in match["primary_url"]
