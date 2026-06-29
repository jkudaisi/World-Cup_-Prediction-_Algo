"""Tests for Kalshi WC market discovery."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from kalshi_market_discovery import (
    _map_advance_event,
    _map_game_event,
    _map_total_event,
    apply_discoveries_to_mapping,
    discover_wc_markets,
    kalshi_advance_market_url,
    parse_event_date,
    parse_match_teams,
)


class TestKalshiDiscoveryParsing:
    def test_parse_event_date(self):
        assert parse_event_date("KXWCGAME-26JUN30FRASWE") == "2026-06-30"

    def test_parse_match_teams(self):
        assert parse_match_teams("France vs Sweden Winner?") == ("France", "Sweden")

    def test_parse_match_teams_advance(self):
        assert parse_match_teams("South Africa vs Canada Advance?") == ("South Africa", "Canada")

    def test_parse_match_teams_to_advance(self):
        assert parse_match_teams("Brazil vs Japan: To Advance") == ("Brazil", "Japan")

    def test_map_game_event(self):
        markets = [
            {"ticker": "T-TIE", "title": "France vs Sweden Winner?", "yes_sub_title": "Reg Time: Tie"},
            {"ticker": "T-FRA", "title": "France vs Sweden Winner?", "yes_sub_title": "Reg Time: France"},
            {"ticker": "T-SWE", "title": "France vs Sweden Winner?", "yes_sub_title": "Reg Time: Sweden"},
        ]
        tickers = _map_game_event(markets)
        assert tickers == {"draw": "T-TIE", "home_win": "T-FRA", "away_win": "T-SWE"}

    def test_map_advance_event(self):
        markets = [
            {"ticker": "KXWCADVANCE-26JUN29BRAJPN-BRA", "yes_sub_title": "Brazil advances"},
            {"ticker": "KXWCADVANCE-26JUN29BRAJPN-JPN", "yes_sub_title": "Japan advances"},
        ]
        tickers = _map_advance_event(markets, "Brazil", "Japan")
        assert tickers["home_advance"] == "KXWCADVANCE-26JUN29BRAJPN-BRA"
        assert tickers["away_advance"] == "KXWCADVANCE-26JUN29BRAJPN-JPN"
        markets = [
            {"ticker": "KXWCTOTAL-26JUN30FRASWE-3", "title": "Over 2.5"},
            {"ticker": "KXWCTOTAL-26JUN30FRASWE-4", "title": "Over 3.5"},
        ]
        tickers = _map_total_event(markets)
        assert tickers["over_2_5"] == "KXWCTOTAL-26JUN30FRASWE-3"
        assert tickers["over_3_5"] == "KXWCTOTAL-26JUN30FRASWE-4"


class TestKalshiDiscoveryIntegration:
    def test_discover_wc_markets_mock(self, tmp_path, monkeypatch):
        cache_path = tmp_path / "discovered.json"
        mapping_path = tmp_path / "mapping.json"
        monkeypatch.setattr("kalshi_market_discovery.DISCOVERY_CACHE_PATH", cache_path)
        monkeypatch.setattr("kalshi_market_discovery.MAPPING_PATH", mapping_path)

        mock_client = MagicMock()

        def fake_fetch(_client, series, **kwargs):
            if series == "KXWCGAME":
                return [
                    {"event_ticker": "KXWCGAME-26JUN11MEXZAF", "ticker": "KXWCGAME-26JUN11MEXZAF-MEX",
                     "title": "Mexico vs South Africa Winner?", "yes_sub_title": "Reg Time: Mexico"},
                    {"event_ticker": "KXWCGAME-26JUN11MEXZAF", "ticker": "KXWCGAME-26JUN11MEXZAF-TIE",
                     "title": "Mexico vs South Africa Winner?", "yes_sub_title": "Reg Time: Tie"},
                    {"event_ticker": "KXWCGAME-26JUN11MEXZAF", "ticker": "KXWCGAME-26JUN11MEXZAF-ZAF",
                     "title": "Mexico vs South Africa Winner?", "yes_sub_title": "Reg Time: South Africa"},
                ]
            if series == "KXWCBTTS":
                return [{"event_ticker": "KXWCBTTS-26JUN11MEXZAF", "ticker": "KXWCBTTS-26JUN11MEXZAF-BTTS",
                         "title": "Will both teams score?"}]
            if series == "KXWCTOTAL":
                return [{"event_ticker": "KXWCTOTAL-26JUN11MEXZAF", "ticker": "KXWCTOTAL-26JUN11MEXZAF-3",
                         "title": "Over 2.5"}]
            if series == "KXWCADVANCE":
                return []
            return []

        monkeypatch.setattr("kalshi_market_discovery.fetch_series_markets", fake_fetch)

        ml_data = [{"mn": 1, "home": "Mexico", "away": "South Africa", "fixture_id": 1}]
        result = discover_wc_markets(mock_client, ml_data=ml_data)
        assert result["matched_fixtures"] == 1
        assert result["matched"][0]["tickers"]["home_win"] == "KXWCGAME-26JUN11MEXZAF-MEX"
        assert result["matched"][0]["tickers"]["btts_yes"] == "KXWCBTTS-26JUN11MEXZAF-BTTS"

        apply = apply_discoveries_to_mapping(result)
        assert apply["keys_updated"] >= 1
        saved = json.loads(mapping_path.read_text(encoding="utf-8"))
        assert "Mexico|South Africa|2026-06-11" in saved or "mn:1" in saved

    def test_discover_advance_markets(self, tmp_path, monkeypatch):
        cache_path = tmp_path / "discovered.json"
        monkeypatch.setattr("kalshi_market_discovery.DISCOVERY_CACHE_PATH", cache_path)

        def fake_fetch(_client, series, **kwargs):
            if series == "KXWCADVANCE":
                return [
                    {"event_ticker": "KXWCADVANCE-26JUN28RSACAN", "ticker": "KXWCADVANCE-26JUN28RSACAN-RSA",
                     "title": "South Africa vs Canada Advance?", "yes_sub_title": "South Africa"},
                    {"event_ticker": "KXWCADVANCE-26JUN28RSACAN", "ticker": "KXWCADVANCE-26JUN28RSACAN-CAN",
                     "title": "South Africa vs Canada Advance?", "yes_sub_title": "Canada"},
                ]
            return []

        monkeypatch.setattr("kalshi_market_discovery.fetch_series_markets", fake_fetch)

        today = [{
            "fixture_id": 999001,
            "ml_home": "South Africa",
            "ml_away": "Canada",
            "kickoff": "2026-06-28T19:00:00+00:00",
            "home": {"name": "South Africa"},
            "away": {"name": "Canada"},
        }]
        result = discover_wc_markets(MagicMock(), ml_data=[], today_matches=today)
        assert result["matched_fixtures"] == 1
        row = result["matched"][0]
        assert row["tickers"]["home_advance"] == "KXWCADVANCE-26JUN28RSACAN-RSA"
        assert row["tickers"]["away_win"] == "KXWCADVANCE-26JUN28RSACAN-CAN"
        assert row["kalshi_advance_url"] == kalshi_advance_market_url(
            "KXWCADVANCE-26JUN28RSACAN", "KXWCADVANCE-26JUN28RSACAN-RSA"
        )

    def test_map_advance_event(self):
        markets = [
            {"ticker": "KXWCADVANCE-26JUN28RSACAN-RSA", "yes_sub_title": "South Africa"},
            {"ticker": "KXWCADVANCE-26JUN28RSACAN-CAN", "yes_sub_title": "Canada"},
        ]
        tickers = _map_advance_event(markets, "South Africa", "Canada")
        assert tickers["home_advance"] == "KXWCADVANCE-26JUN28RSACAN-RSA"
        assert tickers["away_win"] == "KXWCADVANCE-26JUN28RSACAN-CAN"
