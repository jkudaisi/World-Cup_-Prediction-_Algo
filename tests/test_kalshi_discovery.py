"""Tests for Kalshi WC market discovery."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from kalshi_market_discovery import (
    _map_game_event,
    _map_total_event,
    apply_discoveries_to_mapping,
    discover_wc_markets,
    parse_event_date,
    parse_match_teams,
)


class TestKalshiDiscoveryParsing:
    def test_parse_event_date(self):
        assert parse_event_date("KXWCGAME-26JUN30FRASWE") == "2026-06-30"

    def test_parse_match_teams(self):
        assert parse_match_teams("France vs Sweden Winner?") == ("France", "Sweden")

    def test_map_game_event(self):
        markets = [
            {"ticker": "T-TIE", "title": "France vs Sweden Winner?", "yes_sub_title": "Reg Time: Tie"},
            {"ticker": "T-FRA", "title": "France vs Sweden Winner?", "yes_sub_title": "Reg Time: France"},
            {"ticker": "T-SWE", "title": "France vs Sweden Winner?", "yes_sub_title": "Reg Time: Sweden"},
        ]
        tickers = _map_game_event(markets)
        assert tickers == {"draw": "T-TIE", "home_win": "T-FRA", "away_win": "T-SWE"}

    def test_map_total_event(self):
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
