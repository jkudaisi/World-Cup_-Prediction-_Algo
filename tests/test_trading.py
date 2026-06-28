"""Tests for Kalshi trading layer."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from edge_engine import evaluate_edge
from goal_markets import build_goal_markets
from kalshi_auth import sign_message
from kalshi_market_mapper import map_fixture_to_tickers, resolve_kalshi_team
from market_pricing import cents_to_prob, parse_orderbook
from paper_trader import is_duplicate_trade, paper_stats, simulate_fill
from risk_manager import evaluate_risk, kelly_stake, stake_to_contracts
from score_matrix import build_score_matrix
from trading_config import can_place_live_orders, get_config


class TestKalshiAuth:
    def test_sign_message_deterministic_length(self):
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        sig = sign_message(key, "1703123456789GET/trade-api/v2/markets")
        assert isinstance(sig, str)
        assert len(sig) > 20


class TestMarketPricing:
    def test_parse_orderbook_yes_no(self):
        ob = {
            "orderbook": {
                "yes": [[45, 30], [44, 10]],
                "no": [[55, 25], [54, 15]],
            }
        }
        p = parse_orderbook("TEST-TICKER", ob)
        assert p["ticker"] == "TEST-TICKER"
        assert p["best_yes_bid"] == 45
        assert p["best_no_bid"] == 55
        assert p["best_yes_ask"] == 45  # 100 - 55
        assert p["implied_probability"] == cents_to_prob(45)
        assert p["available_liquidity"] == 55

    def test_parse_orderbook_fp_dollars(self):
        from market_pricing import parse_market_quotes
        ob = {
            "orderbook_fp": {
                "yes_dollars": [["0.8400", "100"], ["0.8300", "50"]],
                "no_dollars": [["0.1500", "80"], ["0.1400", "40"]],
            }
        }
        p = parse_orderbook("KXWCGAME-TEST-ARG", ob)
        assert p["best_yes_bid"] == 84.0
        assert p["best_yes_ask"] == 85.0
        assert p["implied_probability"] == pytest.approx(0.845, rel=1e-3)

    def test_parse_market_quotes_dollars(self):
        from market_pricing import parse_market_quotes
        p = parse_market_quotes("T", {
            "market": {
                "yes_bid_dollars": "0.8400",
                "yes_ask_dollars": "0.8500",
                "yes_bid_size_fp": "1000",
            }
        })
        assert p["implied_probability"] == pytest.approx(0.845, rel=1e-3)

    def test_cents_to_prob(self):
        assert cents_to_prob(50) == 0.5
        assert cents_to_prob(100) == 1.0
        assert cents_to_prob(0) == 0.0


class TestEntryRanking:
    def test_ranked_opportunities_prefers_highest_edge_trade(self):
        from position_exits import best_opportunity, ranked_opportunities

        opps = [
            {
                "market_type": "home_win",
                "model_probability": 0.142,
                "recommendation": "SKIP",
                "edge": 0.027,
            },
            {
                "market_type": "btts_yes",
                "model_probability": 0.476,
                "recommendation": "TRADE",
                "edge": 0.199,
            },
            {
                "market_type": "over_3_5",
                "model_probability": 0.345,
                "recommendation": "TRADE",
                "edge": 0.110,
            },
        ]
        ranked = ranked_opportunities(opps, trade_only=True)
        assert [o["market_type"] for o in ranked] == ["btts_yes", "over_3_5"]
        assert best_opportunity(opps)["market_type"] == "btts_yes"


class TestMarketMapper:
    def test_team_aliases(self):
        assert resolve_kalshi_team("United States") == "USA"
        assert resolve_kalshi_team("Korea Republic") == "South Korea"
        assert resolve_kalshi_team("Côte d'Ivoire") == "Ivory Coast"
        assert resolve_kalshi_team("Congo DR") == "DRC"

    def test_manual_mapping(self, tmp_path, monkeypatch):
        mapping = {
            "Portugal|DRC|2026-06-20": {
                "home_win": "KXWCGAME-26JUN20PORDRC-POR",
                "draw": "KXWCGAME-26JUN20PORDRC-TIE",
            }
        }
        p = tmp_path / "kalshi_market_mapping.json"
        p.write_text(json.dumps(mapping), encoding="utf-8")
        monkeypatch.setattr("kalshi_market_mapper.MAPPING_PATH", p)
        result = map_fixture_to_tickers("Portugal", "DRC", mn=21)
        assert result["tickers"].get("home_win") == "KXWCGAME-26JUN20PORDRC-POR"

    def test_list_unmapped_fixtures(self, tmp_path, monkeypatch):
        from kalshi_market_mapper import list_unmapped_fixtures

        mapping = {"mn:21": {"home_win": "KXWCGAME-26JUN20PORDRC-POR"}}
        p = tmp_path / "kalshi_market_mapping.json"
        p.write_text(json.dumps(mapping), encoding="utf-8")
        monkeypatch.setattr("kalshi_market_mapper.MAPPING_PATH", p)

        ml_data = [
            {"mn": 1, "home": "Mexico", "away": "South Africa", "fixture_id": 1489369},
            {"mn": 21, "home": "Portugal", "away": "DRC", "fixture_id": 1539003},
        ]
        unmapped = list_unmapped_fixtures(ml_data)
        assert len(unmapped) == 1
        assert unmapped[0]["home_team"] == "Mexico"
        assert unmapped[0]["away_team"] == "South Africa"
        assert unmapped[0]["date"] == "2026-06-11"
        assert unmapped[0]["fixture_id"] == 1489369


class TestEdgeEngine:
    def test_trade_on_large_edge(self):
        r = evaluate_edge(
            model_probability=0.72,
            market_implied_probability=0.55,
            confidence=0.75,
            spread=3.0,
            liquidity=50,
            market_type="home_win",
            mapping_confidence=0.95,
        )
        assert r["decision"] == "TRADE"
        assert r["edge"] >= 0.08

    def test_skip_low_edge(self):
        r = evaluate_edge(
            model_probability=0.52,
            market_implied_probability=0.50,
            confidence=0.75,
            spread=2.0,
            liquidity=50,
            market_type="home_win",
        )
        assert r["decision"] == "SKIP"
        assert "Edge too small" in r["reason"]

    def test_skip_illiquid(self):
        r = evaluate_edge(
            model_probability=0.70,
            market_implied_probability=0.50,
            confidence=0.75,
            spread=2.0,
            liquidity=5,
            market_type="home_win",
        )
        assert r["decision"] == "SKIP"
        assert "illiquid" in r["reason"].lower()


class TestKellyStake:
    def test_fractional_kelly_high_confidence(self):
        # edge=0.10, market=0.50, full Kelly=0.20, 50% fraction -> 10% of bankroll
        assert kelly_stake(100.0, 0.10, 0.60, 0.85) == 10.0

    def test_fractional_kelly_mid_confidence(self):
        # 35% of full Kelly at confidence 0.75
        stake = kelly_stake(100.0, 0.10, 0.60, 0.75)
        assert stake == pytest.approx(7.0, rel=1e-3)

    def test_fractional_kelly_low_confidence(self):
        stake = kelly_stake(100.0, 0.10, 0.60, 0.65)
        assert stake == pytest.approx(5.0, rel=1e-3)

    def test_hard_cap_at_15_percent(self):
        assert kelly_stake(100.0, 0.50, 0.90, 0.90) == 15.0

    def test_zero_edge_returns_zero(self):
        assert kelly_stake(100.0, 0.0, 0.60, 0.90) == 0.0

    def test_stake_to_contracts(self):
        assert stake_to_contracts(2.0, 40.0) == 5
        assert stake_to_contracts(0.30, 40.0) == 0


class TestRiskManager:
    def test_approve_kelly_sized_stake(self):
        max_stake = kelly_stake(20.0, 0.15, 0.65, 0.85)
        r = evaluate_risk(
            stake=max_stake * 0.5,
            fixture_key="A|B|2026-06-11",
            bankroll=20.0,
            edge=0.15,
            model_p=0.65,
            confidence=0.85,
        )
        assert r["approved"] is True
        assert r["max_allowed_stake"] == max_stake

    def test_reject_oversized_stake(self):
        max_stake = kelly_stake(20.0, 0.08, 0.55, 0.60)
        r = evaluate_risk(
            stake=max_stake + 5.0,
            fixture_key="A|B|2026-06-11",
            bankroll=20.0,
            edge=0.08,
            model_p=0.55,
            confidence=0.60,
        )
        assert r["approved"] is False
        assert "Kelly" in r["reason"]


class TestScoreMatrix:
    def test_outcomes_sum_to_one(self):
        m = build_score_matrix(1.3, 1.1)
        total = m["home_win"] + m["draw"] + m["away_win"]
        assert abs(total - 1.0) < 0.02
        assert len(m["top_exact_scores"]) == 5
        assert "0-0" in m["score_matrix"]


class TestGoalMarkets:
    def test_over_under_complement(self):
        gm = build_goal_markets(1.4, 1.2)
        assert abs(gm["over_2_5"] + gm["under_2_5"] - 1.0) < 0.01
        assert abs(gm["btts_yes"] + gm["btts_no"] - 1.0) < 0.01
        assert len(gm["exact_score_top_5"]) == 5

    def test_live_score_adjustment(self):
        gm = build_goal_markets(0.5, 1.0, score_h=1, score_a=0, live=True)
        assert gm["over_0_5"] == 1.0


class TestScanOpportunities:
    def test_scan_includes_btts_and_exact_scores(self, monkeypatch):
        from trading_service import scan_opportunities

        monkeypatch.setattr("trading_service.fetch_orderbook_safe", lambda *a, **k: None)

        match = {
            "mn": 1,
            "home": "Mexico",
            "away": "South Africa",
            "prediction": {
                "home_win": 0.82,
                "draw": 0.12,
                "away_win": 0.06,
                "both_teams_score": 0.29,
                "over_2_5": 0.66,
                "over_3_5": 0.45,
                "over_under": {
                    "2.5": {"over": 0.66, "under": 0.34},
                    "3.5": {"over": 0.45, "under": 0.55},
                },
                "score_matrix": {
                    "top_exact_scores": [
                        {"score": "3-0", "probability": 0.18},
                        {"score": "2-0", "probability": 0.15},
                        {"score": "1-0", "probability": 0.12},
                    ],
                },
            },
            "confidence": {"score": 0.7},
        }
        goal_mkts = build_goal_markets(3.0, 0.4)
        mapping = {"tickers": {}, "match_confidence": 0.0}
        opps = scan_opportunities(
            match=match,
            mapping=mapping,
            goal_mkts=goal_mkts,
            live_row=None,
            fkey="Mexico|South Africa|2026-06-11",
            client=MagicMock(),
        )
        types = {o["market_type"] for o in opps}
        assert {"home_win", "draw", "away_win", "btts_yes", "btts_no", "over_2_5", "over_3_5"}.issubset(types)
        assert "exact_score_3_0" in types
        assert "exact_score_2_0" in types
        assert "exact_score_1_0" in types
        assert all(o.get("market") for o in opps)
        btts_no = next(o for o in opps if o["market_type"] == "btts_no")
        assert btts_no["model_probability"] == pytest.approx(0.71, rel=1e-2)

    def test_model_prob_from_envelope_over_under(self):
        from trading_service import _model_prob_from_envelope

        match = {
            "prediction": {
                "over_under": {"2.5": {"over": 0.55, "under": 0.45}},
            },
        }
        assert _model_prob_from_envelope(match, {}, "over_2_5") == 0.55


class TestLiveRowDetection:
    def test_stale_minute_not_live(self):
        from trading_service import _is_live_row

        assert _is_live_row({"minute": 90, "status": "FT"}) is False
        assert _is_live_row({"minute": 36, "status": "1H"}) is True
        assert _is_live_row({"minute": 45, "status": "HT"}) is True
        assert _is_live_row({"minute": 90, "status": "2H"}) is True
        assert _is_live_row(None) is False


class TestPaperPositionManager:
    def test_exit_stale_positions_model_reversal(self, tmp_path, monkeypatch):
        paper_path = tmp_path / "paper_trades.json"
        trade = {
            "trade_id": "t1",
            "duplicate_key": "k",
            "ticker": "PAPER",
            "side": "yes",
            "count": 1,
            "entry_price_cents": 60.0,
            "cost": 0.6,
            "model_probability": 0.70,
            "model_probability_at_entry": 0.70,
            "market_probability_at_entry": 0.60,
            "edge_at_entry": 0.10,
            "confidence": 0.7,
            "fixture_key": "A|B|2026-06-26",
            "home": "A",
            "away": "B",
            "market_type": "home_win",
            "status": "open",
            "opened_at": "2026-06-26T12:00:00+00:00",
            "pnl": 0.0,
        }
        paper_path.write_text(
            json.dumps({"trades": [trade], "bankroll": 19.4, "starting_bankroll": 20}),
            encoding="utf-8",
        )
        monkeypatch.setattr("paper_trader.PAPER_TRADES_PATH", paper_path)
        monkeypatch.setattr("trade_logger.DECISIONS_PATH", tmp_path / "decisions.json")

        fixtures = [{
            "fixture_key": "A|B|2026-06-26",
            "home": "A",
            "away": "B",
            "live": True,
            "goal_markets": {"outcomes": {"home_win": 0.55, "draw": 0.25, "away_win": 0.20}},
            "opportunities": [{
                "market_type": "home_win",
                "model_probability": 0.55,
                "kalshi_pct": 55.0,
                "fixture_key": "A|B|2026-06-26",
            }],
        }]
        from paper_position_manager import exit_stale_positions

        closed = exit_stale_positions(fixtures)
        assert len(closed) == 1
        assert closed[0]["trade"]["exit_reason"] == "model_reversal"
        assert closed[0]["trade"]["exit_method"] == "opposite_side_hedge"
        assert closed[0]["trade"]["hedge_side"] == "no"
        # YES 60¢ + NO 45¢ hedge → $1 payout → -5¢ P/L
        assert closed[0]["pnl"] == pytest.approx(-0.05, abs=0.01)

    def test_mark_and_exit_on_edge_reversal(self, tmp_path, monkeypatch):
        paper_path = tmp_path / "paper_trades.json"
        trade = {
            "trade_id": "t1",
            "duplicate_key": "k",
            "ticker": "PAPER",
            "side": "yes",
            "count": 1,
            "entry_price_cents": 60.0,
            "cost": 0.6,
            "model_probability": 0.70,
            "model_probability_at_entry": 0.70,
            "market_probability_at_entry": 0.60,
            "edge_at_entry": 0.10,
            "confidence": 0.7,
            "fixture_key": "A|B|2026-06-26",
            "home": "A",
            "away": "B",
            "market_type": "home_win",
            "status": "open",
            "opened_at": "2026-06-26T12:00:00+00:00",
            "pnl": 0.0,
        }
        paper_path.write_text(
            json.dumps({"trades": [trade], "bankroll": 19.4, "starting_bankroll": 20}),
            encoding="utf-8",
        )
        monkeypatch.setattr("paper_trader.PAPER_TRADES_PATH", paper_path)

        fixtures = [{
            "fixture_key": "A|B|2026-06-26",
            "home": "A",
            "away": "B",
            "live": True,
            "goal_markets": {"outcomes": {"home_win": 0.68, "draw": 0.20, "away_win": 0.12}},
            "opportunities": [{
                "market_type": "home_win",
                "model_probability": 0.68,
                "kalshi_pct": 75.0,
                "fixture_key": "A|B|2026-06-26",
            }],
        }]
        from paper_position_manager import manage_paper_exits, update_paper_marks

        n = update_paper_marks(fixtures)
        assert n == 1
        closed = manage_paper_exits(fixtures)
        assert len(closed) == 1
        assert closed[0]["trade"]["status"] == "closed"
        assert "Edge reversed" in closed[0]["trade"]["exit_reason"]


class TestLivePositionManager:
    def test_exit_stale_live_positions_model_reversal(self, tmp_path, monkeypatch):
        positions_path = tmp_path / "live_positions.json"
        position = {
            "position_id": "p1",
            "entry_order_id": "o1",
            "ticker": "LIVE-TICKER",
            "side": "yes",
            "count": 1,
            "entry_price_cents": 60.0,
            "cost": 0.6,
            "model_probability": 0.70,
            "model_probability_at_entry": 0.70,
            "market_probability_at_entry": 0.60,
            "edge_at_entry": 0.10,
            "confidence": 0.7,
            "fixture_key": "A|B|2026-06-26",
            "home": "A",
            "away": "B",
            "market_type": "home_win",
            "status": "open",
            "opened_at": "2026-06-26T12:00:00+00:00",
        }
        positions_path.write_text(json.dumps({"positions": [position]}), encoding="utf-8")
        monkeypatch.setattr("live_trader.LIVE_POSITIONS_PATH", positions_path)
        monkeypatch.setattr("trade_logger.DECISIONS_PATH", tmp_path / "decisions.json")
        monkeypatch.setenv("KALSHI_DRY_RUN", "false")
        monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
        monkeypatch.setenv("KILL_SWITCH", "false")

        from importlib import reload
        import trading_config
        reload(trading_config)
        trading_config.set_live_trading(True)
        trading_config.set_kill_switch(False)
        monkeypatch.setattr("live_position_manager.can_place_live_orders", lambda: True)
        monkeypatch.setattr("live_trader.can_place_live_orders", lambda: True)

        mock_client = MagicMock()
        mock_client.create_order.return_value = {
            "order": {"order_id": "hedge-1", "status": "submitted"},
        }

        fixtures = [{
            "fixture_key": "A|B|2026-06-26",
            "home": "A",
            "away": "B",
            "live": True,
            "goal_markets": {"outcomes": {"home_win": 0.55, "draw": 0.25, "away_win": 0.20}},
            "opportunities": [{
                "market_type": "home_win",
                "model_probability": 0.55,
                "kalshi_pct": 55.0,
                "fixture_key": "A|B|2026-06-26",
            }],
        }]
        from live_position_manager import exit_stale_live_positions

        monkeypatch.setattr(
            "live_position_manager.fetch_ticker_pricing",
            lambda _client, _ticker: None,
        )
        monkeypatch.setattr(
            "live_trader.fetch_ticker_pricing",
            lambda _client, _ticker: None,
        )
        closed = exit_stale_live_positions(fixtures, kalshi_client=mock_client)
        assert len(closed) == 1
        assert closed[0]["position"]["exit_reason"] == "model_reversal"
        assert closed[0]["position"]["exit_method"] == "opposite_side_hedge"
        assert closed[0]["position"]["hedge_side"] == "no"
        assert closed[0]["pnl"] == pytest.approx(-0.05, abs=0.01)
        mock_client.create_order.assert_called_once()
        call_kwargs = mock_client.create_order.call_args.kwargs
        assert call_kwargs["side"] == "no"
        assert call_kwargs["action"] == "buy"


class TestKalshiClientV2:
    def test_legacy_buy_yes_maps_to_bid(self):
        from kalshi_client import _legacy_order_to_v2

        book_side, price = _legacy_order_to_v2(
            side="yes", action="buy", yes_price=85, no_price=None,
        )
        assert book_side == "bid"
        assert price == "0.8500"

    def test_legacy_buy_no_maps_to_ask(self):
        from kalshi_client import _legacy_order_to_v2

        book_side, price = _legacy_order_to_v2(
            side="no", action="buy", yes_price=None, no_price=15,
        )
        assert book_side == "ask"
        assert price == "0.8500"

    def test_create_order_posts_v2_path(self, monkeypatch):
        from kalshi_client import KalshiClient

        captured: dict = {}

        def fake_request(self, method, path, *, params=None, json_body=None, auth=True):
            captured["method"] = method
            captured["path"] = path
            captured["json_body"] = json_body
            return {
                "order_id": "ord-v2-1",
                "client_order_id": json_body["client_order_id"],
                "fill_count": "0.00",
                "remaining_count": json_body["count"],
                "ts_ms": 123,
            }

        monkeypatch.setattr(KalshiClient, "_request", fake_request)
        monkeypatch.setenv("KALSHI_DRY_RUN", "false")

        from importlib import reload
        import trading_config
        reload(trading_config)

        cli = KalshiClient()
        resp = cli.create_order(
            "KXWCBTTS-TEST",
            side="no",
            action="buy",
            count=3,
            no_price=12,
            client_order_id="test-coid",
        )
        assert captured["method"] == "POST"
        assert captured["path"] == "/portfolio/events/orders"
        body = captured["json_body"]
        assert body["ticker"] == "KXWCBTTS-TEST"
        assert body["side"] == "ask"
        assert body["price"] == "0.8800"
        assert body["count"] == "3.00"
        assert body["time_in_force"] == "good_till_canceled"
        assert resp["order"]["order_id"] == "ord-v2-1"


class TestPaperTrader:
    def test_duplicate_prevention(self, tmp_path, monkeypatch):
        paper_path = tmp_path / "paper_trades.json"
        paper_path.write_text(json.dumps({"trades": [], "bankroll": 20}), encoding="utf-8")
        monkeypatch.setattr("paper_trader.PAPER_TRADES_PATH", paper_path)
        fkey = "Test|Team|2026-06-11"
        r1 = simulate_fill(
            ticker="T1", side="yes", count=1, entry_price_cents=40,
            model_probability=0.6, market_probability=0.4, edge=0.2,
            confidence=0.7, fixture_key=fkey, home="Test", away="Team",
            market_type="home_win",
        )
        assert r1["status"] == "filled"
        r2 = simulate_fill(
            ticker="T1", side="yes", count=1, entry_price_cents=40,
            model_probability=0.6, market_probability=0.4, edge=0.2,
            confidence=0.7, fixture_key=fkey, home="Test", away="Team",
            market_type="home_win",
        )
        assert r2["status"] == "skipped"
        assert is_duplicate_trade("T1", "yes", fkey)


class TestTradingConfig:
    def test_default_dry_run(self, monkeypatch):
        monkeypatch.delenv("KALSHI_DRY_RUN", raising=False)
        monkeypatch.delenv("ENABLE_LIVE_TRADING", raising=False)
        cfg = get_config()
        assert cfg.dry_run is True
        assert can_place_live_orders() is False


class TestTradeExecutor:
    def test_dry_run_blocks_live(self, monkeypatch, tmp_path):
        monkeypatch.setenv("KALSHI_DRY_RUN", "true")
        monkeypatch.setenv("ENABLE_LIVE_TRADING", "false")
        paper_path = tmp_path / "paper_trades.json"
        paper_path.write_text(json.dumps({"trades": [], "bankroll": 20}), encoding="utf-8")
        monkeypatch.setattr("paper_trader.PAPER_TRADES_PATH", paper_path)
        monkeypatch.setattr("trade_executor.can_place_live_orders", lambda: False)
        from importlib import reload
        import trading_config
        reload(trading_config)

        mock_client = MagicMock()
        mock_client.get_orderbook.return_value = {
            "orderbook": {"yes": [[40, 50]], "no": [[60, 50]]},
        }

        from trade_executor import execute_order
        with patch("trade_executor.evaluate_edge") as mock_edge:
            mock_edge.return_value = {
                "decision": "TRADE",
                "reason": "TRADE: test",
                "side": "yes",
                "edge": 0.15,
                "model_probability": 0.65,
                "market_probability": 0.50,
            }
            with patch("trade_executor.evaluate_risk") as mock_risk:
                mock_risk.return_value = {"approved": True, "reason": "ok"}
                result = execute_order(
                    ticker="TEST-UNIQUE",
                    side="yes",
                    count=1,
                    fixture_key="X|Y|2026-06-26",
                    home="X",
                    away="Y",
                    market_type="home_win",
                    model_probability=0.65,
                    confidence=0.7,
                    client=mock_client,
                )
        assert result.get("mode") == "paper"


class TestPositionOutcomes:
    def test_btts_no_lost_when_both_scored(self):
        from position_outcomes import evaluate_position_outcome

        pos = {"market_type": "btts_yes", "side": "no"}
        fixture = {"score_home": 1, "score_away": 1, "match_status": "2H", "match_final": False}
        outcome = evaluate_position_outcome(pos, fixture)
        assert outcome is not None
        assert outcome["yes_won"] is True
        assert outcome["won"] is False
        assert outcome["reason"] == "btts_both_scored"

    def test_btts_no_wins_at_final_one_nil(self):
        from position_outcomes import evaluate_position_outcome

        pos = {"market_type": "btts_yes", "side": "no"}
        fixture = {"score_home": 1, "score_away": 0, "match_status": "FT", "match_final": True}
        outcome = evaluate_position_outcome(pos, fixture)
        assert outcome is not None
        assert outcome["won"] is True


class TestLiveSettlement:
    def test_settle_btts_no_after_both_scored(self, tmp_path, monkeypatch):
        positions_path = tmp_path / "live_positions.json"
        position = {
            "position_id": "p-btts",
            "entry_order_id": "o1",
            "ticker": "KXWCBTTS-TEST",
            "side": "no",
            "count": 100,
            "entry_price_cents": 3.0,
            "cost": 3.0,
            "model_probability": 0.52,
            "model_probability_at_entry": 0.52,
            "fixture_key": "Algeria|Austria|2026-07-02",
            "home": "Algeria",
            "away": "Austria",
            "market_type": "btts_yes",
            "status": "open",
            "opened_at": "2026-06-28T00:00:00+00:00",
        }
        positions_path.write_text(json.dumps({"positions": [position]}), encoding="utf-8")
        monkeypatch.setattr("live_trader.LIVE_POSITIONS_PATH", positions_path)

        fixtures = [{
            "fixture_key": "Algeria|Austria|2026-07-02",
            "home": "Algeria",
            "away": "Austria",
            "score_home": 1,
            "score_away": 2,
            "match_status": "2H",
            "match_final": False,
            "live": True,
        }]
        from live_position_manager import settle_decided_live_positions

        closed = settle_decided_live_positions(fixtures)
        assert len(closed) == 1
        assert closed[0]["position"]["exit_method"] == "settlement"
        assert closed[0]["position"]["won"] is False
        assert closed[0]["pnl"] == pytest.approx(-3.0)

    def test_enrich_shows_loss_not_fake_gain(self, monkeypatch):
        from live_trader import enrich_live_positions

        pos = {
            "position_id": "p1",
            "ticker": "KXWCBTTS-TEST",
            "side": "no",
            "count": 265,
            "entry_price_cents": 3.0,
            "status": "open",
            "home": "Algeria",
            "away": "Austria",
            "fixture_key": "Algeria|Austria|2026-07-02",
            "market_type": "btts_yes",
        }
        fixtures = [{
            "fixture_key": "Algeria|Austria|2026-07-02",
            "home": "Algeria",
            "away": "Austria",
            "score_home": 1,
            "score_away": 1,
            "match_status": "2H",
        }]
        enriched = enrich_live_positions([pos], fixtures, client=None)
        row = enriched[0]
        assert row["outcome_decided"] is True
        assert row["unrealized_pnl"] == pytest.approx(-7.95, abs=0.01)


class TestEntryGuards:
    def test_blocks_over_35_when_four_goals(self):
        from entry_guards import block_live_entry

        reason = block_live_entry(
            market_type="over_3_5",
            score_home=2,
            score_away=2,
            match_final=False,
            model_yes=1.0,
            kalshi_yes=0.99,
            is_live=True,
        )
        assert reason == "Market outcome already decided from live score"

    def test_blocks_btts_when_both_scored(self):
        from entry_guards import block_live_entry

        reason = block_live_entry(
            market_type="btts_yes",
            score_home=1,
            score_away=1,
            is_live=True,
        )
        assert reason == "Market outcome already decided from live score"

    def test_blocks_stale_kalshi_mismatch(self):
        from entry_guards import block_live_entry

        reason = block_live_entry(
            market_type="over_3_5",
            score_home=1,
            score_away=0,
            model_yes=0.35,
            kalshi_yes=0.99,
            is_live=True,
        )
        assert "stale quote" in reason

    def test_live_scan_uses_goal_markets_at_22(self, monkeypatch):
        from trading_service import scan_opportunities

        monkeypatch.setattr("trading_service.fetch_orderbook_safe", lambda *a, **k: None)

        match = {
            "mn": 71,
            "home": "Algeria",
            "away": "Austria",
            "confidence": {"score": 0.7},
            "prediction": {
                "both_teams_score": 0.48,
                "over_3_5": 0.35,
            },
        }
        goal_mkts = build_goal_markets(1.5, 1.4, score_h=2, score_a=2, live=True)
        mapping = {
            "tickers": {"over_3_5": "KXWCTOTAL-TEST-4", "btts_yes": "KXWCBTTS-TEST"},
            "match_confidence": 0.95,
        }
        live_row = {"status": "2H", "score": {"home": 2, "away": 2}, "is_live": True}

        mock_client = MagicMock()
        mock_client.get_market.return_value = {
            "market": {
                "yes_bid_dollars": "0.9900",
                "yes_ask_dollars": "0.9900",
            },
        }

        opps = scan_opportunities(
            match=match,
            mapping=mapping,
            goal_mkts=goal_mkts,
            live_row=live_row,
            fkey="Algeria|Austria|2026-07-02",
            client=mock_client,
        )
        over = next(o for o in opps if o["market_type"] == "over_3_5")
        assert over["model_probability"] == pytest.approx(1.0)
        assert over["recommendation"] == "SKIP"
        assert "already decided" in over["reason"]

        btts = next(o for o in opps if o["market_type"] == "btts_yes")
        assert btts["model_probability"] == pytest.approx(1.0)
        assert btts["recommendation"] == "SKIP"
