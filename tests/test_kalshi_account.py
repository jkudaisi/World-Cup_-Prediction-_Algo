"""Tests for Kalshi account balance fetching."""

from __future__ import annotations

import kalshi_account as ka


class TestKalshiAccountParsing:
    def test_parse_dollars_prefers_string(self):
        assert ka._parse_dollars("18.5162", 1851) == 18.5162

    def test_parse_dollars_falls_back_to_cents(self):
        assert ka._parse_dollars(None, 1851) == 18.51

    def test_position_exposure_sums_market_exposure(self):
        total, count = ka._position_exposure({
            "market_positions": [
                {"market_exposure_dollars": "1.50"},
                {"market_exposure_dollars": "2.25"},
            ],
        })
        assert total == 3.75
        assert count == 2


class TestKalshiAccountFetch:
    def test_fetch_account_summary_mock(self, monkeypatch):
        ka._CACHE["data"] = None
        ka._CACHE["ts"] = 0

        class FakeClient:
            def get_balance(self):
                return {
                    "balance_dollars": "18.5162",
                    "balance": 1851,
                    "portfolio_value": 106,
                }

            def get_positions(self):
                return {
                    "market_positions": [
                        {"market_exposure_dollars": "1.888000", "ticker": "TEST"},
                    ],
                }

        monkeypatch.setattr(ka, "credentials_configured", lambda: True)
        summary = ka.fetch_kalshi_account_summary(FakeClient(), force=True)
        assert summary is not None
        assert summary["available_cash"] == 18.52
        assert summary["in_positions"] == 1.89
        assert summary["account_total"] == 20.4
        assert summary["source"] == "kalshi"

    def test_resolve_bankroll_uses_kalshi_when_live(self, monkeypatch):
        monkeypatch.setattr(ka, "should_use_kalshi_balance", lambda: True)
        monkeypatch.setattr(
            ka,
            "fetch_kalshi_account_summary",
            lambda client=None, **kw: {"account_total": 20.4},
        )
        assert ka.resolve_bankroll() == 20.4
