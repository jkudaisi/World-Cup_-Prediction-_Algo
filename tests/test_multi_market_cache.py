"""Tests for multi-market cache helpers."""

from __future__ import annotations

import pytest

from multi_market_cache import qualification_probs_for_match


class TestQualificationProbs:
    def test_knockout_match_returns_qualification(self, monkeypatch):
        monkeypatch.setattr(
            "multi_market_cache.build_multi_market_bundle",
            lambda ml, **k: {
                "knockout": True,
                "kalshi_markets": {
                    "home_qualifies": 0.41,
                    "away_qualifies": 0.59,
                },
            },
        )
        out = qualification_probs_for_match(
            {
                "fixture_id": 1561329,
                "home": "South Africa",
                "away": "Canada",
                "group": "R32",
            },
        )
        assert out is not None
        assert out["home_qualifies"] == pytest.approx(0.41)
        assert out["away_qualifies"] == pytest.approx(0.59)

    def test_group_stage_returns_none(self):
        assert qualification_probs_for_match(
            {"home": "Mexico", "away": "South Africa", "group": "A"},
        ) is None
