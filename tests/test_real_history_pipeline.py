"""Tests for production synthetic guards and leakage prevention."""
from __future__ import annotations

import os

import pandas as pd
import pytest

from src.data.guards import assert_no_synthetic_rows, mark_synthetic_rows
from src.features.feature_store import filter_matches_before
from src.features.relevance import recency_weight, world_cup_cycle_weight


class TestSyntheticGuards:
    def test_blocks_synthetic_in_production(self, monkeypatch):
        monkeypatch.delenv("WC_ALLOW_SYNTHETIC_TRAINING", raising=False)
        df = mark_synthetic_rows(pd.DataFrame({"goals_h": [1], "goals_a": [0]}), True)
        with pytest.raises(ValueError, match="synthetic"):
            assert_no_synthetic_rows(df)

    def test_allows_synthetic_when_env_set(self, monkeypatch):
        monkeypatch.setenv("WC_ALLOW_SYNTHETIC_TRAINING", "1")
        df = mark_synthetic_rows(pd.DataFrame({"goals_h": [1], "goals_a": [0]}), True)
        assert_no_synthetic_rows(df)  # no raise

    def test_real_rows_pass(self):
        df = pd.DataFrame({"goals_h": [1], "goals_a": [0], "is_synthetic": [False]})
        assert_no_synthetic_rows(df)


class TestLeakageGuards:
    def test_filter_matches_before_excludes_same_day_future(self):
        matches = [
            {"date": "2022-12-17", "id": 1},
            {"date": "2022-12-18", "id": 2},
            {"date": "2022-12-19", "id": 3},
        ]
        prior = filter_matches_before(matches, "2022-12-18")
        assert [m["id"] for m in prior] == [1]

    def test_recency_weight_decreases_with_age(self):
        assert recency_weight(100) > recency_weight(2000)

    def test_wc_cycle_weight_prefers_current_cycle(self):
        assert world_cup_cycle_weight(2024) > world_cup_cycle_weight(2010)
