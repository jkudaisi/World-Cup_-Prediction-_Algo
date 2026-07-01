"""Tests for goalkeeper penalty shootout module."""
from __future__ import annotations

import json

import pytest

from src.features.goalkeeper_penalties import (
    GoalkeeperRegistry,
    blend_pen_skill_with_goalkeepers,
    compute_pen_shootout_skills,
    gk_ids_from_lineup_api_response,
)


SAMPLE_LINEUP = [
    {
        "team": {"id": 26, "name": "Argentina"},
        "startXI": [{"player": {"id": 1001, "name": "Martinez", "pos": "G"}}],
    },
    {
        "team": {"id": 2, "name": "France"},
        "startXI": [{"player": {"id": 1002, "name": "Lloris", "pos": "G"}}],
    },
]


class TestLineupParsing:
    def test_extract_gk_ids(self):
        ids = gk_ids_from_lineup_api_response(SAMPLE_LINEUP)
        assert ids["home_gk_id"] == 1001
        assert ids["away_gk_id"] == 1002


class TestPenSkill:
    def test_equal_gks_near_home_default(self, tmp_path, monkeypatch):
        reg = GoalkeeperRegistry(tmp_path / "gk.json")
        reg.record_pen_result(1001, 1002, home_won=True, home_team="Argentina", away_team="France")
        reg.record_pen_result(1001, 1002, home_won=False, home_team="Argentina", away_team="France")
        monkeypatch.setattr(
            "src.features.goalkeeper_penalties.get_registry", lambda reload=False: reg,
        )
        gk = compute_pen_shootout_skills("Argentina", "France", registry=reg)
        assert 0.45 < gk["home_pen_skill"] < 0.58

    def test_strong_home_gk_increases_home_pen(self, tmp_path, monkeypatch):
        reg = GoalkeeperRegistry(tmp_path / "gk.json")
        for _ in range(4):
            reg.record_pen_result(1001, 1002, home_won=True, home_team="Argentina", away_team="France")
        reg.record_pen_result(1001, 1002, home_won=False, home_team="Argentina", away_team="France")
        monkeypatch.setattr(
            "src.features.goalkeeper_penalties.get_registry", lambda reload=False: reg,
        )
        gk = compute_pen_shootout_skills("Argentina", "France", registry=reg)
        weak = compute_pen_shootout_skills("France", "Argentina", registry=reg)
        assert gk["home_pen_skill"] > weak["home_pen_skill"]

    def test_blend_moves_toward_gk(self, tmp_path, monkeypatch):
        reg = GoalkeeperRegistry(tmp_path / "gk.json")
        for _ in range(5):
            reg.record_pen_result(1001, 1002, home_won=True, home_team="Argentina", away_team="France")
        monkeypatch.setattr(
            "src.features.goalkeeper_penalties.get_registry", lambda reload=False: reg,
        )
        blended, meta = blend_pen_skill_with_goalkeepers("Argentina", "France", 0.52, gk_blend_weight=0.5)
        assert blended != 0.52
        assert "goalkeeper" in str(meta).lower() or "home_goalkeeper" in meta
