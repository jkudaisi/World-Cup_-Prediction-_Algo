"""Tests for team name alias resolution."""

import pytest

from team_names import find_ml_match, resolve_team_name, teams_match


@pytest.mark.parametrize("api_name,canonical", [
    ("Congo DR", "DRC"),
    ("DR Congo", "DRC"),
    ("Cape Verde", "Cabo Verde"),
    ("Cape Verde Islands", "Cabo Verde"),
    ("Capo Verde", "Cabo Verde"),
    ("Côte d'Ivoire", "Ivory Coast"),
    ("Cote d'Ivoire", "Ivory Coast"),
    ("Turkey", "Turkiye"),
    ("Iran", "IR Iran"),
    ("United States", "USA"),
    ("Czech Republic", "Czechia"),
    ("Korea Republic", "South Korea"),
    ("Curaçao", "Curacao"),
])
def test_resolve_api_names(api_name, canonical):
    assert resolve_team_name(api_name) == canonical


def test_find_ml_match_portugal_drc():
    ml_data = [{"home": "Portugal", "away": "DRC", "models": {}}]
    match = find_ml_match("Portugal", "Congo DR", ml_data)
    assert match is not None
    assert match["away"] == "DRC"


def test_teams_match_case_insensitive():
    assert teams_match("CAPE VERDE", "Cabo Verde")
