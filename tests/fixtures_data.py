"""Shared sample data for API tests."""

SAMPLE_FIXTURE = {
    "fixture": {
        "id": 12345,
        "date": "2026-06-17T18:00:00+00:00",
        "status": {"short": "1H", "elapsed": 32},
        "venue": {"name": "Test Stadium", "city": "Test City"},
    },
    "league": {"id": 1, "name": "World Cup", "round": "Group A - 1"},
    "teams": {
        "home": {"id": 10, "name": "Mexico"},
        "away": {"id": 20, "name": "South Africa"},
    },
    "goals": {"home": 1, "away": 0},
    "score": {
        "halftime": {"home": 1, "away": 0},
        "fulltime": {"home": None, "away": None},
    },
}

SAMPLE_STATS_RESPONSE = [
    {
        "team": {"id": 10, "name": "Mexico"},
        "statistics": [
            {"type": "Shots on Goal", "value": 4},
            {"type": "Corner Kicks", "value": 3},
            {"type": "Ball Possession", "value": "55%"},
            {"type": "Yellow Cards", "value": 1},
            {"type": "expected_goals", "value": "1.23"},
        ],
    },
    {
        "team": {"id": 20, "name": "South Africa"},
        "statistics": [
            {"type": "Shots on Goal", "value": 2},
            {"type": "Corner Kicks", "value": 1},
            {"type": "Ball Possession", "value": "45%"},
        ],
    },
]

SAMPLE_EVENTS_RESPONSE = [
    {
        "time": {"elapsed": 15, "extra": None},
        "team": {"id": 10, "name": "Mexico"},
        "player": {"id": 1, "name": "Player A"},
        "type": "Goal",
        "detail": "Normal Goal",
        "comments": None,
    },
    {
        "time": {"elapsed": 30, "extra": None},
        "team": {"id": 20, "name": "South Africa"},
        "player": {"id": 2, "name": "Player B"},
        "type": "Card",
        "detail": "Yellow Card",
        "comments": None,
    },
]

SAMPLE_LINEUPS_RESPONSE = [
    {
        "formation": "4-3-3",
        "startXI": [{"player": {"id": 1, "name": "GK", "pos": "G", "number": 1}}],
        "substitutes": [],
    },
    {
        "formation": "4-4-2",
        "startXI": [{"player": {"id": 2, "name": "GK", "pos": "G", "number": 1}}],
        "substitutes": [],
    },
]

MINIMAL_PREDICTIONS = {
    "ml_data": [{
        "mn": 1,
        "group": "A",
        "home": "Mexico",
        "away": "South Africa",
        "ens_h": 2,
        "ens_a": 1,
        "models": {"Poisson": {"rh": 1.8, "ra": 1.0, "gh": 2, "ga": 1}},
    }],
    "team_elo": {"Mexico": 1800},
    "stats": {"generated_at": "2026-06-17T12:00:00+00:00"},
    "training": None,
}
