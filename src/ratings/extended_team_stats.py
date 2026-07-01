"""Team stat lookup with WC-pool defaults and chronological training state."""
from __future__ import annotations

import copy
from typing import Any

# Median national-team priors for opponents outside the WC 2026 pool.
DEFAULT_NATIONAL_STATS: dict[str, float | int] = {
    "elo": 1500.0,
    "rank": 80,
    "xg": 1.15,
    "xga": 1.25,
    "yc": 1.8,
    "rc": 0.08,
    "sq_val": 120,
    "wc_apps": 2,
    "titles": 0,
    "form": 1.2,
    "press": 10.0,
    "dribble": 4.2,
    "aerial": 50,
}

_EMA_ALPHA = 0.25
_ELO_K = 20.0
_runtime_registry: dict[str, dict[str, Any]] = {}


def _base_team_stats() -> dict[str, dict[str, Any]]:
    from wc2026_ml_pipeline import TEAM_STATS

    return TEAM_STATS


def is_wc_pool_team(team_name: str) -> bool:
    return team_name in _base_team_stats()


def default_stats_for_team(team_name: str, *, team_id: int | None = None) -> dict[str, Any]:
    stats = copy.deepcopy(DEFAULT_NATIONAL_STATS)
    stats["team_id"] = team_id
    return stats


def get_team_stats(team_name: str, *, team_id: int | None = None) -> dict[str, Any]:
    """Return WC-pool stats, runtime-registered stats, or national-team defaults."""
    base = _base_team_stats()
    if team_name in base:
        return base[team_name]
    if team_name in _runtime_registry:
        return _runtime_registry[team_name]
    stats = default_stats_for_team(team_name, team_id=team_id)
    _runtime_registry[team_name] = stats
    return stats


def reset_runtime_registry() -> None:
    _runtime_registry.clear()


def load_wc_pool_team_ids() -> set[int]:
    """Resolved API-Football IDs for the 48-team WC pool."""
    from pathlib import Path
    import json

    path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "manifests"
        / "providers"
        / "api_football"
        / "resolved_teams.json"
    )
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    ids: set[int] = set()
    for team in payload.get("teams") or []:
        tid = team.get("api_football_team_id")
        if tid is not None:
            ids.add(int(tid))
    return ids


def fixture_has_wc_pool_team(fixture: dict[str, Any], wc_ids: set[int] | None = None) -> bool:
    wc_ids = wc_ids if wc_ids is not None else load_wc_pool_team_ids()
    home_id = (fixture.get("teams") or {}).get("home", {}).get("id")
    away_id = (fixture.get("teams") or {}).get("away", {}).get("id")
    return (home_id is not None and int(home_id) in wc_ids) or (
        away_id is not None and int(away_id) in wc_ids
    )


def _elo_prob(ea: float, eb: float) -> float:
    return 1 / (1 + 10 ** ((eb - ea) / 400))


def _ema_update(current: float, observed: float, alpha: float = _EMA_ALPHA) -> float:
    return round((1.0 - alpha) * current + alpha * observed, 3)


def _match_form_points(goals_for: int, goals_against: int) -> float:
    if goals_for > goals_against:
        return 3.0
    if goals_for == goals_against:
        return 1.0
    return 0.0


def _match_score(goals_for: int, goals_against: int) -> float:
    if goals_for > goals_against:
        return 1.0
    if goals_for == goals_against:
        return 0.5
    return 0.0


class ChronologicalTeamStateTracker:
    """Rolling pre-match team state for historical training rows."""

    def __init__(self) -> None:
        self._states: dict[str, dict[str, Any]] = {}

    def snapshot(self, team_name: str, *, team_id: int | None = None) -> dict[str, Any]:
        if team_name not in self._states:
            self._states[team_name] = copy.deepcopy(get_team_stats(team_name, team_id=team_id))
        return copy.deepcopy(self._states[team_name])

    def apply_match(
        self,
        home: str,
        away: str,
        goals_h: int,
        goals_a: int,
        *,
        home_xg: float | None = None,
        away_xg: float | None = None,
    ) -> None:
        home_state = self.snapshot(home)
        away_state = self.snapshot(away)

        hxg = float(home_xg if home_xg is not None else goals_h)
        axg = float(away_xg if away_xg is not None else goals_a)

        home_state["form"] = _ema_update(home_state["form"], _match_form_points(goals_h, goals_a))
        away_state["form"] = _ema_update(away_state["form"], _match_form_points(goals_a, goals_h))
        home_state["xg"] = _ema_update(home_state["xg"], hxg)
        home_state["xga"] = _ema_update(home_state["xga"], float(goals_a))
        away_state["xg"] = _ema_update(away_state["xg"], axg)
        away_state["xga"] = _ema_update(away_state["xga"], float(goals_h))

        exp_home = _elo_prob(home_state["elo"], away_state["elo"])
        score_home = _match_score(goals_h, goals_a)
        home_state["elo"] = round(home_state["elo"] + _ELO_K * (score_home - exp_home), 1)
        away_state["elo"] = round(away_state["elo"] + _ELO_K * ((1.0 - score_home) - (1.0 - exp_home)), 1)

        self._states[home] = home_state
        self._states[away] = away_state
