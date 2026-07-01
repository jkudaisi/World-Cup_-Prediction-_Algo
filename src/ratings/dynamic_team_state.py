"""Dynamic per-team state updated after each completed match."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class TeamState:
    team_id: int
    team_name: str = ""
    overall_rating: float = 1500.0
    attack_rating: float = 0.0
    defense_rating: float = 0.0
    goalkeeper_rating: float = 0.0
    form_rating: float = 0.0
    coach_continuity: float = 1.0
    lineup_continuity: float = 1.0
    injury_impact: float = 0.0
    fatigue_score: float = 0.0
    travel_burden: float = 0.0
    data_quality_score: float = 1.0
    last_updated_match_id: int | None = None
    last_updated_at: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DynamicTeamStateStore:
    """In-memory store; persisted via historical_store / ratings pipeline."""

    def __init__(self) -> None:
        self._states: dict[int, TeamState] = {}

    def get(self, team_id: int) -> TeamState | None:
        return self._states.get(team_id)

    def get_or_create(self, team_id: int, team_name: str = "") -> TeamState:
        if team_id not in self._states:
            self._states[team_id] = TeamState(team_id=team_id, team_name=team_name)
        return self._states[team_id]

    def update_from_match(
        self,
        team_id: int,
        *,
        match_id: int,
        goals_for: int,
        goals_against: int,
        match_date: str | None = None,
        elo_delta: float = 0.0,
        **kwargs: Any,
    ) -> TeamState:
        state = self.get_or_create(team_id, kwargs.get("team_name", ""))
        state.overall_rating += elo_delta
        state.attack_rating = kwargs.get("attack_rating", state.attack_rating)
        state.defense_rating = kwargs.get("defense_rating", state.defense_rating)
        state.goalkeeper_rating = kwargs.get("goalkeeper_rating", state.goalkeeper_rating)
        state.form_rating = kwargs.get("form_rating", state.form_rating)
        state.coach_continuity = kwargs.get("coach_continuity", state.coach_continuity)
        state.lineup_continuity = kwargs.get("lineup_continuity", state.lineup_continuity)
        state.injury_impact = kwargs.get("injury_impact", state.injury_impact)
        state.fatigue_score = kwargs.get("fatigue_score", state.fatigue_score)
        state.travel_burden = kwargs.get("travel_burden", state.travel_burden)
        state.data_quality_score = kwargs.get("data_quality_score", state.data_quality_score)
        state.last_updated_match_id = match_id
        state.last_updated_at = match_date or datetime.utcnow().isoformat()
        return state

    def snapshot(self) -> dict[int, dict[str, Any]]:
        return {tid: s.to_dict() for tid, s in self._states.items()}

    def load_snapshot(self, data: dict[str, Any]) -> None:
        self._states.clear()
        for key, val in data.items():
            tid = int(key)
            self._states[tid] = TeamState(**{k: v for k, v in val.items() if k in TeamState.__dataclass_fields__})
