"""Resolve API-Football / external team names to canonical ML pipeline names."""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

_ALIASES_PATH = Path(__file__).parent / "team_aliases.json"


def normalize_name(name: str) -> str:
    if not name:
        return ""
    text = unicodedata.normalize("NFKD", name)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"[^\w\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@lru_cache(maxsize=1)
def _alias_lookup() -> dict[str, str]:
    with open(_ALIASES_PATH, encoding="utf-8") as f:
        raw: dict[str, list[str]] = json.load(f)

    lookup: dict[str, str] = {}
    for canonical, aliases in raw.items():
        lookup[normalize_name(canonical)] = canonical
        for alias in aliases:
            lookup[normalize_name(alias)] = canonical
    return lookup


def resolve_team_name(name: str) -> str:
    """Map any known alias to the canonical name used in predictions.json."""
    if not name:
        return name
    key = normalize_name(name)
    return _alias_lookup().get(key, name.strip())


def teams_match(name_a: str, name_b: str) -> bool:
    return resolve_team_name(name_a) == resolve_team_name(name_b)


def find_ml_match(home: str, away: str, ml_data: list[dict]) -> dict | None:
    home_c = resolve_team_name(home)
    away_c = resolve_team_name(away)
    for match in ml_data:
        if match.get("home") == home_c and match.get("away") == away_c:
            return match
    for match in ml_data:
        if teams_match(match.get("home", ""), home) and teams_match(match.get("away", ""), away):
            return match
    return None
