"""Lineup continuity features from raw API-Football lineups."""
from __future__ import annotations

from typing import Any


def _player_id(entry: dict[str, Any]) -> int | None:
    player = entry.get("player") if isinstance(entry.get("player"), dict) else entry
    if not isinstance(player, dict):
        return None
    pid = player.get("id")
    try:
        return int(pid) if pid is not None else None
    except (TypeError, ValueError):
        return None


def extract_starter_ids(lineup_side: dict[str, Any] | None) -> list[int]:
    if not lineup_side:
        return []
    starters = lineup_side.get("startXI") or lineup_side.get("start_xi") or []
    ids: list[int] = []
    for item in starters:
        pid = _player_id(item)
        if pid is not None:
            ids.append(pid)
    return ids


def extract_goalkeeper_id(lineup_side: dict[str, Any] | None) -> int | None:
    if not lineup_side:
        return None
    for item in lineup_side.get("startXI") or []:
        player = (item.get("player") or {}) if isinstance(item, dict) else {}
        pos = str(player.get("pos") or "").upper()
        if pos == "G":
            return _player_id(item)
    starters = extract_starter_ids(lineup_side)
    return starters[0] if starters else None


def starting_xi_overlap(ids_a: list[int], ids_b: list[int]) -> float:
    if not ids_a or not ids_b:
        return 0.0
    set_a, set_b = set(ids_a), set(ids_b)
    return float(len(set_a & set_b))


def same_goalkeeper(gk_a: int | None, gk_b: int | None) -> bool | None:
    if gk_a is None or gk_b is None:
        return None
    return gk_a == gk_b


def formation_similarity(formation_a: str | None, formation_b: str | None) -> float:
    if not formation_a or not formation_b:
        return 0.9
    return 1.0 if formation_a.strip() == formation_b.strip() else 0.75


def lineup_continuity_context(
    current_lineups: dict[str, Any] | None,
    reference_lineups: dict[str, Any] | None,
    *,
    side: str = "home",
) -> dict[str, Any]:
    """Compare current vs reference lineup for one side (home/away)."""
    cur = (current_lineups or {}).get(side) or {}
    ref = (reference_lineups or {}).get(side) or {}
    cur_ids = extract_starter_ids(cur)
    ref_ids = extract_starter_ids(ref)
    overlap = starting_xi_overlap(cur_ids, ref_ids)
    return {
        "starting_xi_overlap": overlap,
        "returning_starter_count": overlap,
        "returning_minutes_share": overlap / 11.0 if overlap else 0.0,
        "same_goalkeeper": same_goalkeeper(
            extract_goalkeeper_id(cur),
            extract_goalkeeper_id(ref),
        ),
        "formation_similarity": formation_similarity(
            cur.get("formation"),
            ref.get("formation"),
        ),
        "lineup_similarity_weight": max(0.0, min(1.0, overlap / 11.0)) if overlap else 0.9,
    }


def merge_lineup_context_for_match(
    current_lineups: dict[str, Any] | None,
    reference_lineups: dict[str, Any] | None,
) -> dict[str, Any]:
    home = lineup_continuity_context(current_lineups, reference_lineups, side="home")
    away = lineup_continuity_context(current_lineups, reference_lineups, side="away")
    avg_overlap = (home["starting_xi_overlap"] + away["starting_xi_overlap"]) / 2.0
    gk_home = home.get("same_goalkeeper")
    gk_away = away.get("same_goalkeeper")
    if gk_home is True and gk_away is True:
        same_gk = True
    elif gk_home is False or gk_away is False:
        same_gk = False
    else:
        same_gk = None
    return {
        "has_lineups": bool(current_lineups),
        "starting_xi_overlap": avg_overlap,
        "same_gk": same_gk,
        "home_lineup": home,
        "away_lineup": away,
    }
