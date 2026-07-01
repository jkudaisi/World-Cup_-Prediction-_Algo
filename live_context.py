"""Extract and apply live match context from API-Football bundles."""
from __future__ import annotations

from typing import Any

from feature_builder import build_lineup_features
from src.features.injury_features import injury_impact_score


def count_cards_from_events(
    events: list[dict] | None,
    home_team_id: int | None,
    away_team_id: int | None,
) -> dict[str, int]:
    """Count yellow/red cards from events (often ahead of aggregated stats)."""
    out = {"home_yellow": 0, "away_yellow": 0, "home_red": 0, "away_red": 0}
    if not events:
        return out
    for ev in events:
        if (ev.get("type") or "").lower() != "card":
            continue
        tid = (ev.get("team") or {}).get("id")
        detail = (ev.get("detail") or "").lower()
        if tid == home_team_id:
            side = "home"
        elif tid == away_team_id:
            side = "away"
        else:
            continue
        if "red" in detail:
            out[f"{side}_red"] += 1
        elif "yellow" in detail:
            out[f"{side}_yellow"] += 1
    return out


def count_substitutions(
    events: list[dict] | None,
    home_team_id: int | None,
    away_team_id: int | None,
) -> dict[str, int]:
    out = {"home": 0, "away": 0}
    if not events:
        return out
    for ev in events:
        if (ev.get("type") or "").lower() != "subst":
            continue
        tid = (ev.get("team") or {}).get("id")
        if tid == home_team_id:
            out["home"] += 1
        elif tid == away_team_id:
            out["away"] += 1
    return out


def merge_event_cards_into_stats(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Prefer event-sourced card counts when stats lag or are zero."""
    stats = snapshot.get("stats") or {}
    hs = dict(stats.get("home") or {})
    aws = dict(stats.get("away") or {})
    cards = count_cards_from_events(
        snapshot.get("events"),
        snapshot.get("home_team_id"),
        snapshot.get("away_team_id"),
    )
    if cards["home_red"] > int(hs.get("red_cards") or 0):
        hs["red_cards"] = cards["home_red"]
    if cards["away_red"] > int(aws.get("red_cards") or 0):
        aws["red_cards"] = cards["away_red"]
    if cards["home_yellow"] > int(hs.get("yellow_cards") or 0):
        hs["yellow_cards"] = cards["home_yellow"]
    if cards["away_yellow"] > int(aws.get("yellow_cards") or 0):
        aws["yellow_cards"] = cards["away_yellow"]
    out = dict(snapshot)
    out["stats"] = {"home": hs, "away": aws}
    return out


def load_prematch_injuries(fixture_id: int | None) -> list[dict[str, Any]] | None:
    if fixture_id is None:
        return None
    try:
        from src.data.api_football_backfill import load_raw
        raw = load_raw("injuries", int(fixture_id))
        if raw and raw.get("data"):
            return raw["data"] if isinstance(raw["data"], list) else []
    except Exception:
        pass
    return None


def lineup_completeness_score(lineups: dict | None, players: dict | None) -> float:
    feats = build_lineup_features(lineups, players)
    if not feats.get("lineup_available"):
        return 0.3
    score = 0.55
    if feats.get("formation_known"):
        score += 0.15
    if (feats.get("avg_rating_h", 0) + feats.get("avg_rating_a", 0)) > 12:
        score += 0.15
    return min(1.0, score)


def apply_live_lambda_adjustments(
    adj_lh: float,
    adj_la: float,
    snapshot: dict[str, Any],
    *,
    live_feats: dict[str, float] | None = None,
) -> tuple[float, float, dict[str, Any]]:
    """
    Nudge live lambdas using lineup quality, injuries, subs, and stat differentials.
    Returns (adj_lh, adj_la, context_meta).
    """
    lineups = snapshot.get("lineups")
    players = snapshot.get("players")
    injuries = snapshot.get("injuries")
    events = snapshot.get("events") or []
    home_id = snapshot.get("home_team_id")
    away_id = snapshot.get("away_team_id")
    minute = int(snapshot.get("minute") or 0)

    lineup_feats = build_lineup_features(lineups, players)
    injury_ctx = injury_impact_score(injuries, home_id, away_id)
    subs = count_substitutions(events, home_id, away_id)

    meta: dict[str, Any] = {
        "lineup_feats": lineup_feats,
        "injury_ctx": injury_ctx,
        "substitutions": subs,
    }

    # Player rating vs baseline — team outperforming lineup expectation
    rating_h = float(lineup_feats.get("avg_rating_h", 6.5))
    rating_a = float(lineup_feats.get("avg_rating_a", 6.5))
    if lineup_feats.get("lineup_available"):
        adj_lh *= 1.0 + max(-0.08, min(0.08, (rating_h - 6.8) * 0.04))
        adj_la *= 1.0 + max(-0.08, min(0.08, (rating_a - 6.8) * 0.04))

    # Injury impact (pre-match list still relevant early)
    if injury_ctx.get("has_injuries"):
        adj_lh *= 1.0 - 0.06 * float(injury_ctx.get("home_injury_impact", 0))
        adj_la *= 1.0 - 0.06 * float(injury_ctx.get("away_injury_impact", 0))

    # Live stat pressure from differentials (beyond momentum)
    if live_feats and minute >= 10:
        xg_diff = float(live_feats.get("live_xg_diff", 0))
        sot_diff = float(live_feats.get("live_sot_diff", 0))
        adj_lh *= 1.0 + max(-0.12, min(0.12, xg_diff * 0.08 + sot_diff * 0.015))
        adj_la *= 1.0 + max(-0.12, min(0.12, -xg_diff * 0.08 - sot_diff * 0.015))

    # Yellow accumulation → slightly more open play late
    cards = count_cards_from_events(events, home_id, away_id)
    if minute >= 60:
        total_yellow = cards["home_yellow"] + cards["away_yellow"]
        if total_yellow >= 4:
            bump = min(0.06, total_yellow * 0.01)
            adj_lh *= 1.0 + bump
            adj_la *= 1.0 + bump

    # Defensive subs when leading late
    score = snapshot.get("score") or {}
    sh = int(score.get("home") or 0)
    sa = int(score.get("away") or 0)
    if minute >= 70:
        if sh > sa and subs["home"] >= 3:
            adj_lh *= 0.94
            adj_la *= 1.04
        elif sa > sh and subs["away"] >= 3:
            adj_la *= 0.94
            adj_lh *= 1.04

    return max(0.1, min(5.0, adj_lh)), max(0.1, min(5.0, adj_la)), meta
