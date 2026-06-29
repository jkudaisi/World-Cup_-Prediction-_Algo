"""Extract knockout outcome labels from API-Football fixtures and WC rows."""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

from feature_builder import build_features
from knockout_progression import is_knockout_round
from team_names import resolve_team_name

log = logging.getLogger(__name__)

KNOCKOUT_ROUND_RE = re.compile(
    r"round of (16|32)|quarter.?final|semi.?final|3rd place|third place|\bfinal\b",
    re.I,
)


def is_knockout_fixture(fixture: dict) -> bool:
    rnd = (fixture.get("league") or {}).get("round", "")
    if is_knockout_round(rnd):
        return True
    return bool(KNOCKOUT_ROUND_RE.search(rnd or ""))


def _score_pair(block: dict | None) -> tuple[int | None, int | None]:
    if not block:
        return None, None
    h, a = block.get("home"), block.get("away")
    if h is None or a is None:
        return None, None
    return int(h), int(a)


def parse_knockout_outcome(fixture: dict) -> dict[str, Any] | None:
    """
    Parse regulation / ET / penalty labels from an API-Football fixture dict.

    Returns None for group-stage or incomplete fixtures.
    """
    if not is_knockout_fixture(fixture):
        return None

    status = ((fixture.get("fixture") or {}).get("status") or {}).get("short", "FT")
    score = fixture.get("score") or {}
    goals = fixture.get("goals") or {}

    ft_h, ft_a = _score_pair(score.get("fulltime"))
    et_h, et_a = _score_pair(score.get("extratime"))
    pen_h, pen_a = _score_pair(score.get("penalty"))

    if ft_h is None or ft_a is None:
        gh, ga = goals.get("home"), goals.get("away")
        if status == "FT" and gh is not None and ga is not None:
            ft_h, ft_a = int(gh), int(ga)
        else:
            return None

    # API extratime is cumulative through ET; penalty block is shootout only
    if et_h is None and status in ("AET", "PEN"):
        gh, ga = goals.get("home"), goals.get("away")
        if gh is not None and ga is not None:
            et_h, et_a = int(gh), int(ga)

    draw_at_90 = ft_h == ft_a
    went_to_et = draw_at_90 and status in ("AET", "PEN")
    went_to_pens = pen_h is not None and pen_a is not None or status == "PEN"

    home_won_et = away_won_et = still_draw_et = False
    if went_to_et and et_h is not None and et_a is not None:
        home_won_et = et_h > et_a
        away_won_et = et_a > et_h
        still_draw_et = et_h == et_a

    home_won_pens = away_won_pens = False
    if went_to_pens and pen_h is not None and pen_a is not None:
        home_won_pens = pen_h > pen_a
        away_won_pens = pen_a > pen_h

    if status == "FT":
        home_qualifies = ft_h > ft_a
    elif status == "AET" and et_h is not None:
        home_qualifies = et_h > et_a
    elif went_to_pens and pen_h is not None:
        home_qualifies = home_won_pens
    else:
        gh, ga = goals.get("home"), goals.get("away")
        home_qualifies = int(gh or 0) > int(ga or 0) if gh is not None else ft_h > ft_a

    home = resolve_team_name(fixture["teams"]["home"]["name"])
    away = resolve_team_name(fixture["teams"]["away"]["name"])

    return {
        "fixture_id": fixture["fixture"]["id"],
        "home": home,
        "away": away,
        "round": (fixture.get("league") or {}).get("round", ""),
        "status": status,
        "reg_goals_h": ft_h,
        "reg_goals_a": ft_a,
        "et_goals_h": et_h,
        "et_goals_a": et_a,
        "pen_goals_h": pen_h,
        "pen_goals_a": pen_a,
        "draw_at_90": draw_at_90,
        "went_to_et": went_to_et,
        "went_to_pens": went_to_pens,
        "home_won_et": home_won_et,
        "away_won_et": away_won_et,
        "still_draw_after_et": still_draw_et,
        "home_won_pens": home_won_pens,
        "away_won_pens": away_won_pens,
        "home_qualifies": home_qualifies,
    }


def outcome_row_to_features(outcome: dict[str, Any]) -> dict[str, float]:
    """Team features for a knockout outcome row."""
    from wc2026_ml_pipeline import TEAM_STATS
    from feature_builder import build_features

    home = outcome["home"]
    away = outcome["away"]
    if home in TEAM_STATS and away in TEAM_STATS:
        return build_features(
            home,
            away,
            context={"knockout_stage": 1.0, "neutral_venue": 1.0},
        )

    # Historical WC teams may not be in the 2026 roster — use roster medians.
    med = {
        "elo": float(np.median([s["elo"] for s in TEAM_STATS.values()])),
        "xg": float(np.median([s["xg"] for s in TEAM_STATS.values()])),
        "xga": float(np.median([s["xga"] for s in TEAM_STATS.values()])),
        "form": float(np.median([s["form"] for s in TEAM_STATS.values()])),
    }
    patched = dict(TEAM_STATS)
    for team in (home, away):
        if team not in patched:
            patched[team] = dict(
                elo=med["elo"], rank=40, xg=med["xg"], xga=med["xga"],
                yc=1.8, rc=0.08, sq_val=200, wc_apps=4, titles=0,
                form=med["form"], press=11.0, dribble=4.5, aerial=50,
            )
    import wc2026_ml_pipeline as pipe
    original = pipe.TEAM_STATS
    try:
        pipe.TEAM_STATS = patched
        return build_features(
            home,
            away,
            context={"knockout_stage": 1.0, "neutral_venue": 1.0},
        )
    finally:
        pipe.TEAM_STATS = original


def fetch_knockout_outcomes_from_api(*, seasons: tuple[int, ...] = (2022, 2026)) -> list[dict[str, Any]]:
    """Pull completed knockout fixtures from API-Football season endpoints."""
    from apifootball_client import get_season_fixtures, WC_LEAGUE_ID

    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for season in seasons:
        try:
            fixtures = get_season_fixtures(league_id=WC_LEAGUE_ID, season=season)
        except Exception as exc:
            log.warning("Could not fetch season %s fixtures: %s", season, exc)
            continue
        for fx in fixtures:
            status = ((fx.get("fixture") or {}).get("status") or {}).get("short", "")
            if status not in ("FT", "AET", "PEN"):
                continue
            fid = fx["fixture"]["id"]
            if fid in seen:
                continue
            parsed = parse_knockout_outcome(fx)
            if parsed:
                parsed["season"] = season
                rows.append(parsed)
                seen.add(fid)
    return rows


def load_seed_knockout_outcomes() -> list[dict[str, Any]]:
    """Static historical knockout outcomes (2018 + 2022 WC) for bootstrap training."""
    from pathlib import Path
    import json

    path = Path(__file__).parent / "data" / "knockout_training_seed.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    return list(doc.get("matches") or [])
