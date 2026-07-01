"""Goalkeeper-aware penalty shootout skill from raw lineups and historical pens."""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config.pipeline_config import DATA_PROCESSED_PROVIDER, ROOT
from src.features.lineup_features import extract_goalkeeper_id
from training_store import atomic_write_json, utc_now_iso

log = logging.getLogger(__name__)

REGISTRY_PATH = DATA_PROCESSED_PROVIDER / "goalkeeper_registry.json"
RAW_LINEUPS = ROOT / "data" / "raw" / "providers" / "api_football" / "lineups"
RAW_STATS = ROOT / "data" / "raw" / "providers" / "api_football" / "statistics"
RAW_FIXTURES = ROOT / "data" / "raw" / "providers" / "api_football" / "fixtures" / "by_fixture"

HOME_PEN_LOGIT_ADV = math.log(0.52 / 0.48)  # slight home edge at equal GK skill
GK_BLEND_WEIGHT = 0.45  # how much GK module moves pen skill vs ML/default
PEN_SKILL_CLAMP = (0.32, 0.68)


@dataclass
class GoalkeeperRecord:
    player_id: int
    name: str = ""
    teams: list[int] = field(default_factory=list)
    pen_wins: int = 0
    pen_losses: int = 0
    saves_sum: float = 0.0
    saves_matches: int = 0
    last_seen: str | None = None

    def pen_rating(self) -> float:
        """Beta-smoothed pen shootout win rate."""
        return (self.pen_wins + 1) / (self.pen_wins + self.pen_losses + 2)

    def saves_rating(self) -> float:
        if self.saves_matches <= 0:
            return 0.5
        avg = self.saves_sum / self.saves_matches
        # ~5 saves/match is elite; ~2 is weak
        return max(0.0, min(1.0, 0.25 + avg / 6.0))

    def combined_rating(self) -> float:
        pen_n = self.pen_wins + self.pen_losses
        if pen_n >= 1:
            return 0.6 * self.pen_rating() + 0.4 * self.saves_rating()
        return 0.35 * self.pen_rating() + 0.65 * self.saves_rating()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GoalkeeperRegistry:
    def __init__(self, path: Path | None = None):
        self.path = path or REGISTRY_PATH
        self.goalkeepers: dict[str, GoalkeeperRecord] = {}
        self.team_latest_gk: dict[str, int] = {}
        self.team_pen_record: dict[str, dict[str, int]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for pid, rec in (data.get("goalkeepers") or {}).items():
            self.goalkeepers[str(pid)] = GoalkeeperRecord(**rec)
        self.team_latest_gk = {k: int(v) for k, v in (data.get("team_latest_gk") or {}).items()}
        self.team_pen_record = data.get("team_pen_record") or {}

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "provider": "api_football",
            "updated_at": utc_now_iso(),
            "goalkeepers": {k: v.to_dict() for k, v in self.goalkeepers.items()},
            "team_latest_gk": self.team_latest_gk,
            "team_pen_record": self.team_pen_record,
        }
        atomic_write_json(self.path, payload)
        return self.path

    def get_or_create(self, player_id: int, name: str = "") -> GoalkeeperRecord:
        key = str(player_id)
        if key not in self.goalkeepers:
            self.goalkeepers[key] = GoalkeeperRecord(player_id=player_id, name=name)
        elif name and not self.goalkeepers[key].name:
            self.goalkeepers[key].name = name
        return self.goalkeepers[key]

    def record_pen_result(
        self,
        home_gk_id: int | None,
        away_gk_id: int | None,
        *,
        home_won: bool,
        home_team: str,
        away_team: str,
        match_date: str | None = None,
    ) -> None:
        if home_gk_id:
            hgk = self.get_or_create(home_gk_id)
            if home_won:
                hgk.pen_wins += 1
            else:
                hgk.pen_losses += 1
            if match_date:
                hgk.last_seen = match_date[:10]
            self.team_latest_gk[home_team] = home_gk_id
        if away_gk_id:
            agk = self.get_or_create(away_gk_id)
            if home_won:
                agk.pen_losses += 1
            else:
                agk.pen_wins += 1
            if match_date:
                agk.last_seen = match_date[:10]
            self.team_latest_gk[away_team] = away_gk_id

        for team, won in ((home_team, home_won), (away_team, not home_won)):
            rec = self.team_pen_record.setdefault(team, {"wins": 0, "losses": 0})
            if won:
                rec["wins"] += 1
            else:
                rec["losses"] += 1

    def team_pen_rating(self, team: str) -> float:
        rec = self.team_pen_record.get(team) or {}
        w, l = int(rec.get("wins", 0)), int(rec.get("losses", 0))
        if w + l == 0:
            return 0.5
        return (w + 1) / (w + l + 2)

    def resolve_gk_for_team(self, team: str, fixture_id: int | None = None) -> tuple[int | None, dict[str, Any]]:
        meta: dict[str, Any] = {"source": "unknown"}
        if fixture_id is not None:
            gks = gk_ids_from_fixture(fixture_id)
            # Match by team name in fixture file if possible
            fix = _load_fixture(fixture_id)
            if fix:
                home = (fix.get("teams") or {}).get("home", {}).get("name", "")
                away = (fix.get("teams") or {}).get("away", {}).get("name", "")
                from team_names import resolve_team_name
                if resolve_team_name(home) == team:
                    meta = {"source": "lineup_fixture", "fixture_id": fixture_id, "side": "home"}
                    return gks.get("home_gk_id"), meta
                if resolve_team_name(away) == team:
                    meta = {"source": "lineup_fixture", "fixture_id": fixture_id, "side": "away"}
                    return gks.get("away_gk_id"), meta
        if team in self.team_latest_gk:
            meta = {"source": "team_latest_gk"}
            return self.team_latest_gk[team], meta
        meta = {"source": "team_pen_fallback"}
        return None, meta


_registry: GoalkeeperRegistry | None = None


def get_registry(reload: bool = False) -> GoalkeeperRegistry:
    global _registry
    if _registry is None or reload:
        _registry = GoalkeeperRegistry()
    return _registry


def _load_raw_wrapper(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _load_fixture(fixture_id: int) -> dict[str, Any] | None:
    w = _load_raw_wrapper(RAW_FIXTURES / f"fixture_{fixture_id}.json")
    if not w or not w.get("success"):
        return None
    resp = w.get("api_response")
    return resp if isinstance(resp, dict) else None


def gk_ids_from_lineup_api_response(api_response: list | dict | None) -> dict[str, int | None]:
    """Parse API-Football lineups list into home/away GK ids."""
    out: dict[str, int | None] = {"home_gk_id": None, "away_gk_id": None}
    if not isinstance(api_response, list):
        return out
    for i, block in enumerate(api_response[:2]):
        side = "home" if i == 0 else "away"
        lineup_side = {
            "startXI": block.get("startXI") or [],
            "formation": block.get("formation"),
        }
        pid = extract_goalkeeper_id(lineup_side)
        out[f"{side}_gk_id"] = pid
        name = ""
        for item in lineup_side["startXI"]:
            p = (item.get("player") or {})
            if p.get("pos") == "G":
                name = str(p.get("name") or "")
                break
        out[f"{side}_gk_name"] = name
        tid = (block.get("team") or {}).get("id")
        if tid is not None:
            out[f"{side}_team_id"] = int(tid)
    return out


def gk_ids_from_fixture(fixture_id: int) -> dict[str, Any]:
    w = _load_raw_wrapper(RAW_LINEUPS / f"fixture_{fixture_id}.json")
    if not w or not w.get("success"):
        return gk_ids_from_lineup_api_response(None)
    return gk_ids_from_lineup_api_response(w.get("api_response"))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _clamp(p: float) -> float:
    lo, hi = PEN_SKILL_CLAMP
    return max(lo, min(hi, p))


def rating_for_side(
    registry: GoalkeeperRegistry,
    team: str,
    gk_id: int | None,
) -> tuple[float, dict[str, Any]]:
    if gk_id is not None:
        rec = registry.goalkeepers.get(str(gk_id))
        if rec:
            return rec.combined_rating(), {
                "gk_id": gk_id,
                "name": rec.name,
                "pen_wins": rec.pen_wins,
                "pen_losses": rec.pen_losses,
                "saves_matches": rec.saves_matches,
                "rating": round(rec.combined_rating(), 4),
            }
    team_r = registry.team_pen_rating(team)
    return team_r, {"gk_id": None, "team_pen_rating": round(team_r, 4)}


def compute_pen_shootout_skills(
    home: str,
    away: str,
    *,
    fixture_id: int | None = None,
    registry: GoalkeeperRegistry | None = None,
) -> dict[str, Any]:
    """
    Home pen win probability from GK pen history + saves + team pen fallback.
    Returns home_pen_skill, away_pen_skill, and explanatory metadata.
    """
    reg = registry or get_registry()
    home_gk, home_meta = reg.resolve_gk_for_team(home, fixture_id)
    away_gk, away_meta = reg.resolve_gk_for_team(away, fixture_id)

    home_r, home_detail = rating_for_side(reg, home, home_gk)
    away_r, away_detail = rating_for_side(reg, away, away_gk)

    logit = HOME_PEN_LOGIT_ADV + 2.2 * (home_r - away_r)
    home_pen = _clamp(_sigmoid(logit))

    return {
        "home_pen_skill": round(home_pen, 4),
        "away_pen_skill": round(1.0 - home_pen, 4),
        "home_goalkeeper": {**home_detail, **home_meta},
        "away_goalkeeper": {**away_detail, **away_meta},
        "home_rating": round(home_r, 4),
        "away_rating": round(away_r, 4),
    }


def blend_pen_skill_with_goalkeepers(
    home: str,
    away: str,
    base_home_pen_skill: float,
    *,
    fixture_id: int | None = None,
    gk_blend_weight: float = GK_BLEND_WEIGHT,
) -> tuple[float, dict[str, Any]]:
    """Blend ML/Poisson pen skill with goalkeeper module."""
    gk = compute_pen_shootout_skills(home, away, fixture_id=fixture_id)
    gk_home = float(gk["home_pen_skill"])
    w = max(0.0, min(1.0, gk_blend_weight))
    blended = (1 - w) * base_home_pen_skill + w * gk_home
    blended = _clamp(blended)
    gk["base_home_pen_skill"] = round(base_home_pen_skill, 4)
    gk["blended_home_pen_skill"] = round(blended, 4)
    gk["gk_blend_weight"] = w
    return blended, gk


def _parse_stats_saves(api_response: list | None, team_id: int) -> int:
    if not isinstance(api_response, list):
        return 0
    for block in api_response:
        if (block.get("team") or {}).get("id") == team_id:
            for item in block.get("statistics") or []:
                if item.get("type") == "Goalkeeper Saves":
                    try:
                        return int(item.get("value") or 0)
                    except (TypeError, ValueError):
                        return 0
    return 0


def build_registry_from_sources(*, registry: GoalkeeperRegistry | None = None) -> GoalkeeperRegistry:
    """Rebuild GK registry from knockout outcomes + raw lineups/statistics."""
    from knockout_models import build_knockout_dataset
    from team_names import resolve_team_name

    reg = registry or GoalkeeperRegistry()
    reg.goalkeepers.clear()
    reg.team_latest_gk.clear()
    reg.team_pen_record.clear()

    # Pen shootouts from historical knockout outcomes
    for outcome in build_knockout_dataset(use_api=False):
        if not outcome.get("went_to_pens"):
            continue
        home = outcome["home"]
        away = outcome["away"]
        fid = outcome.get("fixture_id")
        home_gk = away_gk = None
        if fid:
            ids = gk_ids_from_fixture(int(fid))
            home_gk = ids.get("home_gk_id")
            away_gk = ids.get("away_gk_id")
        reg.record_pen_result(
            home_gk, away_gk,
            home_won=bool(outcome.get("home_won_pens")),
            home_team=home,
            away_team=away,
            match_date=outcome.get("date"),
        )

    # Also try API knockout fetch rows (may have fixture_id)
    try:
        from knockout_outcomes import fetch_knockout_outcomes_from_api
        for fx in fetch_knockout_outcomes_from_api():
            pass  # already in dataset via build_knockout_dataset
    except Exception:
        pass

    # Map team_id -> canonical name for latest GK tracking
    id_to_team: dict[int, str] = {}
    wc_ids_path = ROOT / "data" / "wc_team_ids.json"
    if wc_ids_path.exists():
        try:
            mapped = json.loads(wc_ids_path.read_text(encoding="utf-8")).get("mapped") or {}
            id_to_team = {int(v): k for k, v in mapped.items()}
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    # Lineups + saves from raw lake
    if RAW_LINEUPS.exists():
        for path in RAW_LINEUPS.glob("fixture_*.json"):
            w = _load_raw_wrapper(path)
            if not w or not w.get("success"):
                continue
            try:
                fid = int(path.stem.replace("fixture_", ""))
            except ValueError:
                continue
            ids = gk_ids_from_lineup_api_response(w.get("api_response"))
            fetched = w.get("fetched_at", "")[:10]
            for side, gk_key, name_key in (
                ("home", "home_gk_id", "home_gk_name"),
                ("away", "away_gk_id", "away_gk_name"),
            ):
                pid = ids.get(gk_key)
                if pid is None:
                    continue
                rec = reg.get_or_create(int(pid), str(ids.get(name_key) or ""))
                if fetched:
                    rec.last_seen = fetched
                tid = ids.get(f"{side}_team_id")
                if tid and tid not in rec.teams:
                    rec.teams.append(int(tid))
                tname = id_to_team.get(int(tid)) if tid else None
                if tname and fetched:
                    if not rec.last_seen or fetched >= rec.last_seen:
                        reg.team_latest_gk[tname] = int(pid)

            stats_w = _load_raw_wrapper(RAW_STATS / path.name)
            if stats_w and stats_w.get("success"):
                fix = _load_fixture(fid)
                if fix:
                    home_tid = (fix.get("teams") or {}).get("home", {}).get("id")
                    away_tid = (fix.get("teams") or {}).get("away", {}).get("id")
                    api_stats = stats_w.get("api_response")
                    if home_tid and ids.get("home_gk_id"):
                        saves = _parse_stats_saves(api_stats, int(home_tid))
                        hgk = reg.get_or_create(int(ids["home_gk_id"]))
                        hgk.saves_sum += saves
                        hgk.saves_matches += 1
                    if away_tid and ids.get("away_gk_id"):
                        saves = _parse_stats_saves(api_stats, int(away_tid))
                        agk = reg.get_or_create(int(ids["away_gk_id"]))
                        agk.saves_sum += saves
                        agk.saves_matches += 1

    # Team latest GK from most recent lineup per team name
    from wc2026_ml_pipeline import TEAM_STATS
    for team in TEAM_STATS:
        if team in reg.team_latest_gk:
            continue
        # fallback: team pen record only

    reg.save()
    log.info(
        "GK registry built: %s keepers, %s teams with pen history",
        len(reg.goalkeepers), len(reg.team_pen_record),
    )
    return reg
