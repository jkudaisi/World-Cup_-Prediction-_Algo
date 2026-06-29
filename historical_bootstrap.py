"""Pull historical national-team fixtures into the WC training dataset via API-Football."""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from apifootball_client import (
    APIFootballError,
    calls_remaining,
    get_fixture_stats,
    get_league_coverage,
    get_team_fixtures,
    get_team_id,
    get_wc_team_id_map,
)
from bootstrap_weights import compute_sample_weight
from feature_builder import calc_xg_proxy
from incremental_trainer import completed_match_to_training_row
from knockout_outcomes import is_knockout_fixture
from team_names import resolve_team_name
from training_store import (
    atomic_write_json,
    load_wc_matches,
    append_wc_matches,
    utc_now_iso,
)
from wc2026_ml_pipeline import TEAM_STATS

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
TEAM_IDS_PATH = DATA_DIR / "wc_team_ids.json"
COVERAGE_PATH = DATA_DIR / "bootstrap_coverage.json"
STATE_PATH = DATA_DIR / "bootstrap_state.json"

DEFAULT_DAILY_BUDGET = 4000
RESERVE_BUFFER = 500
API_SLEEP_SECONDS = 0.35
CACHE_MAX_AGE_DAYS = 7


def _parse_possession_pct(val: Any) -> float | None:
    if val is None:
        return None
    if isinstance(val, str):
        text = val.strip().replace("%", "")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    try:
        v = float(val)
        return v * 100.0 if 0.0 <= v <= 1.0 else v
    except (TypeError, ValueError):
        return None


def _load_json_if_fresh(path: Path, max_age_days: int = CACHE_MAX_AGE_DAYS) -> dict | None:
    if not path.exists():
        return None
    try:
        import json
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        generated = data.get("generated_at") or data.get("last_run")
        if not generated:
            return None
        ts = datetime.fromisoformat(str(generated).replace("Z", "+00:00"))
        if datetime.now(timezone.utc) - ts > timedelta(days=max_age_days):
            return None
        return data
    except (OSError, ValueError, TypeError) as exc:
        log.warning("Could not load cache %s: %s", path, exc)
        return None


def _default_state() -> dict[str, Any]:
    return {
        "calls_used_today": 0,
        "completed_team_ids": [],
        "team_season_progress": {},
        "total_rows_added": 0,
        "last_run": None,
        "needs_manual_mapping": [],
    }


def _load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return _default_state()
    try:
        import json
        with open(STATE_PATH, encoding="utf-8") as f:
            state = json.load(f)
        for key, val in _default_state().items():
            state.setdefault(key, val if not isinstance(val, list) else [])
        return state
    except (OSError, ValueError, TypeError) as exc:
        log.warning("bootstrap state corrupt, resetting: %s", exc)
        return _default_state()


def _save_state(state: dict[str, Any]) -> None:
    state["last_run"] = utc_now_iso()
    atomic_write_json(STATE_PATH, state)


class BootstrapRunner:
    def __init__(
        self,
        *,
        budget: int = DEFAULT_DAILY_BUDGET,
        dry_run: bool = False,
    ) -> None:
        self.budget = budget
        self.dry_run = dry_run
        self.state = _load_state()
        self.calls_used = 0
        self.rows_added = 0
        self.status = "complete"
        self.team_name_to_id: dict[str, int] = {}
        self.coverage_map: dict[str, dict[str, dict[str, dict[str, bool]]]] = {}

    def _can_call(self) -> bool:
        if self.calls_used >= self.budget:
            self.status = "budget_hit"
            return False
        if calls_remaining() < RESERVE_BUFFER:
            log.warning(
                "API budget buffer reached (%s remaining < %s) — stopping bootstrap",
                calls_remaining(),
                RESERVE_BUFFER,
            )
            self.status = "budget_hit"
            return False
        return True

    def _api_call(self, label: str, fn, *args, **kwargs):
        if not self._can_call():
            return None
        time.sleep(API_SLEEP_SECONDS)
        try:
            result = fn(*args, **kwargs)
            self.calls_used += 1
            self.state["calls_used_today"] = int(self.state.get("calls_used_today", 0)) + 1
            return result
        except APIFootballError as exc:
            if exc.status == 429:
                log.warning("Rate limited on %s — sleeping 60s and retrying once", label)
                time.sleep(60)
                if not self._can_call():
                    return None
                try:
                    result = fn(*args, **kwargs)
                    self.calls_used += 1
                    self.state["calls_used_today"] = int(self.state.get("calls_used_today", 0)) + 1
                    return result
                except APIFootballError as exc2:
                    log.warning("Retry failed for %s: %s", label, exc2)
                    return None
            log.warning("%s failed: %s", label, exc)
            return None
        except Exception as exc:
            log.warning("%s unexpected error: %s", label, exc)
            return None

    def resolve_team_ids(self) -> dict[str, Any]:
        cached = _load_json_if_fresh(TEAM_IDS_PATH)
        if cached and cached.get("mapped"):
            mapped_count = len(cached["mapped"])
            needs = list(cached.get("needs_manual_mapping") or [])
            if mapped_count >= len(TEAM_STATS) // 2:
                self.team_name_to_id = {k: int(v) for k, v in cached["mapped"].items()}
                self.state["needs_manual_mapping"] = needs
                log.info("Loaded %s team ids from cache", len(self.team_name_to_id))
                return cached
            log.warning(
                "Team id cache incomplete (%s/%s mapped) — rebuilding",
                mapped_count,
                len(TEAM_STATS),
            )

        mapped: dict[str, int] = {}
        needs_manual: list[str] = []

        wc_map = self._api_call("get_wc_team_id_map", get_wc_team_id_map) or {}
        for team_name in TEAM_STATS:
            tid = wc_map.get(team_name)
            if tid is None:
                tid = self._api_call(f"get_team_id({team_name})", get_team_id, team_name)
            if tid is None:
                needs_manual.append(team_name)
                log.warning("Could not resolve API team id for %s", team_name)
            else:
                mapped[team_name] = int(tid)

        payload = {
            "mapped": mapped,
            "needs_manual_mapping": needs_manual,
            "generated_at": utc_now_iso(),
        }
        if not self.dry_run:
            atomic_write_json(TEAM_IDS_PATH, payload)
        self.team_name_to_id = mapped
        self.state["needs_manual_mapping"] = needs_manual
        return payload

    def build_coverage_map(self) -> dict[str, Any]:
        cached = _load_json_if_fresh(COVERAGE_PATH)
        if cached and cached.get("teams"):
            self.coverage_map = cached["teams"]
            log.info("Loaded coverage map for %s teams from cache", len(self.coverage_map))
            return cached

        teams_cov: dict[str, dict[str, dict[str, dict[str, bool]]]] = {}
        for team_name, team_id in self.team_name_to_id.items():
            leagues = self._api_call(
                f"get_league_coverage({team_id})",
                get_league_coverage,
                team_id,
            ) or []
            team_entry: dict[str, dict[str, dict[str, bool]]] = {}
            for block in leagues:
                league = block.get("league") or {}
                lid = league.get("id")
                if lid is None:
                    continue
                seasons_out: dict[str, dict[str, bool]] = {}
                for season_block in block.get("seasons") or []:
                    year = season_block.get("year")
                    if year is None:
                        continue
                    cov = season_block.get("coverage") or {}
                    fixtures_cov = cov.get("fixtures") or {}
                    stats_cov = fixtures_cov.get("statistics") or {}
                    seasons_out[str(year)] = {
                        "stats": bool(stats_cov.get("fixtures")),
                        "events": bool((fixtures_cov.get("events") or False)),
                        "lineups": bool((fixtures_cov.get("lineups") or False)),
                    }
                if seasons_out:
                    team_entry[str(lid)] = seasons_out
            teams_cov[str(team_id)] = team_entry

        payload = {
            "teams": teams_cov,
            "generated_at": utc_now_iso(),
        }
        if not self.dry_run:
            atomic_write_json(COVERAGE_PATH, payload)
        self.coverage_map = teams_cov
        return payload

    def _has_fixture_stats(self, team_id: int, league_id: int, season: int) -> bool:
        team_cov = self.coverage_map.get(str(team_id)) or {}
        league_cov = team_cov.get(str(league_id)) or {}
        season_cov = league_cov.get(str(season)) or {}
        return bool(season_cov.get("stats"))

    def _fixture_row(
        self,
        fixture: dict,
        *,
        stats: dict | None,
        existing_ids: set[int],
    ) -> dict[str, Any] | None:
        fid = (fixture.get("fixture") or {}).get("id")
        if fid is None or int(fid) in existing_ids:
            return None

        goals = fixture.get("goals") or {}
        ft = (fixture.get("score") or {}).get("fulltime") or {}
        gh = goals.get("home") if goals.get("home") is not None else ft.get("home")
        ga = goals.get("away") if goals.get("away") is not None else ft.get("away")
        if gh is None or ga is None:
            return None

        home_raw = fixture.get("teams", {}).get("home", {}).get("name", "")
        away_raw = fixture.get("teams", {}).get("away", {}).get("name", "")
        home = resolve_team_name(home_raw)
        away = resolve_team_name(away_raw)
        if home not in TEAM_STATS or away not in TEAM_STATS:
            return None

        home_id = fixture.get("teams", {}).get("home", {}).get("id")
        away_id = fixture.get("teams", {}).get("away", {}).get("id")
        events: list[dict] = []
        row = completed_match_to_training_row(fixture, stats, events)
        if row is None:
            return None

        hs = (stats or {}).get("home") or {}
        aws = (stats or {}).get("away") or {}
        league = fixture.get("league") or {}
        league_id = int(league.get("id") or 0)
        league_name = str(league.get("name") or "")
        match_date = row.get("date") or (fixture.get("fixture", {}).get("date") or "")[:10]

        row.update({
            "home_shots": hs.get("shots_total"),
            "away_shots": aws.get("shots_total"),
            "home_shots_on_target": hs.get("shots_on_goal"),
            "away_shots_on_target": aws.get("shots_on_goal"),
            "home_possession": _parse_possession_pct(hs.get("ball_possession")),
            "away_possession": _parse_possession_pct(aws.get("ball_possession")),
            "home_xg_proxy": round(calc_xg_proxy(hs, events, home_id), 3),
            "away_xg_proxy": round(calc_xg_proxy(aws, events, away_id), 3),
            "sample_weight": compute_sample_weight(match_date, league_id),
            "is_national_team": True,
            "league_id": league_id,
            "league_name": league_name,
            "source": "historical_bootstrap",
        })
        existing_ids.add(int(fid))
        return row

    def _seasons_for_team(self, team_id: int) -> list[int]:
        """Calendar years to pull fixtures for, from coverage map or sensible default."""
        team_cov = self.coverage_map.get(str(team_id)) or {}
        years: set[int] = set()
        for league_seasons in team_cov.values():
            for year_str in league_seasons:
                try:
                    years.add(int(year_str))
                except (TypeError, ValueError):
                    continue
        if not years:
            years = set(range(2006, datetime.now(timezone.utc).year + 1))
        return sorted(years, reverse=True)

    def _fetch_team_fixtures(self, team_id: int) -> list[dict]:
        """Pull completed fixtures season-by-season (API requires season)."""
        seasons = self._seasons_for_team(team_id)
        progress = self.state.setdefault("team_season_progress", {})
        done_seasons = {int(s) for s in progress.get(str(team_id), [])}
        by_id: dict[int, dict] = {}

        for season in seasons:
            if season in done_seasons:
                continue
            if not self._can_call():
                break
            batch = self._api_call(
                f"get_team_fixtures({team_id},{season})",
                get_team_fixtures,
                team_id,
                season,
            ) or []
            for fixture in batch:
                fid = (fixture.get("fixture") or {}).get("id")
                if fid is not None:
                    by_id[int(fid)] = fixture
            done_seasons.add(season)
            progress[str(team_id)] = sorted(done_seasons)
            _save_state(self.state)

        if len(done_seasons) >= len(seasons):
            completed_ids = {int(x) for x in self.state.get("completed_team_ids") or []}
            completed_ids.add(team_id)
            self.state["completed_team_ids"] = sorted(completed_ids)

        return list(by_id.values())

    def pull_fixtures(self) -> None:
        if self.dry_run:
            log.info("Dry run — skipping fixture pull")
            return

        existing_ids = {
            int(m["fixture_id"])
            for m in load_wc_matches()
            if m.get("fixture_id") is not None
        }
        completed_ids = {int(x) for x in self.state.get("completed_team_ids") or []}

        for team_name, team_id in self.team_name_to_id.items():
            if team_id in completed_ids:
                continue
            if not self._can_call():
                break

            fixtures = self._fetch_team_fixtures(team_id)
            team_rows: list[dict[str, Any]] = []

            for fixture in fixtures:
                if not self._can_call():
                    break
                fid = (fixture.get("fixture") or {}).get("id")
                if fid is None or int(fid) in existing_ids:
                    continue

                league = fixture.get("league") or {}
                league_id = int(league.get("id") or 0)
                season = int(league.get("season") or 0)
                stats = None
                if self._has_fixture_stats(team_id, league_id, season):
                    home_id = fixture.get("teams", {}).get("home", {}).get("id")
                    away_id = fixture.get("teams", {}).get("away", {}).get("id")
                    stats = self._api_call(
                        f"get_fixture_stats({fid})",
                        get_fixture_stats,
                        int(fid),
                        home_id,
                        away_id,
                    )
                    if stats is None:
                        stats = {"home": {}, "away": {}}

                row = self._fixture_row(fixture, stats=stats, existing_ids=existing_ids)
                if row:
                    team_rows.append(row)

            if team_rows:
                append_wc_matches(team_rows)
                self.rows_added += len(team_rows)
                self.state["total_rows_added"] = int(self.state.get("total_rows_added", 0)) + len(team_rows)
                log.info("Team %s (%s): appended %s historical rows", team_name, team_id, len(team_rows))

            _save_state(self.state)

    def run(self) -> dict[str, Any]:
        try:
            self.resolve_team_ids()
            if self.status == "budget_hit":
                return self.summary()

            self.build_coverage_map()
            if self.status == "budget_hit":
                return self.summary()

            self.pull_fixtures()
        except Exception as exc:
            log.exception("Bootstrap failed: %s", exc)
            self.status = "error"
        finally:
            _save_state(self.state)

        return self.summary()

    def summary(self) -> dict[str, Any]:
        total_teams = len(TEAM_STATS)
        completed = len(self.state.get("completed_team_ids") or [])
        return {
            "status": self.status,
            "teams_completed": completed,
            "teams_remaining": max(0, total_teams - completed),
            "rows_added": self.rows_added,
            "calls_used": self.calls_used,
            "needs_manual_mapping": list(self.state.get("needs_manual_mapping") or []),
        }


def run_bootstrap(budget_override: int | None = None) -> dict[str, Any]:
    budget = budget_override if budget_override is not None else DEFAULT_DAILY_BUDGET
    runner = BootstrapRunner(budget=budget, dry_run=False)
    return runner.run()


def run_bootstrap_dry(budget_override: int | None = None) -> dict[str, Any]:
    budget = budget_override if budget_override is not None else DEFAULT_DAILY_BUDGET
    runner = BootstrapRunner(budget=budget, dry_run=True)
    return runner.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Bootstrap historical national-team training data")
    parser.add_argument("--budget", type=int, default=None, help="Override daily API call budget")
    parser.add_argument("--reset", action="store_true", help="Delete bootstrap_state.json and start fresh")
    parser.add_argument("--dry-run", action="store_true", help="Resolve team IDs and coverage only")
    args = parser.parse_args()

    if args.reset and STATE_PATH.exists():
        STATE_PATH.unlink()
        log.info("Deleted %s", STATE_PATH)

    if args.dry_run:
        result = run_bootstrap_dry(budget_override=args.budget)
    else:
        result = run_bootstrap(budget_override=args.budget)

    print(result)
