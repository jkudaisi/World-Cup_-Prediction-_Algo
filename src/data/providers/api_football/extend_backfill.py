"""Extend API-Football raw backfill with earlier seasons and new fixture details."""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from src.data.providers.api_football.backfill import (
    APIFootballBackfiller,
    load_backfill_config,
    load_teams_config,
    resolve_team_ids,
)
from src.data.providers.api_football.client import APIFootballClient, APIFootballClientError
from src.data.providers.api_football.manifest import load_resume_state, save_resume_state
from src.data.providers.api_football.paths import COMPLETED_STATUSES, RAW_ROOT
from src.data.providers.api_football.raw_store import APIFootballRawStore
from training_store import atomic_write_json, utc_now_iso

log = logging.getLogger(__name__)

EXTEND_MANIFEST = (
    Path(__file__).resolve().parents[4]
    / "data"
    / "manifests"
    / "providers"
    / "api_football"
    / "extend_backfill_manifest.json"
)


def _load_team_fixtures_from_raw(raw_store: APIFootballRawStore, team_id: int) -> dict[int, dict[str, Any]]:
    path = raw_store.root / "fixtures" / "by_team" / f"team_{team_id}.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    by_id: dict[int, dict[str, Any]] = {}
    for fixture in payload.get("api_response") or []:
        fid = (fixture.get("fixture") or {}).get("id")
        if fid is not None:
            by_id[int(fid)] = fixture
    return by_id


def _years_in_fixtures(fixtures: dict[int, dict[str, Any]]) -> set[int]:
    years: set[int] = set()
    for fx in fixtures.values():
        fx_date = str((fx.get("fixture") or {}).get("date") or "")[:4]
        if fx_date.isdigit():
            years.add(int(fx_date))
    return years


class APIFootballBackfillExtender:
    """Fetch additional seasons for WC teams and backfill new fixture detail layers."""

    def __init__(
        self,
        *,
        client: APIFootballClient | None = None,
        raw_store: APIFootballRawStore | None = None,
        config: dict[str, Any] | None = None,
        dry_run: bool = False,
    ):
        self.config = config or load_backfill_config()
        self.client = client or APIFootballClient(
            rate_limit_sleep_seconds=float((self.config.get("rate_limit") or {}).get("sleep_seconds", 0.25)),
            max_retries=int((self.config.get("rate_limit") or {}).get("max_retries", 3)),
        )
        self.raw_store = raw_store or APIFootballRawStore()
        self.backfiller = APIFootballBackfiller(
            client=self.client,
            raw_store=self.raw_store,
            config=self.config,
            dry_run=dry_run,
        )
        self.dry_run = dry_run
        self.resume = load_resume_state()
        self.summary: dict[str, Any] = {
            "provider": "api_football",
            "mode": "extend_backfill",
            "started_at": utc_now_iso(),
            "teams_processed": 0,
            "seasons_fetched": 0,
            "fixtures_discovered": 0,
            "fixtures_new": 0,
            "fixtures_backfilled": 0,
            "endpoint_requests": {},
        }

    def _season_years(self, from_year: int, to_year: int) -> list[int]:
        return list(range(from_year, to_year + 1))

    def _fetch_team_seasons(
        self,
        team_id: int,
        years: list[int],
        existing: dict[int, dict[str, Any]],
    ) -> dict[int, dict[str, Any]]:
        merged = dict(existing)
        progress = self.resume.setdefault("extended_team_years", {})
        done_years = {int(y) for y in progress.get(str(team_id), [])}

        for year in years:
            if year in done_years:
                continue
            try:
                batch = self.client.get_fixtures(team=team_id, season=year) or []
            except APIFootballClientError as exc:
                log.warning("fixtures team=%s year=%s failed: %s", team_id, year, exc)
                continue
            if not isinstance(batch, list):
                continue
            for fx in batch:
                fid = (fx.get("fixture") or {}).get("id")
                if fid is not None:
                    merged[int(fid)] = fx
            done_years.add(year)
            self.summary["seasons_fetched"] += 1
            progress[str(team_id)] = sorted(done_years)
            if not self.dry_run:
                save_resume_state(self.resume)

        return merged

    def extend_team_fixtures(self, from_year: int, to_year: int) -> set[int]:
        teams, _ = resolve_team_ids(load_teams_config(), self.client, dry_run=self.dry_run)
        completed_ids = {int(x) for x in self.resume.get("completed_fixture_ids") or []}
        all_before = set(completed_ids)

        for team in teams:
            tid = team.get("api_football_team_id")
            if tid is None:
                continue
            tid = int(tid)
            existing = _load_team_fixtures_from_raw(self.raw_store, tid)
            years_needed = self._season_years(from_year, to_year)
            if not years_needed:
                continue
            merged = self._fetch_team_seasons(tid, years_needed, existing)
            self.summary["teams_processed"] += 1
            self.summary["fixtures_discovered"] = max(
                self.summary["fixtures_discovered"],
                len(merged),
            )

            if self.dry_run:
                continue

            self.raw_store.save_endpoint_response(
                "fixtures_by_team",
                str(tid),
                endpoint="/fixtures",
                params={"team": tid, "from_year": from_year, "to_year": to_year, "extend": True},
                api_response=list(merged.values()),
                success=True,
            )

        all_after: set[int] = set()
        team_dir = self.raw_store.root / "fixtures" / "by_team"
        for path in team_dir.glob("team_*.json"):
            for fx in _load_team_fixtures_from_raw(self.raw_store, int(path.stem.split("_", 1)[1])).values():
                fid = (fx.get("fixture") or {}).get("id")
                if fid is not None:
                    all_after.add(int(fid))

        new_ids = all_after - all_before
        self.summary["fixtures_new"] = len(new_ids)
        return new_ids

    def backfill_fixture_ids(self, fixture_ids: set[int]) -> int:
        if self.dry_run or not fixture_ids:
            return 0

        completed_ids = {int(x) for x in self.resume.get("completed_fixture_ids") or []}
        failed_ids = {int(x) for x in self.resume.get("failed_fixture_ids") or []}
        fixture_map = self.backfiller._load_fixtures_from_raw_store(set(fixture_ids))

        done = 0
        for fid in sorted(fixture_ids):
            fx = fixture_map.get(fid)
            if fx is None:
                continue
            status = (fx.get("fixture") or {}).get("status", {}).get("short", "")
            if status not in COMPLETED_STATUSES and status not in ("NS", "TBD", "PST"):
                continue
            try:
                self.backfiller.backfill_fixture(fid, fx)
                completed_ids.add(fid)
                self.resume["completed_fixture_ids"] = sorted(completed_ids)
                save_resume_state(self.resume)
                done += 1
            except Exception as exc:
                log.exception("fixture %s extend backfill failed: %s", fid, exc)
                failed_ids.add(fid)
                self.resume["failed_fixture_ids"] = sorted(failed_ids)
                save_resume_state(self.resume)

        self.summary["fixtures_backfilled"] = done
        return done

    def run(self, *, from_year: int, to_year: int) -> dict[str, Any]:
        new_ids = self.extend_team_fixtures(from_year, to_year)
        self.backfill_fixture_ids(new_ids)
        self.summary["endpoint_requests"] = self.client.endpoint_stats()
        self.summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        if not self.dry_run:
            EXTEND_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(EXTEND_MANIFEST, self.summary)
        return self.summary

    def print_summary(self) -> None:
        s = self.summary
        print(f"Provider: {s.get('provider')}")
        print(f"Mode: {s.get('mode')}")
        print(f"Teams processed: {s.get('teams_processed')}")
        print(f"Seasons fetched: {s.get('seasons_fetched')}")
        print(f"Fixtures discovered: {s.get('fixtures_discovered')}")
        print(f"New fixtures: {s.get('fixtures_new')}")
        print(f"Fixtures backfilled: {s.get('fixtures_backfilled')}")
        print(f"Endpoint requests: {s.get('endpoint_requests')}")
        print(f"Manifest: {EXTEND_MANIFEST}")
