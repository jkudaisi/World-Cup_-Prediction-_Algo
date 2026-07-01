"""Resumable API-Football historical backfill engine."""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.data.providers.api_football.client import APIFootballClient, APIFootballClientError
from src.data.providers.api_football.coverage import APIFootballCoverageDiscoverer
from src.data.providers.api_football.manifest import (
    append_failed_request,
    build_backfill_manifest,
    load_resume_state,
    save_resume_state,
    write_manifest,
)
from src.data.providers.api_football.models import compute_endpoint_coverage_score
from src.data.providers.api_football.paths import (
    COMPLETED_STATUSES,
    CONFIG_BACKFILL,
    CONFIG_TEAMS,
    MANIFEST_ROOT,
    RESOLVED_TEAMS,
)
from src.data.providers.api_football.raw_store import APIFootballRawStore
from training_store import atomic_write_json, utc_now_iso

log = logging.getLogger(__name__)

FIXTURE_DETAIL_ENDPOINTS = (
    ("events", "/fixtures/events", "get_fixture_events"),
    ("statistics", "/fixtures/statistics", "get_fixture_statistics"),
    ("players", "/fixtures/players", "get_fixture_players"),
    ("lineups", "/fixtures/lineups", "get_fixture_lineups"),
    ("injuries", "/injuries", "get_injuries"),
)


def load_backfill_config(path: Path | None = None) -> dict[str, Any]:
    p = path or CONFIG_BACKFILL
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_teams_config(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or CONFIG_TEAMS
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return list(data.get("teams") or [])


def resolve_team_ids(
    teams: list[dict[str, Any]],
    client: APIFootballClient,
    *,
    dry_run: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve API-Football team IDs; write resolved_teams.json (never overwrites yaml)."""
    resolved: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    # Merge known IDs from existing wc_team_ids.json
    legacy_map: dict[str, int] = {}
    legacy_path = Path(__file__).resolve().parents[4] / "data" / "wc_team_ids.json"
    if legacy_path.exists():
        try:
            legacy_map = json.loads(legacy_path.read_text(encoding="utf-8")).get("mapped") or {}
        except (json.JSONDecodeError, OSError):
            pass

    for team in teams:
        name = team.get("name", "")
        tid = team.get("api_football_team_id")
        if tid is None and name in legacy_map:
            tid = legacy_map[name]
        if tid is None and not dry_run:
            for alias in [name] + list(team.get("aliases") or []):
                try:
                    hits = client.get_teams(search=alias) or []
                    for hit in hits:
                        t = hit.get("team") or {}
                        if t.get("national") is True:
                            tid = t.get("id")
                            break
                    if tid:
                        break
                except APIFootballClientError:
                    continue
        entry = {**team, "api_football_team_id": tid, "resolved_at": utc_now_iso()}
        if tid:
            resolved.append(entry)
        else:
            unresolved.append(entry)

    if not dry_run:
        MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
        atomic_write_json(RESOLVED_TEAMS, {
            "provider": "api_football",
            "generated_at": utc_now_iso(),
            "teams": resolved,
            "unresolved": unresolved,
        })
    return resolved, unresolved


class APIFootballBackfiller:
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
            force_refresh=bool((self.config.get("cache") or {}).get("force_refresh")),
            rate_limit_sleep_seconds=float((self.config.get("rate_limit") or {}).get("sleep_seconds", 0.25)),
            max_retries=int((self.config.get("rate_limit") or {}).get("max_retries", 3)),
        )
        self.raw_store = raw_store or APIFootballRawStore()
        self.dry_run = dry_run
        self.discoverer = APIFootballCoverageDiscoverer(self.client, self.raw_store, dry_run=dry_run)
        self.resume = load_resume_state()
        self.manifest = build_backfill_manifest()
        self._missing_summary: dict[str, int] = {}

    def _date_range(self, date_from: str | None, date_to: str | None) -> tuple[str, str]:
        dr = self.config.get("date_range") or {}
        d_from = date_from or dr.get("from") or "2000-01-01"
        d_to = date_to or dr.get("to") or "today"
        if d_to == "today":
            d_to = date.today().isoformat()
        return d_from, d_to

    def _year_range(self, d_from: str, d_to: str) -> tuple[int, int]:
        return int(d_from[:4]), int(d_to[:4])

    def _load_fixtures_from_raw_store(self, fixture_ids: set[int]) -> dict[int, dict]:
        """Resolve fixture payloads from cached team lists or per-fixture files."""
        by_id: dict[int, dict] = {}
        if not fixture_ids:
            return by_id

        team_dir = self.raw_store.root / "fixtures" / "by_team"
        for path in team_dir.glob("team_*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            batch = data.get("api_response") or []
            if not isinstance(batch, list):
                continue
            for fx in batch:
                fid = (fx.get("fixture") or {}).get("id")
                if fid is None:
                    continue
                fid = int(fid)
                if fid in fixture_ids and fid not in by_id:
                    by_id[fid] = fx
            if len(by_id) == len(fixture_ids):
                break

        for fid in fixture_ids - set(by_id):
            wrapper = self.raw_store.load_endpoint("fixtures_by_fixture", str(fid))
            if not wrapper:
                continue
            api_response = wrapper.get("api_response")
            if isinstance(api_response, dict):
                by_id[fid] = api_response
            elif isinstance(api_response, list) and api_response:
                by_id[fid] = api_response[0]
        return by_id

    def discover_fixtures_for_team(self, team_id: int, from_year: int, to_year: int) -> dict[int, dict]:
        by_id: dict[int, dict] = {}
        for year in range(from_year, to_year + 1):
            try:
                batch = self.client.get_fixtures(team=team_id, season=year) or []
            except APIFootballClientError as exc:
                log.warning("fixtures team=%s year=%s: %s", team_id, year, exc)
                continue
            if not isinstance(batch, list):
                continue
            for fx in batch:
                fid = (fx.get("fixture") or {}).get("id")
                if fid is not None:
                    by_id[int(fid)] = fx
        return by_id

    def backfill_fixture(self, fixture_id: int, fixture: dict[str, Any]) -> dict[str, bool]:
        """Fetch all detail endpoints; never discard fixture on missing data."""
        results: dict[str, bool] = {}
        status = (fixture.get("fixture") or {}).get("status", {}).get("short", "")

        if self.dry_run:
            return {k: False for k, _, _ in FIXTURE_DETAIL_ENDPOINTS}

        # Always store core fixture
        self.raw_store.save_endpoint_response(
            "fixtures_by_fixture",
            str(fixture_id),
            endpoint="/fixtures",
            params={"fixture": fixture_id},
            api_response=fixture,
            success=True,
        )

        for kind, endpoint, method_name in FIXTURE_DETAIL_ENDPOINTS:
            params = {"fixture": fixture_id}
            try:
                method = getattr(self.client, method_name)
                if method_name == "get_injuries":
                    data = method(fixture=fixture_id)
                else:
                    data = method(fixture_id)
                ok = data is not None and (not isinstance(data, list) or len(data) > 0)
                results[kind] = ok
                if ok:
                    self.raw_store.save_endpoint_response(
                        kind, str(fixture_id),
                        endpoint=endpoint, params=params,
                        api_response=data, success=True,
                    )
                else:
                    self._record_missing(fixture_id, endpoint, "no data", status)
                    self.raw_store.save_endpoint_response(
                        kind, str(fixture_id),
                        endpoint=endpoint, params=params,
                        api_response=data, success=False, error="endpoint returned no data",
                    )
            except APIFootballClientError as exc:
                results[kind] = False
                self._record_missing(fixture_id, endpoint, str(exc), status)
                append_failed_request({"fixture_id": fixture_id, "endpoint": endpoint, "error": str(exc)})
                self.raw_store.save_endpoint_response(
                    kind, str(fixture_id),
                    endpoint=endpoint, params=params,
                    api_response=None, success=False, error=str(exc),
                )
        return results

    def _record_missing(self, fixture_id: int, endpoint: str, reason: str, status: str) -> None:
        key = endpoint.replace("/", "_")
        self._missing_summary[key] = self._missing_summary.get(key, 0) + 1
        self.raw_store.append_missing_endpoint_log({
            "fixture_id": fixture_id,
            "endpoint": endpoint,
            "reason": reason,
            "status": status,
        })

    def run(
        self,
        *,
        date_from: str | None = None,
        date_to: str | None = None,
        teams: list[dict[str, Any]] | None = None,
        run_discovery_first: bool | None = None,
        sample_fixtures: int = 5,
        resume: bool = True,
    ) -> dict[str, Any]:
        d_from, d_to = self._date_range(date_from, date_to)
        from_year, to_year = self._year_range(d_from, d_to)
        teams = teams or load_teams_config()
        self.manifest["date_range"] = {"from": d_from, "to": d_to}
        self.manifest["teams_requested"] = len(teams)

        resolved, unresolved = resolve_team_ids(teams, self.client, dry_run=self.dry_run)
        self.manifest["teams_resolved"] = len(resolved)
        if unresolved and not self.dry_run:
            write_manifest("unresolved_teams.json", unresolved)

        cov_cfg = self.config.get("coverage") or {}
        if run_discovery_first is None:
            run_discovery_first = bool(cov_cfg.get("run_discovery_first", True))

        all_fixtures: dict[int, dict] = {}
        completed_ids = set(self.resume.get("completed_fixture_ids") or []) if resume else set()
        failed_ids = set(self.resume.get("failed_fixture_ids") or [])

        team_reports: dict[str, Any] = {}
        for team in resolved:
            tid = team.get("api_football_team_id")
            if tid is None:
                continue
            tid = int(tid)
            if resume and tid in (self.resume.get("completed_team_ids") or []):
                continue

            if run_discovery_first:
                report = self.discoverer.discover_team_coverage(
                    tid, from_year, to_year,
                    team_name=team.get("name"),
                    sample_fixtures=sample_fixtures,
                )
                team_reports[str(tid)] = report.to_dict()

            fixtures = self.discover_fixtures_for_team(tid, from_year, to_year)
            # Filter by date range
            for fid, fx in fixtures.items():
                fx_date = str((fx.get("fixture") or {}).get("date") or "")[:10]
                if fx_date and (fx_date < d_from[:10] or fx_date > d_to[:10]):
                    continue
                all_fixtures[fid] = fx

            if not self.dry_run:
                # Append team fixture list
                self.raw_store.save_endpoint_response(
                    "fixtures_by_team",
                    str(tid),
                    endpoint="/fixtures",
                    params={"team": tid, "from_year": from_year, "to_year": to_year},
                    api_response=list(fixtures.values()),
                    success=True,
                )
                self.resume.setdefault("completed_team_ids", []).append(tid)
                self.resume["last_completed_team_id"] = tid
                self.resume["last_successful_checkpoint_at"] = utc_now_iso()
                save_resume_state(self.resume)

        if resume and not all_fixtures:
            saved_pending = {
                int(fid)
                for fid in (self.resume.get("pending_fixture_ids") or [])
                if int(fid) not in completed_ids
            }
            if saved_pending:
                all_fixtures = self._load_fixtures_from_raw_store(saved_pending)
                missing = saved_pending - set(all_fixtures)
                for fid in sorted(missing):
                    try:
                        batch = self.client.get_fixtures(fixture=fid) or []
                        if isinstance(batch, list) and batch:
                            all_fixtures[fid] = batch[0]
                    except APIFootballClientError as exc:
                        log.warning("fixture %s lookup failed: %s", fid, exc)

        self.manifest["fixtures_discovered"] = len(all_fixtures)
        deduped = dict(all_fixtures)
        self.manifest["fixtures_deduplicated"] = len(deduped)

        if team_reports and not self.dry_run:
            self.raw_store.save_coverage_report("endpoint_coverage.json", {"teams": team_reports})

        pending = [
            fid for fid in deduped
            if fid not in completed_ids
        ]
        self.resume["pending_fixture_ids"] = pending

        for fid in pending:
            fx = deduped[fid]
            status = (fx.get("fixture") or {}).get("status", {}).get("short", "")
            if status not in COMPLETED_STATUSES and status not in ("NS", "TBD", "PST"):
                continue
            try:
                self.backfill_fixture(fid, fx)
                completed_ids.add(fid)
                self.resume["completed_fixture_ids"] = sorted(completed_ids)
                if not self.dry_run:
                    save_resume_state(self.resume)
            except Exception as exc:
                log.exception("fixture %s backfill error: %s", fid, exc)
                failed_ids.add(fid)
                self.resume["failed_fixture_ids"] = sorted(failed_ids)

        self.manifest["endpoint_requests"] = self.client.endpoint_stats()
        self.manifest["cache"] = self.client.cache_stats()
        self.manifest["missing_data_summary"] = self._missing_summary
        self.manifest["failed_requests_count"] = len(self.client.failed_requests)
        self.manifest["status"] = "completed" if not self.dry_run else "dry_run"
        self.manifest["finished_at"] = datetime.now(timezone.utc).isoformat()

        if not self.dry_run:
            write_manifest("backfill_manifest.json", self.manifest)

        return self.manifest

    def print_summary(self) -> None:
        m = self.manifest
        print(f"Provider: api_football")
        print(f"Mode: {'dry-run' if self.dry_run else 'full backfill'}")
        print(f"Date range: {m.get('date_range')}")
        print(f"Teams requested: {m.get('teams_requested')}")
        print(f"Teams resolved: {m.get('teams_resolved')}")
        print(f"Cache hits: {m.get('cache', {}).get('hits', 0)}")
        print(f"Cache misses: {m.get('cache', {}).get('misses', 0)}")
        print(f"Fixtures discovered: {m.get('fixtures_discovered')}")
        print(f"Fixtures deduplicated: {m.get('fixtures_deduplicated')}")
        print(f"Endpoint requests: {m.get('endpoint_requests')}")
        print(f"Failed requests: {m.get('failed_requests_count')}")
        print(f"Output paths: {MANIFEST_ROOT}")
