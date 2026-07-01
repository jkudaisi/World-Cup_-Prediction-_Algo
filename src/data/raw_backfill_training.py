"""Build production training rows from the API-Football raw data lake."""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Iterator

from apifootball_client import STAT_TYPE_MAP, _empty_side_stats, _parse_stat_value
from bootstrap_weights import compute_sample_weight
from feature_builder import build_features
from incremental_trainer import completed_match_to_training_row
from src.data.api_football_backfill import load_raw
from src.data.providers.api_football.backfill import load_backfill_config
from src.data.providers.api_football.manifest import load_resume_state
from src.data.providers.api_football.paths import COMPLETED_STATUSES, RAW_ROOT
from src.ratings.extended_team_stats import (
    ChronologicalTeamStateTracker,
    fixture_has_wc_pool_team,
    is_wc_pool_team,
    load_wc_pool_team_ids,
    reset_runtime_registry,
)
from team_names import resolve_team_name
from training_store import load_wc_matches, save_wc_matches, utc_now_iso

log = logging.getLogger(__name__)

TRAINING_DATA_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "manifests"
    / "providers"
    / "api_football"
    / "training_data_build.json"
)


def parse_fixture_statistics(
    api_response: list | None,
    home_team_id: int | None,
    away_team_id: int | None,
) -> dict[str, dict[str, Any]]:
    """Convert raw /fixtures/statistics payload to home/away stat dicts."""
    out = {"home": _empty_side_stats(), "away": _empty_side_stats()}
    if not isinstance(api_response, list):
        return out

    for block in api_response:
        team_id = (block.get("team") or {}).get("id")
        if home_team_id is not None and team_id == home_team_id:
            side = "home"
        elif away_team_id is not None and team_id == away_team_id:
            side = "away"
        else:
            idx = api_response.index(block)
            side = "home" if idx == 0 else "away"
        target = out[side]
        for item in block.get("statistics") or []:
            key = STAT_TYPE_MAP.get(item.get("type"))
            if key:
                target[key] = _parse_stat_value(item.get("value"), key)
    return out


def _load_raw_events(fixture_id: int) -> list[dict[str, Any]]:
    raw = load_raw("events", fixture_id)
    data = (raw or {}).get("data")
    return data if isinstance(data, list) else []


def _fixture_in_date_range(fixture: dict[str, Any], date_from: str, date_to: str) -> bool:
    fx_date = str((fixture.get("fixture") or {}).get("date") or "")[:10]
    if not fx_date:
        return False
    return date_from[:10] <= fx_date <= date_to[:10]


def _fixture_goals(fixture: dict[str, Any]) -> tuple[int, int] | None:
    goals = fixture.get("goals") or {}
    ft = (fixture.get("score") or {}).get("fulltime") or {}
    gh = goals.get("home") if goals.get("home") is not None else ft.get("home")
    ga = goals.get("away") if goals.get("away") is not None else ft.get("away")
    if gh is None or ga is None:
        return None
    return int(gh), int(ga)


def iter_backfill_fixtures(
    raw_root: Path | None = None,
    *,
    completed_fixture_ids: set[int] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield deduplicated fixture payloads from cached team fixture lists."""
    root = raw_root or RAW_ROOT
    team_dir = root / "fixtures" / "by_team"
    if not team_dir.exists():
        return

    seen: set[int] = set()
    for path in sorted(team_dir.glob("team_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Skipping unreadable fixture cache %s: %s", path, exc)
            continue
        for fixture in payload.get("api_response") or []:
            fid = (fixture.get("fixture") or {}).get("id")
            if fid is None:
                continue
            fid = int(fid)
            if fid in seen:
                continue
            if completed_fixture_ids is not None and fid not in completed_fixture_ids:
                continue
            seen.add(fid)
            yield fixture


def _qualifies_fixture(
    fixture: dict[str, Any],
    *,
    wc_ids: set[int],
    allow_external_opponents: bool,
) -> bool:
    status = (fixture.get("fixture") or {}).get("status", {}).get("short", "")
    if status not in COMPLETED_STATUSES:
        return False
    if _fixture_goals(fixture) is None:
        return False
    if allow_external_opponents:
        return fixture_has_wc_pool_team(fixture, wc_ids)
    home = resolve_team_name((fixture.get("teams") or {}).get("home", {}).get("name", ""))
    away = resolve_team_name((fixture.get("teams") or {}).get("away", {}).get("name", ""))
    return is_wc_pool_team(home) and is_wc_pool_team(away)


def fixture_to_training_row(
    fixture: dict[str, Any],
    *,
    home_stats: dict[str, Any] | None = None,
    away_stats: dict[str, Any] | None = None,
    allow_external_opponents: bool = True,
) -> dict[str, Any] | None:
    """Convert a cached API-Football fixture + raw layers into a training row."""
    status = (fixture.get("fixture") or {}).get("status", {}).get("short", "")
    if status not in COMPLETED_STATUSES:
        return None

    fid = int((fixture.get("fixture") or {})["id"])
    home_id = (fixture.get("teams") or {}).get("home", {}).get("id")
    away_id = (fixture.get("teams") or {}).get("away", {}).get("id")

    stats_raw = load_raw("statistics", fid)
    stats = parse_fixture_statistics(
        (stats_raw or {}).get("data"),
        home_id,
        away_id,
    )
    events = _load_raw_events(fid)

    row = completed_match_to_training_row(
        fixture,
        stats,
        events,
        home_stats=home_stats,
        away_stats=away_stats,
        allow_external_opponents=allow_external_opponents,
    )
    if row is None:
        return None

    league = fixture.get("league") or {}
    league_id = int(league.get("id") or 0)
    league_name = str(league.get("name") or "")
    match_date = row.get("date") or str((fixture.get("fixture") or {}).get("date") or "")[:10]

    row.update({
        "home_team_id": home_id,
        "away_team_id": away_id,
        "league_id": league_id,
        "league_name": league_name,
        "competition_round": str(league.get("round") or ""),
        "is_national_team": True,
        "sample_weight": compute_sample_weight(match_date, league_id),
        "source": "api_football_raw_backfill",
        "source_timestamp": utc_now_iso(),
    })
    return row


def build_training_rows_from_raw_backfill(
    *,
    raw_root: Path | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    use_resume_completed_only: bool = True,
    allow_external_opponents: bool = True,
    chronological_features: bool = True,
) -> list[dict[str, Any]]:
    """Materialize training rows from the provider raw lake."""
    config = load_backfill_config()
    dr = config.get("date_range") or {}
    d_from = date_from or dr.get("from") or "2000-01-01"
    d_to = date_to or dr.get("to") or "today"
    if d_to == "today":
        d_to = date.today().isoformat()

    completed_ids: set[int] | None = None
    if use_resume_completed_only:
        resume = load_resume_state()
        completed_ids = {int(x) for x in resume.get("completed_fixture_ids") or []}

    wc_ids = load_wc_pool_team_ids()
    fixtures: list[dict[str, Any]] = []
    skipped_date = 0
    skipped_qualify = 0

    for fixture in iter_backfill_fixtures(raw_root, completed_fixture_ids=completed_ids):
        if not _fixture_in_date_range(fixture, d_from, d_to):
            skipped_date += 1
            continue
        if not _qualifies_fixture(
            fixture,
            wc_ids=wc_ids,
            allow_external_opponents=allow_external_opponents,
        ):
            skipped_qualify += 1
            continue
        fixtures.append(fixture)

    fixtures.sort(
        key=lambda fx: (
            str((fx.get("fixture") or {}).get("date") or ""),
            int((fx.get("fixture") or {}).get("id") or 0),
        )
    )

    reset_runtime_registry()
    tracker = ChronologicalTeamStateTracker() if chronological_features else None
    rows_by_id: dict[int, dict[str, Any]] = {}

    for fixture in fixtures:
        home = resolve_team_name((fixture.get("teams") or {}).get("home", {}).get("name", ""))
        away = resolve_team_name((fixture.get("teams") or {}).get("away", {}).get("name", ""))
        home_id = (fixture.get("teams") or {}).get("home", {}).get("id")
        away_id = (fixture.get("teams") or {}).get("away", {}).get("id")

        home_stats = away_stats = None
        if tracker is not None:
            home_stats = tracker.snapshot(home, team_id=home_id)
            away_stats = tracker.snapshot(away, team_id=away_id)

        row = fixture_to_training_row(
            fixture,
            home_stats=home_stats,
            away_stats=away_stats,
            allow_external_opponents=allow_external_opponents,
        )
        if row is None:
            continue
        rows_by_id[int(row["fixture_id"])] = row

        if tracker is not None:
            goals = _fixture_goals(fixture)
            if goals is not None:
                gh, ga = goals
                tracker.apply_match(
                    home,
                    away,
                    gh,
                    ga,
                    home_xg=row.get("home_xg_proxy"),
                    away_xg=row.get("away_xg_proxy"),
                )

    rows = sorted(rows_by_id.values(), key=lambda m: (m.get("date", ""), m.get("fixture_id", 0)))
    log.info(
        "Built %s training rows from raw backfill "
        "(skipped date=%s qualify=%s external_opponents=%s chronological=%s)",
        len(rows),
        skipped_date,
        skipped_qualify,
        allow_external_opponents,
        chronological_features,
    )
    return rows


def merge_training_rows(
    existing: list[dict[str, Any]],
    backfill_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge legacy/bootstrap rows with raw-backfill rows; backfill wins on overlap."""
    by_id: dict[int, dict[str, Any]] = {}
    for row in existing:
        fid = row.get("fixture_id")
        if fid is not None:
            by_id[int(fid)] = row

    for row in backfill_rows:
        fid = int(row["fixture_id"])
        prior = by_id.get(fid) or {}
        merged = {**prior, **row}
        if prior.get("source") and prior.get("source") != row.get("source"):
            merged["prior_source"] = prior.get("source")
        by_id[fid] = merged

    return sorted(by_id.values(), key=lambda m: (m.get("date", ""), m.get("fixture_id", 0)))


def materialize_training_dataset(
    *,
    raw_root: Path | None = None,
    merge_existing: bool = True,
    write_matches: bool = True,
    allow_external_opponents: bool = True,
    chronological_features: bool = True,
) -> list[dict[str, Any]]:
    """Build from raw lake, optionally merge with legacy rows, and persist."""
    backfill_rows = build_training_rows_from_raw_backfill(
        raw_root=raw_root,
        allow_external_opponents=allow_external_opponents,
        chronological_features=chronological_features,
    )
    if not backfill_rows:
        log.warning("No training rows built from raw backfill")
        return load_wc_matches() if merge_existing else []

    existing = load_wc_matches() if merge_existing else []
    merged = merge_training_rows(existing, backfill_rows) if merge_existing else backfill_rows

    wc_only = sum(1 for r in backfill_rows if r.get("home_in_wc_pool") and r.get("away_in_wc_pool"))
    external = len(backfill_rows) - wc_only

    manifest = {
        "provider": "api_football",
        "generated_at": utc_now_iso(),
        "backfill_rows": len(backfill_rows),
        "backfill_wc_vs_wc_rows": wc_only,
        "backfill_external_opponent_rows": external,
        "existing_rows": len(existing),
        "merged_rows": len(merged),
        "merge_existing": merge_existing,
        "allow_external_opponents": allow_external_opponents,
        "chronological_features": chronological_features,
        "source": "raw_backfill_training",
    }
    TRAINING_DATA_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_DATA_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if write_matches:
        save_wc_matches(merged)
        log.info("Wrote %s training rows to world_cup_completed_matches.json", len(merged))

    return merged
