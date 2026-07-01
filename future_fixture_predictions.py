"""Discover upcoming WC fixtures from API-Football and cache ML predictions."""

from __future__ import annotations

import json
import logging
import re
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from apifootball_client import APIFootballError, WC_LEAGUE_ID, WC_SEASON, get_season_fixtures
from local_schedule import parse_kickoff_utc
from model_store import load_artifacts, models_exist
from team_names import resolve_team_name
from training_store import atomic_write_json, utc_now_iso
from wc2026_ml_pipeline import TEAM_STATS
from src.models.predict_match import predict_match

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
CACHE_PATH = ROOT / "data" / "future_fixture_prediction_cache.json"
CACHE_VERSION = 1

DEFAULT_CACHE: dict[str, Any] = {
    "version": CACHE_VERSION,
    "updated_at": None,
    "fixtures": {},
    "last_refresh": {},
}

SCHEDULED_STATUSES = frozenset({"NS", "TBD"})
SKIP_STATUSES = frozenset({"PST", "CANC", "ABD", "AWD", "WO"})
PLAYED_OR_LIVE = frozenset({
    "FT", "AET", "PEN", "1H", "HT", "2H", "ET", "P", "LIVE", "BT", "INT", "SUSP",
})

_PLACEHOLDER_RE = re.compile(
    r"(?:\b(tbd|to be determined)\b|"
    r"\bwinner\b|\brunner.?up\b|\bloser\b|"
    r"\b(?:1st|2nd|3rd|best)\b|"
    r"\b\d+(?:st|nd|rd|th)\s+(?:group|placed)\b|"
    r"\bgroup\s+[a-l]\b.*\b(?:winner|runner|second|third)\b)",
    re.IGNORECASE,
)

_ROUND_GROUP_MAP = (
    (re.compile(r"round of 32", re.I), "R32"),
    (re.compile(r"round of 16", re.I), "R16"),
    (re.compile(r"quarter.?final", re.I), "QF"),
    (re.compile(r"semi.?final", re.I), "SF"),
    (re.compile(r"3rd place|third place", re.I), "3P"),
    (re.compile(r"\bfinal\b", re.I), "F"),
    (re.compile(r"group\s+([a-l])\b", re.I), None),
)


def _round_to_group(round_name: str) -> str:
    text = (round_name or "").strip()
    if not text:
        return "KO"
    for pattern, label in _ROUND_GROUP_MAP:
        m = pattern.search(text)
        if m:
            if label is None:
                return m.group(1).upper()
            return label
    return "KO"


def _is_knockout_round(round_name: str) -> bool:
    text = (round_name or "").lower()
    if not text:
        return False
    if text.startswith("group"):
        return False
    return True


def _team_side_ok(team: dict | None) -> tuple[bool, str]:
    if not team:
        return False, "missing team"
    name = (team.get("name") or "").strip()
    team_id = team.get("id")
    if not name or team_id in (None, 0):
        return False, "missing team name or id"
    if _PLACEHOLDER_RE.search(name):
        return False, f"placeholder team name: {name!r}"
    return True, name


def is_fixture_predictable(fixture: dict) -> tuple[bool, str]:
    """Return (True, reason) when both teams are confirmed and match is upcoming."""
    fix = fixture.get("fixture") or {}
    status = (fix.get("status") or {}).get("short", "NS").upper()
    if status in SKIP_STATUSES:
        return False, f"status {status}"
    if status in PLAYED_OR_LIVE:
        return False, f"already started or finished ({status})"

    kickoff = parse_kickoff_utc(fixture)
    now = datetime.now(timezone.utc)
    if kickoff < now:
        return False, "kickoff in the past"

    home_ok, home_info = _team_side_ok(fixture.get("teams", {}).get("home"))
    if not home_ok:
        return False, f"home: {home_info}"
    away_ok, away_info = _team_side_ok(fixture.get("teams", {}).get("away"))
    if not away_ok:
        return False, f"away: {away_info}"

    home_ml = resolve_team_name(home_info)
    away_ml = resolve_team_name(away_info)
    if home_ml not in TEAM_STATS:
        return False, f"home team not in model database: {home_info!r} -> {home_ml!r}"
    if away_ml not in TEAM_STATS:
        return False, f"away team not in model database: {away_info!r} -> {away_ml!r}"

    if status not in SCHEDULED_STATUSES:
        return False, f"unexpected status {status}"

    return True, "ok"


def fetch_future_world_cup_fixtures() -> list[dict]:
    """Upcoming WC fixtures from today onward (includes knockout as they appear)."""
    try:
        all_fixtures = get_season_fixtures()
    except APIFootballError as exc:
        log.warning("Could not fetch season fixtures: %s", exc)
        raise

    now = datetime.now(timezone.utc)
    upcoming: list[dict] = []
    for fx in all_fixtures:
        if fx.get("league", {}).get("id") != WC_LEAGUE_ID:
            continue
        kickoff = parse_kickoff_utc(fx)
        if kickoff < now:
            continue
        status = (fx.get("fixture") or {}).get("status", {}).get("short", "NS").upper()
        if status in SKIP_STATUSES or status in PLAYED_OR_LIVE:
            continue
        upcoming.append(fx)

    upcoming.sort(key=parse_kickoff_utc)
    return upcoming


def load_future_prediction_cache() -> dict[str, Any]:
    if not CACHE_PATH.exists():
        return deepcopy(DEFAULT_CACHE)

    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Future cache corrupted (%s) — resetting", exc)
        _backup_corrupt_cache()
        return deepcopy(DEFAULT_CACHE)

    if not isinstance(raw, dict):
        log.warning("Future cache corrupted (not a dict) — resetting")
        _backup_corrupt_cache()
        return deepcopy(DEFAULT_CACHE)

    fixtures = raw.get("fixtures")
    if fixtures is not None and not isinstance(fixtures, dict):
        log.warning("Future cache corrupted (fixtures not a dict) — resetting")
        _backup_corrupt_cache()
        return deepcopy(DEFAULT_CACHE)

    raw.setdefault("version", CACHE_VERSION)
    raw.setdefault("fixtures", {})
    raw.setdefault("updated_at", None)
    raw.setdefault("last_refresh", {})
    return raw


def _backup_corrupt_cache() -> None:
    if CACHE_PATH.exists():
        backup = CACHE_PATH.with_suffix(".json.bak")
        try:
            shutil.copy2(CACHE_PATH, backup)
            log.info("Backed up corrupt cache to %s", backup)
        except OSError as exc:
            log.warning("Could not back up corrupt cache: %s", exc)


def save_future_prediction_cache(cache: dict[str, Any]) -> None:
    cache["version"] = CACHE_VERSION
    cache["updated_at"] = utc_now_iso()
    atomic_write_json(CACHE_PATH, cache)


def is_fixture_cached(fixture_id: int, cache: dict[str, Any]) -> bool:
    return str(fixture_id) in (cache.get("fixtures") or {})


def _build_cache_entry(fixture: dict, ml_match: dict, model_versions: dict[str, str]) -> dict[str, Any]:
    fix = fixture["fixture"]
    league = fixture.get("league") or {}
    teams = fixture.get("teams") or {}
    home = teams.get("home") or {}
    away = teams.get("away") or {}
    now = utc_now_iso()
    pred = ml_match.get("prediction") or {}

    return {
        "fixture_id": fix["id"],
        "league_id": league.get("id", WC_LEAGUE_ID),
        "season": league.get("season", WC_SEASON),
        "round": league.get("round", ""),
        "date": fix.get("date"),
        "venue": fix.get("venue") or {},
        "home_team": ml_match.get("home"),
        "away_team": ml_match.get("away"),
        "home_team_id": home.get("id"),
        "away_team_id": away.get("id"),
        "home_team_api": home.get("name"),
        "away_team_api": away.get("name"),
        "predicted_score": ml_match.get("ens"),
        "predicted_home_goals": ml_match.get("ens_h"),
        "predicted_away_goals": ml_match.get("ens_a"),
        "probabilities": {
            "home_win": pred.get("home_win"),
            "draw": pred.get("draw"),
            "away_win": pred.get("away_win"),
            "projected_home_goals": pred.get("projected_home_goals"),
            "projected_away_goals": pred.get("projected_away_goals"),
        },
        "confidence": ml_match.get("confidence"),
        "model_version": model_versions,
        "ml_match": ml_match,
        "created_at": now,
        "updated_at": now,
    }


def predict_and_cache_fixture(
    fixture: dict,
    cache: dict[str, Any],
    *,
    artifacts: dict[str, Any] | None = None,
    force: bool = False,
) -> dict[str, Any] | None:
    """Predict one fixture and store in cache. Returns cache entry or None."""
    fid = fixture["fixture"]["id"]
    if not force and is_fixture_cached(fid, cache):
        return cache["fixtures"][str(fid)]

    ok, reason = is_fixture_predictable(fixture)
    if not ok:
        raise ValueError(reason)

    home_raw = fixture["teams"]["home"]["name"]
    away_raw = fixture["teams"]["away"]["name"]
    home = resolve_team_name(home_raw)
    away = resolve_team_name(away_raw)
    round_name = (fixture.get("league") or {}).get("round", "")
    group = _round_to_group(round_name)
    knockout = _is_knockout_round(round_name)

    if artifacts is None:
        artifacts = load_artifacts()
    if not artifacts:
        raise RuntimeError("No trained model artifacts available")

    home_id = fixture["teams"]["home"]["id"]
    away_id = fixture["teams"]["away"]["id"]
    kickoff = fixture["fixture"].get("date")

    ml_match = predict_match(
        home,
        away,
        home_team_id=home_id,
        away_team_id=away_id,
        match_date=kickoff,
        group=group,
        match_number=fid,
        trained=artifacts["trained"],
        scaler=artifacts["scaler"],
        feature_cols=artifacts["feature_cols"],
        knockout=knockout,
        competition_context={
            "fixture_id": fid,
            "knockout_stage": 1.0 if knockout else 0.0,
        },
    )
    ml_match["fixture_id"] = fid
    ml_match["round"] = round_name
    ml_match["kickoff"] = fixture["fixture"].get("date")
    ml_match["source"] = "future_cache"

    entry = _build_cache_entry(fixture, ml_match, artifacts.get("model_versions") or {})
    key = str(fid)
    existing = cache["fixtures"].get(key)
    if existing and force:
        entry["created_at"] = existing.get("created_at") or entry["created_at"]
    cache["fixtures"][key] = entry
    if knockout:
        try:
            from multi_market_cache import build_and_cache_fixture
            build_and_cache_fixture(ml_match, force=force, run_simulation=True)
        except Exception as exc:
            log.warning("Multi-market cache for fixture %s skipped: %s", fid, exc)
    return entry


def refresh_future_fixture_predictions(*, force: bool = False) -> dict[str, Any]:
    """
    Fetch upcoming fixtures, predict missing ones, persist cache.

    Returns summary dict with counts and skip reasons.
    """
    summary: dict[str, Any] = {
        "status": "ok",
        "force": force,
        "found": 0,
        "already_cached": 0,
        "predicted": 0,
        "skipped": [],
        "errors": [],
        "fixture_ids_predicted": [],
    }

    if not models_exist():
        summary["status"] = "skipped"
        summary["reason"] = "No trained models on disk — run /api/run or training first"
        log.warning(summary["reason"])
        return summary

    try:
        fixtures = fetch_future_world_cup_fixtures()
    except APIFootballError as exc:
        summary["status"] = "error"
        summary["error"] = str(exc)
        log.warning("Future fixture fetch failed: %s", exc)
        return summary

    summary["found"] = len(fixtures)
    cache = load_future_prediction_cache()
    artifacts = load_artifacts()

    for fx in fixtures:
        fid = fx["fixture"]["id"]
        if not force and is_fixture_cached(fid, cache):
            summary["already_cached"] += 1
            continue

        ok, reason = is_fixture_predictable(fx)
        if not ok:
            summary["skipped"].append({"fixture_id": fid, "reason": reason})
            continue

        try:
            predict_and_cache_fixture(fx, cache, artifacts=artifacts, force=force)
            summary["predicted"] += 1
            summary["fixture_ids_predicted"].append(fid)
            log.info(
                "Cached future prediction: fixture %s %s vs %s (%s)",
                fid,
                resolve_team_name(fx["teams"]["home"]["name"]),
                resolve_team_name(fx["teams"]["away"]["name"]),
                (fx.get("league") or {}).get("round", ""),
            )
        except Exception as exc:
            msg = f"fixture {fid}: {exc}"
            summary["errors"].append(msg)
            log.warning("Future fixture prediction failed: %s", msg)

    if summary["predicted"] > 0 or force:
        cache["last_refresh"] = {
            **summary,
            "completed_at": utc_now_iso(),
        }
        save_future_prediction_cache(cache)
    else:
        cache["last_refresh"] = {
            **summary,
            "completed_at": utc_now_iso(),
        }
        save_future_prediction_cache(cache)

    log.info(
        "Future fixture refresh: found=%s cached=%s predicted=%s skipped=%s errors=%s",
        summary["found"],
        summary["already_cached"],
        summary["predicted"],
        len(summary["skipped"]),
        len(summary["errors"]),
    )
    return summary


def refresh_future_fixture_predictions_on_startup() -> dict[str, Any]:
    """Startup hook — never raises."""
    try:
        return refresh_future_fixture_predictions(force=False)
    except Exception as exc:
        log.exception("Future fixture startup refresh failed: %s", exc)
        return {"status": "error", "error": str(exc)}


def load_merged_predictions_doc() -> dict[str, Any]:
    """Load predictions.json merged with future knockout fixture cache."""
    pred_path = ROOT / "predictions.json"
    doc: dict[str, Any] = {"ml_data": [], "team_elo": {}, "stats": None, "training": None}
    if pred_path.exists():
        try:
            with open(pred_path, encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Could not read predictions.json: %s", exc)
    return merge_future_predictions_into_doc(doc)


def load_merged_ml_data() -> list[dict]:
    return load_merged_predictions_doc().get("ml_data") or []


def merge_future_predictions_into_doc(predictions_doc: dict[str, Any]) -> dict[str, Any]:
    """Append cached future fixtures to ml_data without modifying predictions.json."""
    cache = load_future_prediction_cache()
    merged = deepcopy(predictions_doc)
    ml_data = list(merged.get("ml_data") or [])

    existing_fixture_ids = {
        int(m["fixture_id"])
        for m in ml_data
        if m.get("fixture_id") is not None
    }
    existing_pairs = {
        (m.get("home"), m.get("away"))
        for m in ml_data
        if m.get("home") and m.get("away")
    }

    added = 0
    for entry in (cache.get("fixtures") or {}).values():
        ml_match = entry.get("ml_match")
        if not ml_match:
            continue
        fid = entry.get("fixture_id")
        pair = (ml_match.get("home"), ml_match.get("away"))
        if fid in existing_fixture_ids:
            continue
        if pair in existing_pairs:
            continue
        ml_data.append(deepcopy(ml_match))
        existing_fixture_ids.add(fid)
        existing_pairs.add(pair)
        added += 1

    def _sort_key(m: dict) -> tuple:
        kickoff = m.get("kickoff") or ""
        mn = m.get("mn")
        if mn is None:
            mn = m.get("fixture_id") or 0
        return kickoff, int(mn) if isinstance(mn, int) else 0

    ml_data.sort(key=_sort_key)
    merged["ml_data"] = ml_data
    merged["future_fixture_cache"] = {
        "count": len(cache.get("fixtures") or {}),
        "added_to_ml_data": added,
        "updated_at": cache.get("updated_at"),
        "last_refresh": cache.get("last_refresh"),
    }
    return merged


def lookup_ml_prediction(
    fixture_id: int | None,
    home: str,
    away: str,
) -> dict[str, Any] | None:
    """Find ml_match by fixture_id, then team pair (cache + group-stage predictions.json)."""
    cache = load_future_prediction_cache()
    fixtures_map = cache.get("fixtures") or {}

    if fixture_id is not None:
        entry = fixtures_map.get(str(fixture_id))
        if entry and entry.get("ml_match"):
            return deepcopy(entry["ml_match"])

    for entry in fixtures_map.values():
        ml = entry.get("ml_match") or {}
        if ml.get("home") == home and ml.get("away") == away:
            return deepcopy(ml)

    pred_path = ROOT / "predictions.json"
    if pred_path.exists():
        try:
            with open(pred_path, encoding="utf-8") as f:
                doc = json.load(f)
            for m in doc.get("ml_data") or []:
                if m.get("home") == home and m.get("away") == away:
                    return deepcopy(m)
        except (OSError, json.JSONDecodeError) as exc:
            log.debug("Could not read predictions.json for lookup: %s", exc)

    return None


def ensure_predictions_for_fixtures(fixtures: list[dict]) -> int:
    """Predict and cache any missing predictable fixtures from a fixture list."""
    if not fixtures or not models_exist():
        return 0

    cache = load_future_prediction_cache()
    artifacts = load_artifacts()
    if not artifacts:
        return 0

    predicted = 0
    for fx in fixtures:
        fid = fx["fixture"]["id"]
        if is_fixture_cached(fid, cache):
            continue
        ok, _ = is_fixture_predictable(fx)
        if not ok:
            continue
        try:
            predict_and_cache_fixture(fx, cache, artifacts=artifacts)
            predicted += 1
        except Exception as exc:
            log.debug("Could not predict fixture %s: %s", fid, exc)

    if predicted:
        save_future_prediction_cache(cache)
        log.info("Cached %s prediction(s) for today/upcoming fixtures", predicted)
    return predicted


def attach_ml_predictions_to_today_matches(
    matches: list[dict],
    raw_fixtures: list[dict],
) -> None:
    """Mutate today match dicts with ml_prediction from cache / predictions.json."""
    fixture_by_id = {f["fixture"]["id"]: f for f in raw_fixtures}
    ensure_predictions_for_fixtures(raw_fixtures)

    for entry in matches:
        fid = entry.get("fixture_id")
        ml_home = entry.get("ml_home") or ""
        ml_away = entry.get("ml_away") or ""
        pred = lookup_ml_prediction(fid, ml_home, ml_away)
        if pred:
            entry["ml_prediction"] = pred
