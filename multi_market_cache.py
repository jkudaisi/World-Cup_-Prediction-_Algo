"""Persistent cache for full multi-market probability bundles."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

from future_fixture_predictions import (
    load_future_prediction_cache,
    load_merged_ml_data,
)
from knockout_progression import is_knockout_round
from multi_market_engine import build_multi_market_bundle, flatten_for_api
from training_store import atomic_write_json, utc_now_iso

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
CACHE_PATH = ROOT / "data" / "multi_market_cache.json"

DEFAULT_DOC: dict[str, Any] = {
    "version": 1,
    "updated_at": None,
    "fixtures": {},
    "stats": {"hits": 0, "misses": 0, "builds": 0},
}


def load_multi_market_cache() -> dict[str, Any]:
    if not CACHE_PATH.exists():
        return deepcopy(DEFAULT_DOC)
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not read multi-market cache: %s", exc)
        return deepcopy(DEFAULT_DOC)
    doc.setdefault("fixtures", {})
    doc.setdefault("version", 1)
    doc.setdefault("stats", {"hits": 0, "misses": 0, "builds": 0})
    return doc


def save_multi_market_cache(doc: dict[str, Any]) -> None:
    doc["updated_at"] = utc_now_iso()
    atomic_write_json(CACHE_PATH, doc)


def get_cached_bundle(fixture_id: int) -> dict[str, Any] | None:
    doc = load_multi_market_cache()
    entry = (doc.get("fixtures") or {}).get(str(fixture_id))
    if entry:
        doc["stats"]["hits"] = int(doc["stats"].get("hits", 0)) + 1
        save_multi_market_cache(doc)
        return entry.get("bundle")
    doc["stats"]["misses"] = int(doc["stats"].get("misses", 0)) + 1
    save_multi_market_cache(doc)
    return None


def build_and_cache_fixture(
    ml_match: dict,
    *,
    force: bool = False,
    run_simulation: bool = True,
) -> dict[str, Any]:
    """Build multi-market bundle and persist to cache."""
    fid = ml_match.get("fixture_id") or ml_match.get("mn")
    if fid is None:
        raise ValueError("ml_match missing fixture_id")
    fid = int(fid)

    doc = load_multi_market_cache()
    if not force and str(fid) in (doc.get("fixtures") or {}):
        return doc["fixtures"][str(fid)]["bundle"]

    bundle = build_multi_market_bundle(ml_match, run_simulation=run_simulation)
    doc.setdefault("fixtures", {})[str(fid)] = {
        "fixture_id": fid,
        "home": ml_match.get("home"),
        "away": ml_match.get("away"),
        "group": ml_match.get("group"),
        "knockout": bundle.get("knockout"),
        "bundle": bundle,
        "api": flatten_for_api(bundle),
        "cached_at": utc_now_iso(),
    }
    doc["stats"]["builds"] = int(doc["stats"].get("builds", 0)) + 1
    save_multi_market_cache(doc)
    return bundle


def _knockout_ml_matches() -> list[dict]:
    matches = []
    for m in load_merged_ml_data():
        if is_knockout_round(m.get("group")) or m.get("knockout"):
            matches.append(m)
    if matches:
        return matches
    cache = load_future_prediction_cache()
    for entry in (cache.get("fixtures") or {}).values():
        ml = entry.get("ml_match")
        if ml:
            rnd = (entry.get("round") or ml.get("group") or "")
            if is_knockout_round(rnd) or is_knockout_round(ml.get("group")):
                matches.append(ml)
    return matches


def refresh_multi_market_cache(*, force: bool = False) -> dict[str, Any]:
    """Build/cache multi-market bundles for all knockout fixtures."""
    summary = {
        "found": 0,
        "built": 0,
        "skipped": 0,
        "errors": [],
        "fixture_ids": [],
    }
    matches = _knockout_ml_matches()
    summary["found"] = len(matches)

    doc = load_multi_market_cache()
    for ml in matches:
        fid = ml.get("fixture_id") or ml.get("mn")
        if fid is None:
            summary["skipped"] += 1
            continue
        fid = int(fid)
        if not force and str(fid) in (doc.get("fixtures") or {}):
            summary["skipped"] += 1
            continue
        try:
            build_and_cache_fixture(ml, force=force)
            summary["built"] += 1
            summary["fixture_ids"].append(fid)
        except Exception as exc:
            summary["errors"].append(f"{fid}: {exc}")
            log.warning("Multi-market cache build failed for %s: %s", fid, exc)

    log.info(
        "Multi-market cache: found=%s built=%s skipped=%s errors=%s",
        summary["found"], summary["built"], summary["skipped"], len(summary["errors"]),
    )
    return summary


def refresh_multi_market_on_startup() -> dict[str, Any]:
    try:
        return refresh_multi_market_cache(force=False)
    except Exception as exc:
        log.exception("Multi-market startup refresh failed: %s", exc)
        return {"status": "error", "error": str(exc)}


def list_cached_fixtures(*, include_api: bool = False) -> list[dict[str, Any]]:
    doc = load_multi_market_cache()
    out = []
    for entry in (doc.get("fixtures") or {}).values():
        row = {
            "fixture_id": entry.get("fixture_id"),
            "home": entry.get("home"),
            "away": entry.get("away"),
            "group": entry.get("group"),
            "knockout": entry.get("knockout"),
            "cached_at": entry.get("cached_at"),
        }
        if include_api:
            api = entry.get("api")
            if not api and entry.get("bundle"):
                api = flatten_for_api(entry["bundle"])
            if api:
                row["bundle"] = api
        out.append(row)
    return sorted(out, key=lambda x: (x.get("group") or "", x.get("cached_at") or ""))


def _read_cached_bundle(fixture_id: int) -> dict[str, Any] | None:
    """Read bundle from cache without updating hit/miss stats."""
    entry = (load_multi_market_cache().get("fixtures") or {}).get(str(fixture_id))
    if not entry:
        return None
    bundle = entry.get("bundle")
    if bundle:
        return bundle
    api = entry.get("api")
    return api if api else None


def qualification_probs_for_match(
    ml_match: dict,
    *,
    score_h: int = 0,
    score_a: int = 0,
    live: bool = False,
) -> dict[str, float] | None:
    """
    Knockout qualification probabilities (advance markets), including ET/pens path.

    Uses cached pre-match bundle at 0-0; rebuilds when live or score is non-zero.
    """
    group = ml_match.get("group") or ""
    if not (is_knockout_round(group) or ml_match.get("knockout")):
        return None

    use_cache = score_h == 0 and score_a == 0 and not live
    fid = ml_match.get("fixture_id") or ml_match.get("mn")
    bundle: dict[str, Any] | None = None

    if use_cache and fid is not None:
        bundle = _read_cached_bundle(int(fid))

    if bundle is None:
        try:
            bundle = build_multi_market_bundle(
                ml_match,
                score_h=score_h,
                score_a=score_a,
                live=live,
                run_simulation=False,
            )
        except Exception as exc:
            log.debug("Qualification probability build failed: %s", exc)
            return None

    if not bundle.get("knockout"):
        return None

    km = bundle.get("kalshi_markets") or {}
    if km.get("home_qualifies") is not None and km.get("away_qualifies") is not None:
        return {
            "home_qualifies": float(km["home_qualifies"]),
            "away_qualifies": float(km["away_qualifies"]),
        }

    qual = (bundle.get("knockout_progression") or {}).get("qualification") or {}
    if qual.get("home") is None:
        return None
    return {
        "home_qualifies": float(qual["home"]),
        "away_qualifies": float(qual["away"]),
    }
