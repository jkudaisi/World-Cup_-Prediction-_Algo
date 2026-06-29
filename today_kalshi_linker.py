"""Link today's WC fixtures to Kalshi markets on startup (cached across restarts)."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kalshi_market_mapper import (
    MAPPING_PATH,
    filter_wc_tickers,
    fixture_key,
    is_fixture_mapped,
    load_manual_mapping,
    map_fixture_to_tickers,
    resolve_kalshi_team,
)
from team_names import resolve_team_name, teams_match
from training_store import atomic_write_json, utc_now_iso

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
TODAY_LINKS_PATH = ROOT / "data" / "today_kalshi_links.json"

DEFAULT_LINKS_DOC: dict[str, Any] = {
    "version": 1,
    "updated_at": None,
    "fixtures": {},
}


def load_today_kalshi_links() -> dict[str, Any]:
    if not TODAY_LINKS_PATH.exists():
        return deepcopy(DEFAULT_LINKS_DOC)
    try:
        with open(TODAY_LINKS_PATH, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not read today Kalshi links cache: %s", exc)
        return deepcopy(DEFAULT_LINKS_DOC)
    doc.setdefault("fixtures", {})
    doc.setdefault("version", 1)
    return doc


def save_today_kalshi_links(doc: dict[str, Any]) -> None:
    doc["updated_at"] = utc_now_iso()
    atomic_write_json(TODAY_LINKS_PATH, doc)


def _match_date_from_today_entry(entry: dict) -> str | None:
    kickoff = entry.get("kickoff")
    if kickoff:
        return str(kickoff)[:10]
    return None


def _cache_key_for_match(entry: dict) -> str:
    home = entry.get("ml_home") or (entry.get("home") or {}).get("name", "")
    away = entry.get("ml_away") or (entry.get("away") or {}).get("name", "")
    date = _match_date_from_today_entry(entry)
    fid = entry.get("fixture_id")
    if date:
        return fixture_key(home, away, date)
    if fid is not None:
        return f"fixture_id:{fid}"
    return fixture_key(home, away)


def _entry_has_kalshi_link(entry: dict, links_doc: dict[str, Any]) -> bool:
    key = _cache_key_for_match(entry)
    cached = (links_doc.get("fixtures") or {}).get(key) or {}
    if cached.get("tickers") and filter_wc_tickers(cached["tickers"]):
        return True

    home = resolve_kalshi_team(entry.get("ml_home") or (entry.get("home") or {}).get("name", ""))
    away = resolve_kalshi_team(entry.get("ml_away") or (entry.get("away") or {}).get("name", ""))
    date = _match_date_from_today_entry(entry)
    mn = entry.get("fixture_id")
    return is_fixture_mapped(home, away, mn=mn, date=date)


def _discovery_row_for_match(entry: dict, discovery: dict[str, Any]) -> dict | None:
    from kalshi_market_discovery import find_discovery_row_for_fixture

    home = resolve_kalshi_team(entry.get("ml_home") or (entry.get("home") or {}).get("name", ""))
    away = resolve_kalshi_team(entry.get("ml_away") or (entry.get("away") or {}).get("name", ""))
    date = _match_date_from_today_entry(entry)
    return find_discovery_row_for_fixture(home, away, date, discovery)


def _write_link_from_row(
    entry: dict,
    row: dict[str, Any],
    links_doc: dict[str, Any],
) -> dict[str, Any]:
    key = _cache_key_for_match(entry)
    home = row.get("home") or resolve_kalshi_team((entry.get("home") or {}).get("name", ""))
    away = row.get("away") or resolve_kalshi_team((entry.get("away") or {}).get("name", ""))
    date = row.get("date") or _match_date_from_today_entry(entry)
    tickers = filter_wc_tickers(row.get("tickers") or {})

    link = {
        "fixture_id": entry.get("fixture_id"),
        "home": home,
        "away": away,
        "date": date,
        "fixture_key": row.get("fixture_key") or fixture_key(home, away, date),
        "kalshi_url": row.get("kalshi_advance_url") or row.get("kalshi_url"),
        "kalshi_game_url": row.get("kalshi_url"),
        "kalshi_advance_url": row.get("kalshi_advance_url"),
        "kalshi_event_ticker": row.get("kalshi_event_ticker"),
        "kalshi_advance_event_ticker": row.get("kalshi_advance_event_ticker"),
        "tickers": tickers,
        "linked_at": utc_now_iso(),
    }
    links_doc.setdefault("fixtures", {})[key] = link
    return link


def _apply_row_to_manual_mapping(row: dict[str, Any]) -> None:
    from kalshi_market_discovery import apply_discoveries_to_mapping

    apply_discoveries_to_mapping(
        {"matched": [row], "unmatched_kalshi": []},
        only_matched=True,
        overwrite_empty=True,
    )


def lookup_kalshi_link_for_today_match(entry: dict) -> dict[str, Any] | None:
    """Return cached Kalshi link metadata for a /api/today match entry."""
    links_doc = load_today_kalshi_links()
    key = _cache_key_for_match(entry)
    link = (links_doc.get("fixtures") or {}).get(key)
    if link and link.get("tickers"):
        return link

    home = resolve_kalshi_team(entry.get("ml_home") or (entry.get("home") or {}).get("name", ""))
    away = resolve_kalshi_team(entry.get("ml_away") or (entry.get("away") or {}).get("name", ""))
    date = _match_date_from_today_entry(entry)
    mapping = map_fixture_to_tickers(home, away, mn=entry.get("fixture_id"), date=date)
    tickers = filter_wc_tickers(mapping.get("tickers") or {})
    if not tickers:
        return link

    return {
        "fixture_id": entry.get("fixture_id"),
        "home": home,
        "away": away,
        "date": date,
        "fixture_key": mapping.get("mapping_key"),
        "tickers": tickers,
        "kalshi_url": link.get("kalshi_url") if link else None,
        "linked_at": link.get("linked_at") if link else None,
    }


def attach_kalshi_links_to_today_matches(matches: list[dict]) -> None:
    for entry in matches:
        link = lookup_kalshi_link_for_today_match(entry)
        if link:
            entry["kalshi"] = {
                "url": link.get("kalshi_url") or link.get("kalshi_advance_url"),
                "game_url": link.get("kalshi_game_url"),
                "advance_url": link.get("kalshi_advance_url"),
                "event_ticker": link.get("kalshi_event_ticker"),
                "advance_event_ticker": link.get("kalshi_advance_event_ticker"),
                "tickers": link.get("tickers") or {},
                "mapped_markets": len(link.get("tickers") or {}),
            }


def refresh_today_kalshi_links_on_startup(*, force: bool = False) -> dict[str, Any]:
    """
    On server start: find Kalshi markets for today's fixtures, persist links.

    Skips Kalshi API calls when every today fixture already has cached tickers.
    """
    summary: dict[str, Any] = {
        "status": "ok",
        "force": force,
        "today_matches": 0,
        "already_linked": 0,
        "newly_linked": 0,
        "skipped": [],
        "errors": [],
    }

    try:
        from kalshi_auth import credentials_configured
        if not credentials_configured():
            summary["status"] = "skipped"
            summary["reason"] = "Kalshi credentials not configured"
            return summary
    except ImportError:
        summary["status"] = "skipped"
        summary["reason"] = "Kalshi auth unavailable"
        return summary

    try:
        import scheduler
        today_view = scheduler.get_today_view()
    except Exception as exc:
        summary["status"] = "error"
        summary["error"] = f"Could not load today fixtures: {exc}"
        log.warning("Today Kalshi link startup failed: %s", exc)
        return summary

    matches = today_view.get("matches") or []
    summary["today_matches"] = len(matches)
    if not matches:
        summary["status"] = "skipped"
        summary["reason"] = "No fixtures today"
        return summary

    links_doc = load_today_kalshi_links()
    needs_kalshi = force
    for entry in matches:
        if _entry_has_kalshi_link(entry, links_doc):
            summary["already_linked"] += 1
        else:
            needs_kalshi = True

    discovery: dict[str, Any] = {}
    if needs_kalshi:
        try:
            from kalshi_client import KalshiClient
            from future_fixture_predictions import load_merged_ml_data
            from kalshi_market_discovery import discover_wc_markets, load_discovery_cache

            client = KalshiClient()
            cached = load_discovery_cache()
            # Reuse recent full discovery (< 6h) unless force or missing advance scan
            age_ok = False
            if cached.get("discovered_at_ts") and not force:
                age = datetime.now(timezone.utc).timestamp() - float(cached["discovered_at_ts"])
                series = cached.get("series_scanned") or []
                age_ok = age < 21600 and "KXWCADVANCE" in series

            if age_ok:
                discovery = cached
                summary["discovery"] = "cached"
            else:
                discovery = discover_wc_markets(
                    client,
                    ml_data=load_merged_ml_data(),
                    today_matches=matches,
                )
                summary["discovery"] = "refreshed"
        except Exception as exc:
            summary["errors"].append(str(exc))
            log.warning("Kalshi discovery for today failed: %s", exc)
            discovery = {}
            try:
                from kalshi_market_discovery import load_discovery_cache
                discovery = load_discovery_cache()
            except Exception:
                pass

    for entry in matches:
        key = _cache_key_for_match(entry)
        if not force and _entry_has_kalshi_link(entry, links_doc):
            continue

        row = _discovery_row_for_match(entry, discovery) if discovery else None
        if row and filter_wc_tickers(row.get("tickers") or {}):
            try:
                _apply_row_to_manual_mapping(row)
                _write_link_from_row(entry, row, links_doc)
                summary["newly_linked"] += 1
                log.info(
                    "Linked today fixture to Kalshi: %s vs %s → %s",
                    row.get("home"),
                    row.get("away"),
                    row.get("kalshi_advance_url") or row.get("kalshi_url"),
                )
            except Exception as exc:
                summary["errors"].append(f"{key}: {exc}")
            continue

        home = (entry.get("home") or {}).get("name", "")
        away = (entry.get("away") or {}).get("name", "")
        summary["skipped"].append({
            "fixture_key": key,
            "home": home,
            "away": away,
            "reason": "no Kalshi row found for today fixture",
        })

    if summary["newly_linked"] or needs_kalshi:
        save_today_kalshi_links(links_doc)

    log.info(
        "Today Kalshi links: %s matches, %s cached, %s newly linked, %s skipped",
        summary["today_matches"],
        summary["already_linked"],
        summary["newly_linked"],
        len(summary["skipped"]),
    )
    return summary


def build_kalshi_linked_matches_view() -> dict[str, Any]:
    """Unified API payload: discovery cache + today's persisted Kalshi links."""
    from kalshi_market_discovery import load_discovery_cache

    today_doc = load_today_kalshi_links()
    discovery = load_discovery_cache()
    by_key: dict[str, dict[str, Any]] = {}

    def _merge_row(key: str, row: dict[str, Any]) -> None:
        tickers = filter_wc_tickers(row.get("tickers") or {})
        if not tickers and not row.get("kalshi_url") and not row.get("kalshi_advance_url"):
            return
        existing = by_key.get(key) or {}
        merged_tickers = dict(existing.get("tickers") or {})
        merged_tickers.update(tickers)
        primary_url = (
            row.get("kalshi_advance_url")
            or row.get("kalshi_url")
            or existing.get("primary_url")
        )
        by_key[key] = {
            "fixture_key": key,
            "fixture_id": row.get("fixture_id") or existing.get("fixture_id"),
            "home": row.get("home") or existing.get("home"),
            "away": row.get("away") or existing.get("away"),
            "date": row.get("date") or existing.get("date"),
            "primary_url": primary_url,
            "kalshi_game_url": row.get("kalshi_game_url") or row.get("kalshi_url") or existing.get("kalshi_game_url"),
            "kalshi_advance_url": row.get("kalshi_advance_url") or existing.get("kalshi_advance_url"),
            "kalshi_event_ticker": row.get("kalshi_event_ticker") or existing.get("kalshi_event_ticker"),
            "kalshi_advance_event_ticker": row.get("kalshi_advance_event_ticker") or existing.get("kalshi_advance_event_ticker"),
            "tickers": merged_tickers,
            "mapped_markets": len(merged_tickers),
            "in_predictions": row.get("in_predictions", existing.get("in_predictions")),
            "linked_at": row.get("linked_at") or existing.get("linked_at"),
            "sources": sorted(set((existing.get("sources") or []) + [row.get("source", "unknown")])),
        }

    for row in (discovery.get("matched") or []) + (discovery.get("unmatched_kalshi") or []):
        key = row.get("fixture_key") or fixture_key(row.get("home", ""), row.get("away", ""), row.get("date"))
        payload = dict(row)
        payload["source"] = "discovery"
        payload["kalshi_game_url"] = row.get("kalshi_url")
        _merge_row(key, payload)

    for key, link in (today_doc.get("fixtures") or {}).items():
        payload = dict(link)
        payload["source"] = "today_cache"
        payload["fixture_key"] = key
        _merge_row(key, payload)

    matches = sorted(
        by_key.values(),
        key=lambda m: (m.get("date") or "9999", m.get("home") or "", m.get("away") or ""),
    )

    today_keys = set((today_doc.get("fixtures") or {}).keys())
    today_in_list = sum(1 for m in matches if m.get("fixture_key") in today_keys)

    return {
        "updated_at": utc_now_iso(),
        "today_links_updated_at": today_doc.get("updated_at"),
        "discovery_updated_at": discovery.get("discovered_at"),
        "discovery_status": discovery.get("status"),
        "count": len(matches),
        "today_cached_count": len(today_doc.get("fixtures") or {}),
        "today_in_list_count": today_in_list,
        "discovery_matched_count": len(discovery.get("matched") or []),
        "discovery_total_events": discovery.get("game_events"),
        "advance_events": discovery.get("advance_events"),
        "series_scanned": discovery.get("series_scanned") or [],
        "matches": matches,
    }
