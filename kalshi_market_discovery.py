"""Discover FIFA World Cup markets from Kalshi series (KXWCGAME, etc.)."""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kalshi_client import KalshiClient, KalshiClientError
from kalshi_market_mapper import (
    MAPPING_PATH,
    fixture_key,
    get_fixture_date,
    load_manual_mapping,
    resolve_kalshi_team,
)
from team_names import find_ml_match, resolve_team_name, teams_match

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
DISCOVERY_CACHE_PATH = ROOT / "data" / "kalshi_discovered_markets.json"

KALSHI_WC_GAMES_URL = (
    "https://kalshi.com/category/sports/soccer/fifa-world-cup/world-cup/games"
)


def kalshi_game_market_url(event_ticker: str, market_ticker: str | None = None) -> str:
    """Build Kalshi web URL for a WC game event (matches kalshi.com/markets/kxwcgame/...)."""
    slug = (event_ticker or "").lower()
    if not slug.startswith("kxwcgame-"):
        slug = slug.replace("kxwcgame", "kxwcgame")
    base = f"https://kalshi.com/markets/kxwcgame/world-cup-game/{slug}"
    if market_ticker:
        return f"{base}?op_market_ticker={market_ticker}"
    return base

# Kalshi FIFA World Cup series (see kalshi.com WC games category)
WC_SERIES = (
    "KXWCGAME",   # home / draw / away
    "KXWCBTTS",   # both teams to score
    "KXWCTOTAL",  # over/under goal lines
)

# KXWCTOTAL suffix -> our market_type (Kalshi uses -1=0.5, -2=1.5, -3=2.5, ...)
TOTAL_SUFFIX_TO_MARKET: dict[str, str] = {
    "1": "over_0_5",
    "2": "over_1_5",
    "3": "over_2_5",
    "4": "over_3_5",
    "5": "over_4_5",
}

_EVENT_DATE_RE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})[A-Z-]*$", re.I)
_VS_TITLE_RE = re.compile(r"(.+?)\s+vs\.?\s+(.+?)(?:\s+Winner|\?|$)", re.I)


def parse_event_date(event_ticker: str) -> str | None:
    """Parse KXWCGAME-26JUN30FRASWE -> 2026-06-30."""
    m = _EVENT_DATE_RE.search(event_ticker or "")
    if not m:
        return None
    yy, mon, dd = m.group(1), m.group(2).upper(), m.group(3)
    months = {
        "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
        "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
        "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
    }
    mm = months.get(mon)
    if not mm:
        return None
    return f"20{yy}-{mm}-{dd}"


def parse_match_teams(title: str) -> tuple[str, str] | None:
    m = _VS_TITLE_RE.match((title or "").strip())
    if not m:
        return None
    home = resolve_kalshi_team(m.group(1).strip())
    away = resolve_kalshi_team(m.group(2).strip())
    return home, away


def fetch_series_markets(
    client: KalshiClient,
    series_ticker: str,
    *,
    status: str = "open",
) -> list[dict]:
    markets: list[dict] = []
    cursor: str | None = None
    while True:
        params: dict[str, Any] = {
            "limit": 1000,
            "status": status,
            "series_ticker": series_ticker,
        }
        if cursor:
            params["cursor"] = cursor
        try:
            resp = client.get_markets(**params)
        except KalshiClientError as exc:
            log.warning("Failed to fetch %s markets: %s", series_ticker, exc)
            break
        batch = resp.get("markets") or []
        markets.extend(batch)
        cursor = resp.get("cursor")
        if not cursor or not batch:
            break
    return markets


def _group_by_event(markets: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for m in markets:
        et = m.get("event_ticker") or ""
        if et:
            grouped.setdefault(et, []).append(m)
    return grouped


def _map_game_event(event_markets: list[dict]) -> dict[str, str]:
    tickers: dict[str, str] = {}
    title = event_markets[0].get("title") or ""
    teams = parse_match_teams(title)
    if not teams:
        return tickers
    home, away = teams
    for m in event_markets:
        ticker = m.get("ticker") or m.get("market_ticker")
        if not ticker:
            continue
        sub = (m.get("yes_sub_title") or m.get("subtitle") or "").replace("Reg Time:", "").strip()
        low = sub.lower()
        if low in ("tie", "draw"):
            tickers["draw"] = ticker
        elif teams_match(sub, home):
            tickers["home_win"] = ticker
        elif teams_match(sub, away):
            tickers["away_win"] = ticker
    return tickers


def _map_btts_event(event_markets: list[dict]) -> dict[str, str]:
    for m in event_markets:
        ticker = m.get("ticker") or m.get("market_ticker")
        if ticker:
            return {"btts_yes": ticker}
    return {}


def _map_total_event(event_markets: list[dict]) -> dict[str, str]:
    tickers: dict[str, str] = {}
    for m in event_markets:
        ticker = m.get("ticker") or m.get("market_ticker")
        if not ticker:
            continue
        suffix = ticker.rsplit("-", 1)[-1]
        mt = TOTAL_SUFFIX_TO_MARKET.get(suffix)
        if mt:
            tickers[mt] = ticker
    return tickers


def _merge_event_tickers(
    game: dict[str, str],
    btts: dict[str, str],
    totals: dict[str, str],
) -> dict[str, str]:
    out: dict[str, str] = {}
    for src in (game, btts, totals):
        for k, v in src.items():
            if v:
                out[k] = v
    return out


def discover_wc_markets(
    client: KalshiClient | None = None,
    *,
    ml_data: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Scan Kalshi WC series and match events to prediction fixtures.

    Returns discovery payload with matched/unmatched Kalshi games and tickers.
    """
    cli = client or KalshiClient()
    if ml_data is None:
        pred_path = ROOT / "predictions.json"
        if pred_path.exists():
            with open(pred_path, encoding="utf-8") as f:
                ml_data = json.load(f).get("ml_data", [])
        else:
            ml_data = []

    by_series: dict[str, list[dict]] = {}
    for series in WC_SERIES:
        by_series[series] = fetch_series_markets(cli, series)

    game_events = _group_by_event(by_series.get("KXWCGAME", []))
    btts_events = _group_by_event(by_series.get("KXWCBTTS", []))
    total_events = _group_by_event(by_series.get("KXWCTOTAL", []))

    def event_suffix(event_ticker: str, series: str) -> str:
        prefix = f"{series}-"
        return event_ticker[len(prefix):] if event_ticker.startswith(prefix) else event_ticker

    btts_by_suffix = {event_suffix(et, "KXWCBTTS"): ms for et, ms in btts_events.items()}
    totals_by_suffix = {event_suffix(et, "KXWCTOTAL"): ms for et, ms in total_events.items()}

    matched: list[dict] = []
    unmatched_kalshi: list[dict] = []

    for event_ticker, event_markets in sorted(game_events.items()):
        title = event_markets[0].get("title") or ""
        teams = parse_match_teams(title)
        if not teams:
            continue
        home, away = teams
        date = parse_event_date(event_ticker)
        suffix = event_suffix(event_ticker, "KXWCGAME")

        game_tickers = _map_game_event(event_markets)
        btts_tickers = _map_btts_event(btts_by_suffix.get(suffix, []))
        total_tickers = _map_total_event(totals_by_suffix.get(suffix, []))
        tickers = _merge_event_tickers(game_tickers, btts_tickers, total_tickers)
        fkey = fixture_key(home, away, date)

        fx = find_ml_match(home, away, ml_data) or find_ml_match(away, home, ml_data)

        row = {
            "kalshi_event_ticker": event_ticker,
            "kalshi_title": title,
            "kalshi_url": kalshi_game_market_url(event_ticker),
            "home": home,
            "away": away,
            "date": date,
            "fixture_key": fkey,
            "tickers": tickers,
            "mapped_markets": len(tickers),
            "in_predictions": fx is not None,
            "mn": fx.get("mn") if fx else None,
        }
        if fx:
            row["prediction_home"] = fx.get("home")
            row["prediction_away"] = fx.get("away")
            matched.append(row)
        else:
            unmatched_kalshi.append(row)

    payload = {
        "discovered_at": datetime.now(timezone.utc).isoformat(),
        "discovered_at_ts": time.time(),
        "kalshi_wc_url": KALSHI_WC_GAMES_URL,
        "series_scanned": list(WC_SERIES),
        "series_counts": {s: len(by_series.get(s, [])) for s in WC_SERIES},
        "game_events": len(game_events),
        "matched_fixtures": len(matched),
        "unmatched_kalshi_games": len(unmatched_kalshi),
        "matched": matched,
        "unmatched_kalshi": unmatched_kalshi,
    }
    _save_discovery_cache(payload)
    return payload


def _save_discovery_cache(payload: dict[str, Any]) -> None:
    DISCOVERY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DISCOVERY_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_discovery_cache() -> dict[str, Any]:
    if not DISCOVERY_CACHE_PATH.exists():
        return {}
    with open(DISCOVERY_CACHE_PATH, encoding="utf-8") as f:
        return json.load(f)


def discovered_tickers_for_fixture(
    home: str,
    away: str,
    *,
    date: str | None = None,
    mn: int | None = None,
    cache: dict | None = None,
) -> dict[str, str]:
    """Look up auto-discovered tickers for a fixture from cache."""
    cache = cache if cache is not None else load_discovery_cache()
    home_c = resolve_team_name(home)
    away_c = resolve_team_name(away)
    match_date = date or get_fixture_date(mn)

    for row in cache.get("matched", []) + cache.get("unmatched_kalshi", []):
        if not teams_match(row.get("home", ""), home_c):
            continue
        if not teams_match(row.get("away", ""), away_c):
            continue
        if match_date and row.get("date") and row["date"] != match_date:
            continue
        tickers = row.get("tickers") or {}
        if tickers:
            return dict(tickers)
    return {}


def apply_discoveries_to_mapping(
    discoveries: dict[str, Any] | None = None,
    *,
    only_matched: bool = True,
    overwrite_empty: bool = True,
) -> dict[str, Any]:
    """
    Merge discovered tickers into data/kalshi_market_mapping.json.

    Manual non-empty entries are preserved unless overwrite_empty is False.
    """
    discoveries = discoveries or load_discovery_cache()
    manual = load_manual_mapping()
    rows = discoveries.get("matched", [])
    if not only_matched:
        rows = rows + discoveries.get("unmatched_kalshi", [])

    updated_keys: list[str] = []
    added_tickers = 0

    for row in rows:
        tickers = {k: v for k, v in (row.get("tickers") or {}).items() if v}
        if not tickers:
            continue
        keys = [row.get("fixture_key")]
        if row.get("mn"):
            keys.append(f"mn:{row['mn']}")
        if row.get("date"):
            keys.append(fixture_key(row["home"], row["away"], row["date"]))

        for key in keys:
            if not key:
                continue
            entry = dict(manual.get(key) or {})
            changed = False
            for mt, ticker in tickers.items():
                if not overwrite_empty and entry.get(mt):
                    continue
                if entry.get(mt) != ticker:
                    entry[mt] = ticker
                    changed = True
                    added_tickers += 1
            if changed or key not in manual:
                manual[key] = entry
                if key not in updated_keys:
                    updated_keys.append(key)

    doc: dict[str, Any] = {"_comment": "Kalshi ticker overrides and auto-discovered WC mappings"}
    if MAPPING_PATH.exists():
        with open(MAPPING_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        if raw.get("_comment"):
            doc["_comment"] = raw["_comment"]
    doc.update(manual)

    MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)

    return {
        "status": "applied",
        "keys_updated": len(updated_keys),
        "tickers_written": added_tickers,
        "updated_keys": updated_keys,
    }


def refresh_wc_discovery_if_stale(
    client: KalshiClient | None = None,
    *,
    max_age_seconds: int = 900,
    auto_apply: bool = True,
) -> dict[str, Any]:
    """Refresh discovery cache if older than max_age_seconds."""
    cache = load_discovery_cache()
    age = time.time() - float(cache.get("discovered_at_ts") or 0)
    if cache and age < max_age_seconds:
        return {"status": "cached", "age_seconds": round(age), **cache}

    result = discover_wc_markets(client=client)
    apply_result = {}
    if auto_apply and result.get("matched"):
        apply_result = apply_discoveries_to_mapping(result)
    return {
        "status": "refreshed",
        **result,
        "apply": apply_result,
    }
