"""Map internal WC fixtures to Kalshi market tickers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from team_names import normalize_name, resolve_team_name

ROOT = Path(__file__).parent
MAPPING_PATH = ROOT / "data" / "kalshi_market_mapping.json"

# Extra aliases for Kalshi market titles (beyond team_aliases.json)
KALSHI_ALIASES: dict[str, str] = {
    "usa": "USA",
    "united states": "USA",
    "united states of america": "USA",
    "korea republic": "South Korea",
    "republic of korea": "South Korea",
    "south korea": "South Korea",
    "korea": "South Korea",
    "ivory coast": "Ivory Coast",
    "cote divoire": "Ivory Coast",
    "côte d'ivoire": "Ivory Coast",
    "congo dr": "DRC",
    "dr congo": "DRC",
    "democratic republic of congo": "DRC",
    "congo democratic republic": "DRC",
    "congo drc": "DRC",
    "drc": "DRC",
    "curacao": "Curacao",
    "cabo verde": "Cabo Verde",
    "cape verde": "Cabo Verde",
    "cape verde islands": "Cabo Verde",
    "iran": "IR Iran",
    "ir iran": "IR Iran",
    "turkey": "Turkiye",
    "turkiye": "Turkiye",
    "bosnia": "Bosnia and Herzegovina",
    "bosnia herzegovina": "Bosnia and Herzegovina",
}

MARKET_TYPES = (
    "home_win", "draw", "away_win",
    "over_0_5", "over_1_5", "over_2_5", "over_3_5", "over_4_5",
    "under_0_5", "under_1_5", "under_2_5", "under_3_5", "under_4_5",
    "btts_yes", "btts_no",
    "home_over_0_5", "home_over_1_5", "away_over_0_5", "away_over_1_5",
    "home_double_chance", "away_double_chance", "no_draw",
)

WC_TICKER_PREFIXES = ("KXWCGAME-", "KXWCBTTS-", "KXWCTOTAL-", "KXWCADVANCE-")


def is_wc_ticker(ticker: str | None) -> bool:
    t = (ticker or "").upper()
    return any(t.startswith(p) for p in WC_TICKER_PREFIXES)


def filter_wc_tickers(tickers: dict[str, str]) -> dict[str, str]:
    return {mt: t for mt, t in tickers.items() if is_wc_ticker(t)}

# Approximate WC 2026 group stage dates by match number (for mapping keys)
MATCH_DATES: dict[int, str] = {
    1: "2026-06-11", 2: "2026-06-11", 3: "2026-06-12", 4: "2026-06-12",
    5: "2026-06-12", 6: "2026-06-13", 7: "2026-06-13", 8: "2026-06-13",
    9: "2026-06-14", 10: "2026-06-14", 11: "2026-06-14", 12: "2026-06-14",
    13: "2026-06-15", 14: "2026-06-15", 15: "2026-06-15", 16: "2026-06-15",
    17: "2026-06-16", 18: "2026-06-16", 19: "2026-06-16", 20: "2026-06-16",
    21: "2026-06-20", 22: "2026-06-20", 23: "2026-06-20", 24: "2026-06-20",
    25: "2026-06-21", 26: "2026-06-21", 27: "2026-06-21", 28: "2026-06-21",
    29: "2026-06-22", 30: "2026-06-22", 31: "2026-06-22", 32: "2026-06-22",
    33: "2026-06-23", 34: "2026-06-23", 35: "2026-06-23", 36: "2026-06-23",
    37: "2026-06-24", 38: "2026-06-24", 39: "2026-06-24", 40: "2026-06-24",
    41: "2026-06-25", 42: "2026-06-25", 43: "2026-06-25", 44: "2026-06-25",
    45: "2026-06-26", 46: "2026-06-26", 47: "2026-06-26", 48: "2026-06-26",
    49: "2026-06-27", 50: "2026-06-27", 51: "2026-06-27", 52: "2026-06-27",
    53: "2026-06-28", 54: "2026-06-28", 55: "2026-06-28", 56: "2026-06-28",
    57: "2026-06-29", 58: "2026-06-29", 59: "2026-06-29", 60: "2026-06-29",
    61: "2026-06-30", 62: "2026-06-30", 63: "2026-06-30", 64: "2026-06-30",
    65: "2026-07-01", 66: "2026-07-01", 67: "2026-07-01", 68: "2026-07-01",
    69: "2026-07-02", 70: "2026-07-02", 71: "2026-07-02", 72: "2026-07-02",
}


def resolve_kalshi_team(name: str) -> str:
    key = normalize_name(name)
    if key in KALSHI_ALIASES:
        return KALSHI_ALIASES[key]
    return resolve_team_name(name)


def load_manual_mapping() -> dict[str, dict[str, str]]:
    if not MAPPING_PATH.exists():
        return {}
    with open(MAPPING_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def fixture_key(home: str, away: str, date: str | None = None, mn: int | None = None) -> str:
    h = resolve_kalshi_team(home)
    a = resolve_kalshi_team(away)
    if date:
        return f"{h}|{a}|{date}"
    if mn is not None:
        return f"mn:{mn}"
    return f"{h}|{a}"


def get_fixture_date(mn: int | None, date: str | None = None) -> str | None:
    if date:
        return date
    if mn is not None:
        return MATCH_DATES.get(mn)
    return None


def map_fixture_to_tickers(
    home: str,
    away: str,
    *,
    mn: int | None = None,
    date: str | None = None,
    kalshi_markets: list[dict] | None = None,
    discovery_cache: dict | None = None,
) -> dict[str, Any]:
    """Resolve tickers for a fixture from manual JSON, WC discovery, and title auto-match."""
    manual = load_manual_mapping()
    match_date = get_fixture_date(mn, date)
    keys = []
    if match_date:
        keys.append(fixture_key(home, away, match_date))
    if mn is not None:
        keys.append(f"mn:{mn}")
    keys.append(fixture_key(home, away))

    tickers: dict[str, str] = {}
    matched_key = None
    for key in keys:
        if key in manual:
            matched_key = key
            for mt, ticker in manual[key].items():
                if ticker:
                    tickers[mt] = ticker
            break

    try:
        from kalshi_market_discovery import discovered_tickers_for_fixture, load_discovery_cache
        cache = discovery_cache if discovery_cache is not None else load_discovery_cache()
        for mt, ticker in discovered_tickers_for_fixture(
            home, away, date=match_date, mn=mn, cache=cache,
        ).items():
            tickers.setdefault(mt, ticker)
    except ImportError:
        pass

    auto_matched = {}
    confidence = 0.0
    if kalshi_markets:
        auto_matched = _auto_match_markets(home, away, kalshi_markets)
        for mt, ticker in auto_matched.items():
            tickers.setdefault(mt, ticker)
        if auto_matched:
            confidence = max(confidence, 0.75)

    tickers = filter_wc_tickers(tickers)

    if matched_key:
        confidence = max(confidence, 0.95)
    elif tickers:
        confidence = max(confidence, 0.85 if any(k for k in keys if k in manual) else 0.75)
    else:
        confidence = 0.0

    return {
        "home": resolve_kalshi_team(home),
        "away": resolve_kalshi_team(away),
        "mn": mn,
        "date": match_date,
        "mapping_key": matched_key or keys[0],
        "tickers": tickers,
        "match_confidence": round(confidence, 3),
        "auto_matched": filter_wc_tickers(auto_matched),
    }


def _auto_match_markets(home: str, away: str, markets: list[dict]) -> dict[str, str]:
    """Heuristic match Kalshi market titles to fixture market types."""
    h = normalize_name(resolve_kalshi_team(home))
    a = normalize_name(resolve_kalshi_team(away))
    result: dict[str, str] = {}

    for m in markets:
        title = normalize_name(m.get("title") or m.get("subtitle") or "")
        ticker = m.get("ticker") or m.get("market_ticker")
        if not ticker or not is_wc_ticker(ticker):
            continue
        if h not in title or a not in title:
            continue
        # Word-boundary match — avoid matching "tie" inside "winner" or "win" in "winner".
        if re.search(r"\bdraw\b", title) or re.search(r"\btie\b", title):
            result.setdefault("draw", ticker)
        elif re.search(r"\bwinner\b", title):
            t_up = ticker.upper()
            if t_up.endswith("-TIE"):
                result.setdefault("draw", ticker)
            elif h in title and not a.endswith(" win"):
                result.setdefault("home_win", ticker)
            elif a in title:
                result.setdefault("away_win", ticker)
        elif re.search(rf"\b{re.escape(h)}\b.*\bwin", title) or f"{h} wins" in title:
            result.setdefault("home_win", ticker)
        elif re.search(rf"\b{re.escape(a)}\b.*\bwin", title) or f"{a} wins" in title:
            result.setdefault("away_win", ticker)
        elif "both teams" in title or "btts" in title:
            if "no" not in title and "not" not in title:
                result.setdefault("btts_yes", ticker)
        elif "over 0.5" in title or "over 0 5" in title:
            result.setdefault("over_0_5", ticker)
        elif "over 1.5" in title or "over 1 5" in title:
            result.setdefault("over_1_5", ticker)
        elif "over 2.5" in title or "over 2 5" in title:
            result.setdefault("over_2_5", ticker)
        elif "over 3.5" in title or "over 3 5" in title:
            result.setdefault("over_3_5", ticker)
    return result


def all_fixture_mappings(ml_data: list[dict], kalshi_markets: list[dict] | None = None) -> list[dict]:
    discovery_cache = None
    try:
        from kalshi_market_discovery import load_discovery_cache
        discovery_cache = load_discovery_cache()
    except ImportError:
        pass
    out = []
    for m in ml_data:
        kickoff = m.get("kickoff") or m.get("date")
        match_date = str(kickoff)[:10] if kickoff else get_fixture_date(m.get("mn"))
        out.append(map_fixture_to_tickers(
            m.get("home", ""),
            m.get("away", ""),
            mn=m.get("mn"),
            date=match_date,
            kalshi_markets=kalshi_markets,
            discovery_cache=discovery_cache,
        ))
    return out


def fixture_mapping_keys(home: str, away: str, *, mn: int | None = None, date: str | None = None) -> list[str]:
    """Candidate keys used to look up a fixture in kalshi_market_mapping.json."""
    match_date = get_fixture_date(mn, date)
    keys: list[str] = []
    if match_date:
        keys.append(fixture_key(home, away, match_date))
    if mn is not None:
        keys.append(f"mn:{mn}")
    keys.append(fixture_key(home, away))
    return keys


def is_fixture_mapped(
    home: str,
    away: str,
    *,
    mn: int | None = None,
    date: str | None = None,
    manual: dict | None = None,
    discovery_cache: dict | None = None,
) -> bool:
    manual = manual if manual is not None else load_manual_mapping()
    if any(key in manual for key in fixture_mapping_keys(home, away, mn=mn, date=date)):
        return True
    try:
        from kalshi_market_discovery import discovered_tickers_for_fixture, load_discovery_cache
        cache = discovery_cache if discovery_cache is not None else load_discovery_cache()
        return bool(discovered_tickers_for_fixture(
            home, away, date=date or get_fixture_date(mn), mn=mn, cache=cache,
        ))
    except ImportError:
        return False


def list_unmapped_fixtures(ml_data: list[dict]) -> list[dict[str, Any]]:
    """Fixtures from predictions with no Kalshi tickers (manual or discovered)."""
    manual = load_manual_mapping()
    try:
        from kalshi_market_discovery import load_discovery_cache
        cache = load_discovery_cache()
    except ImportError:
        cache = {}
    unmapped: list[dict[str, Any]] = []
    for match in ml_data:
        home = match.get("home", "")
        away = match.get("away", "")
        mn = match.get("mn")
        match_date = match.get("date") or (
            str(match.get("kickoff") or "")[:10] if match.get("kickoff") else None
        ) or get_fixture_date(mn)
        if is_fixture_mapped(home, away, mn=mn, date=match_date, manual=manual, discovery_cache=cache):
            continue
        unmapped.append({
            "home_team": home,
            "away_team": away,
            "date": match_date,
            "fixture_id": match.get("fixture_id"),
            "mn": mn,
        })
    return unmapped
