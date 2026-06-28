"""Orchestrate trading opportunities, paper runs, and Kalshi integration."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from edge_engine import evaluate_edge
from entry_guards import block_live_entry
from goal_markets import build_goal_markets, exact_score_market_type, model_prob_for_market_type
from kalshi_client import KalshiClient, KalshiClientError
from kalshi_market_mapper import all_fixture_mappings, filter_wc_tickers, fixture_key, is_wc_ticker
from market_pricing import parse_market_quotes, parse_orderbook
from paper_trader import load_paper_trades, paper_stats, simulate_fill
from live_position_manager import exit_stale_live_positions, run_live_cycle, settle_decided_live_positions
from paper_position_manager import exit_stale_positions, run_paper_cycle, update_paper_marks
from risk_manager import risk_dashboard
from trade_executor import execute_order
from trade_logger import log_decision, load_all_logs
from trading_config import can_place_live_orders, config_summary, get_config
from live_trader import load_live_positions

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
PREDICTIONS_FILE = ROOT / "predictions.json"
LIVE_PREDICTIONS_FILE = ROOT / "live_predictions.json"
OPPORTUNITIES_CACHE = ROOT / "data" / "trading_opportunities.json"

_cache_lock = threading.Lock()
_build_lock = threading.Lock()
_cached_opportunities: dict[str, Any] = {"updated_at": None, "opportunities": []}


MARKET_LABELS = {
    "home_win": "Home Win",
    "draw": "Draw",
    "away_win": "Away Win",
    "over_0_5": "Over 0.5",
    "over_1_5": "Over 1.5",
    "over_2_5": "Over 2.5",
    "over_3_5": "Over 3.5",
    "over_4_5": "Over 4.5",
    "btts_yes": "BTTS Yes",
    "btts_no": "BTTS No",
    "home_over_0_5": "Home Over 0.5",
    "home_over_1_5": "Home Over 1.5",
    "away_over_0_5": "Away Over 0.5",
    "away_over_1_5": "Away Over 1.5",
    "home_double_chance": "Home or Draw",
    "away_double_chance": "Away or Draw",
    "no_draw": "No Draw",
}

SCAN_MARKETS_1X2 = ("home_win", "draw", "away_win")
SCAN_MARKETS_GOALS = ("btts_yes", "btts_no", "over_2_5", "over_3_5")
EXACT_SCORES_TO_SCAN = 3

LIVE_STATUSES = frozenset({"1H", "HT", "2H", "ET", "P", "LIVE", "BT"})
FINAL_STATUSES = frozenset({"FT", "AET", "PEN"})


def market_label(market_type: str) -> str:
    if market_type in MARKET_LABELS:
        return MARKET_LABELS[market_type]
    if market_type.startswith("exact_score_"):
        score = market_type.replace("exact_score_", "").replace("_", "-")
        return f"Exact Score {score}"
    return market_type.replace("_", " ").title()


def _clamp_prob(p: float) -> float:
    return round(min(1.0, max(0.0, float(p))), 4)


def _top_exact_scores_from_envelope(match: dict, goal_mkts: dict, n: int = EXACT_SCORES_TO_SCAN) -> list[dict]:
    """Top exact scorelines from prediction envelope score_matrix, with goal_mkts fallback."""
    pred = match.get("prediction") or {}
    sm = pred.get("score_matrix") or {}
    top = sm.get("top_exact_scores")
    if not top and isinstance(sm.get("score_matrix"), dict):
        table = sm["score_matrix"]
        top = [{"score": k, "probability": v} for k, v in sorted(table.items(), key=lambda x: x[1], reverse=True)]
    if not top:
        top = (goal_mkts.get("score_matrix_summary") or {}).get("top_exact_scores")
    if not top:
        top = goal_mkts.get("exact_score_top_5") or []
    return list(top)[:n]


def _model_prob_from_envelope(match: dict, goal_mkts: dict, market_type: str) -> float | None:
    """Model probability from prediction envelope, falling back to goal markets."""
    pred = match.get("prediction") or {}
    mt = market_type.lower()

    if mt == "home_win":
        val = pred.get("home_win")
    elif mt == "draw":
        val = pred.get("draw")
    elif mt == "away_win":
        val = pred.get("away_win")
    elif mt == "btts_yes":
        val = pred.get("both_teams_score")
    elif mt == "btts_no":
        btts = pred.get("both_teams_score")
        val = _clamp_prob(1.0 - btts) if btts is not None else None
    elif mt == "over_2_5":
        ou = pred.get("over_under") or {}
        val = ou.get("2.5", {}).get("over") if isinstance(ou.get("2.5"), dict) else pred.get("over_2_5")
    elif mt == "over_3_5":
        ou = pred.get("over_under") or {}
        val = ou.get("3.5", {}).get("over") if isinstance(ou.get("3.5"), dict) else pred.get("over_3_5")
    elif mt.startswith("exact_score_"):
        score_key = mt.replace("exact_score_", "").replace("_", "-")
        sm = pred.get("score_matrix") or {}
        for item in sm.get("top_exact_scores") or []:
            if item.get("score") == score_key:
                val = item.get("probability")
                break
        else:
            table = sm.get("score_matrix") or {}
            val = table.get(score_key) if isinstance(table, dict) else None
        if val is None:
            val = model_prob_for_market_type(goal_mkts, mt)
        return float(val) if val is not None else None
    else:
        val = None

    if val is not None:
        return float(val)
    return model_prob_for_market_type(goal_mkts, mt)


def _model_prob_for_scan(
    match: dict,
    goal_mkts: dict,
    market_type: str,
    *,
    is_live: bool,
    score_home: int = 0,
    score_away: int = 0,
) -> float | None:
    """Use score-aware goal markets in live play; pre-match envelope otherwise."""
    use_live_model = is_live or score_home > 0 or score_away > 0
    if use_live_model:
        gm_p = model_prob_for_market_type(goal_mkts, market_type)
        if gm_p is not None:
            return float(gm_p)
        outcomes = goal_mkts.get("outcomes") or {}
        mt = market_type.lower()
        if mt in outcomes:
            return float(outcomes[mt])
    return _model_prob_from_envelope(match, goal_mkts, market_type)


def _scan_market_types(match: dict, goal_mkts: dict) -> list[str]:
    """All market types to evaluate for one fixture."""
    types = list(SCAN_MARKETS_1X2) + list(SCAN_MARKETS_GOALS)
    for item in _top_exact_scores_from_envelope(match, goal_mkts):
        score = item.get("score")
        if score:
            types.append(exact_score_market_type(str(score)))
    return types


def scan_opportunities(
    *,
    match: dict,
    mapping: dict,
    goal_mkts: dict,
    live_row: dict | None,
    fkey: str,
    client: KalshiClient,
    cfg: Any | None = None,
    price_cache: dict[str, dict | None] | None = None,
    fetch_prices: bool = True,
) -> list[dict]:
    """Evaluate 1x2, BTTS, O/U, and top exact scores for one fixture."""
    cfg = cfg or get_config()
    home = match.get("home", "")
    away = match.get("away", "")
    mn = match.get("mn")
    is_live = _is_live_row(live_row)
    conf = _confidence_score(match, live_row)
    sh, sa, match_status = _score_from_live_row(live_row)
    match_final = match_status in FINAL_STATUSES
    match_status = live_row.get("status") if live_row and live_row.get("status") else ("LIVE" if is_live else "NS")
    tickers = filter_wc_tickers(mapping.get("tickers") or {})
    fixture_opps: list[dict] = []
    cache = price_cache if price_cache is not None else {}

    for mt in _scan_market_types(match, goal_mkts):
        model_p = _model_prob_for_scan(
            match,
            goal_mkts,
            mt,
            is_live=is_live,
            score_home=sh,
            score_away=sa,
        )
        if model_p is None:
            continue

        ticker = tickers.get(mt, "")
        pricing = None
        kalshi_p = None
        spread = None
        liquidity = 0
        stale = False

        if ticker and fetch_prices:
            pricing = resolve_pricing(client, ticker, cache, cfg)
            if pricing:
                kalshi_p = pricing.get("implied_probability")
                spread = pricing.get("spread")
                liquidity = pricing.get("available_liquidity", 0)
                stale = pricing.get("stale_price_warning", False)

        block_reason = block_live_entry(
            market_type=mt,
            score_home=sh,
            score_away=sa,
            match_final=match_final,
            model_yes=float(model_p),
            kalshi_yes=float(kalshi_p) if kalshi_p is not None else None,
            spread=spread,
            is_live=is_live,
        )
        if block_reason:
            edge_result = {
                "decision": "SKIP",
                "reason": f"SKIP: {block_reason}",
                "should_trade": False,
                "edge": None,
                "side": None,
            }
        elif kalshi_p is None:
            edge_result = {
                "decision": "SKIP",
                "reason": "SKIP: No Kalshi price (ticker unmapped or unavailable)",
                "should_trade": False,
                "edge": None,
                "side": None,
            }
        else:
            edge_result = evaluate_edge(
                model_probability=float(model_p),
                market_implied_probability=float(kalshi_p),
                confidence=conf,
                spread=spread,
                liquidity=liquidity,
                market_type=mt,
                match_status=match_status,
                live=is_live,
                stale=stale,
                mapping_confidence=float(mapping.get("match_confidence", 0)),
            )
        if ticker:
            log_decision(
                fixture=f"{home} vs {away}",
                market=market_label(mt),
                ticker=ticker,
                model_probability=float(model_p),
                kalshi_probability=kalshi_p,
                edge=edge_result.get("edge"),
                confidence=conf,
                spread=spread,
                liquidity=liquidity,
                decision=edge_result["decision"],
                reason=edge_result["reason"],
                risk_approval=None,
                extra={"scan": True, "mn": mn, "market_type": mt},
            )

        row = {
            "mn": mn,
            "match": f"{home} vs {away}",
            "home": home,
            "away": away,
            "fixture_key": fkey,
            "market": market_label(mt),
            "market_type": mt,
            "ticker": ticker or None,
            "model_probability": round(float(model_p), 4),
            "model_pct": round(float(model_p) * 100, 1),
            "kalshi_probability": kalshi_p,
            "kalshi_pct": round(float(kalshi_p) * 100, 1) if kalshi_p is not None else None,
            "edge": edge_result.get("edge"),
            "edge_pct": round(float(edge_result["edge"]) * 100, 1) if edge_result.get("edge") is not None else None,
            "confidence": round(conf, 3),
            "confidence_pct": round(conf * 100, 1),
            "spread": spread,
            "liquidity": liquidity,
            "recommendation": edge_result["decision"],
            "reason": edge_result["reason"],
            "side": edge_result.get("side"),
            "live": is_live,
            "mapping_confidence": mapping.get("match_confidence"),
        }
        fixture_opps.append(row)

    return fixture_opps


def _load_predictions() -> list[dict]:
    if not PREDICTIONS_FILE.exists():
        return []
    with open(PREDICTIONS_FILE, encoding="utf-8") as f:
        return json.load(f).get("ml_data", [])


def _load_live_by_teams() -> dict[tuple[str, str], dict]:
    if not LIVE_PREDICTIONS_FILE.exists():
        return {}
    with open(LIVE_PREDICTIONS_FILE, encoding="utf-8") as f:
        doc = json.load(f)
    return {(m.get("home"), m.get("away")): m for m in (doc.get("matches") or {}).values()}


def _confidence_score(match: dict, live_row: dict | None) -> float:
    if live_row:
        c = live_row.get("confidence")
        if isinstance(c, dict):
            return float(c.get("score", 0.5))
        if isinstance(c, (int, float)):
            return float(c)
    conf = match.get("confidence") or {}
    if isinstance(conf, dict):
        return float(conf.get("score", 0.5))
    return 0.5


def _is_live_row(live_row: dict | None) -> bool:
    """True only when the live snapshot is an in-play match, not stale finished data."""
    if not live_row:
        return False
    status = (live_row.get("status") or "").upper()
    if status in FINAL_STATUSES:
        return False
    if status in LIVE_STATUSES:
        return True
    return bool(live_row.get("is_live"))


def _score_from_live_row(live_row: dict | None) -> tuple[int, int, str]:
    if not live_row:
        return 0, 0, "NS"
    status = (live_row.get("status") or "NS").upper()
    score = live_row.get("score") or {}
    sh = int(score.get("home", 0))
    sa = int(score.get("away", 0))
    return sh, sa, status


def _lambdas_from_match(match: dict, live_row: dict | None) -> tuple[float, float, int, int, bool]:
    pred = match.get("prediction") or {}
    lh = float(pred.get("projected_home_goals", 1.2))
    la = float(pred.get("projected_away_goals", 1.0))
    sh, sa, status = _score_from_live_row(live_row)
    live = _is_live_row(live_row)
    if live_row:
        if live:
            lh = float(live_row.get("adj_lambda_home") or lh)
            la = float(live_row.get("adj_lambda_away") or la)
        elif status in FINAL_STATUSES:
            live = False
    return lh, la, sh, sa, live


def fetch_kalshi_markets(client: KalshiClient | None = None, limit: int = 500) -> list[dict]:
    try:
        cli = client or KalshiClient()
        seen: set[str] = set()
        markets: list[dict] = []
        resp = cli.get_markets(limit=limit, status="open")
        for m in resp.get("markets") or []:
            t = m.get("ticker") or m.get("market_ticker")
            if t and t not in seen:
                seen.add(t)
                markets.append(m)
        try:
            from kalshi_market_discovery import WC_SERIES, fetch_series_markets
            for series in WC_SERIES:
                for m in fetch_series_markets(cli, series):
                    t = m.get("ticker") or m.get("market_ticker")
                    if t and t not in seen:
                        seen.add(t)
                        markets.append(m)
        except ImportError:
            pass
        return markets
    except KalshiClientError as exc:
        log.debug("Kalshi markets unavailable: %s", exc)
        return []


def fetch_orderbook_safe(client: KalshiClient, ticker: str) -> dict | None:
    if not ticker:
        return None
    try:
        return client.get_orderbook(ticker)
    except Exception as exc:
        log.debug("Orderbook unavailable for %s: %s", ticker, exc)
        return None


def resolve_pricing(
    client: KalshiClient,
    ticker: str,
    cache: dict[str, dict | None],
    cfg: Any | None = None,
) -> dict | None:
    """Fetch and parse Kalshi pricing with per-scan cache and market-quote fallback."""
    if ticker in cache:
        return cache[ticker]

    cfg = cfg or get_config()
    pricing: dict | None = None

    raw_ob = fetch_orderbook_safe(client, ticker)
    if raw_ob:
        parsed = parse_orderbook(ticker, raw_ob, stale_seconds=cfg.stale_price_seconds)
        if parsed.get("implied_probability") is not None:
            pricing = parsed

    if pricing is None:
        try:
            mk = client.get_market(ticker)
            parsed = parse_market_quotes(ticker, mk, stale_seconds=cfg.stale_price_seconds)
            if parsed.get("implied_probability") is not None:
                pricing = parsed
        except Exception as exc:
            log.debug("Market quotes unavailable for %s: %s", ticker, exc)

    cache[ticker] = pricing
    return pricing


def _discovered_tradeable_count() -> int:
    try:
        from kalshi_market_discovery import load_discovery_cache
        cache = load_discovery_cache()
        return int(cache.get("matched_fixtures") or 0)
    except Exception:
        return 0


def _count_wc_mapped_fixtures(fixture_details: list[dict]) -> int:
    count = 0
    for f in fixture_details:
        mapping = f.get("mapping") or {}
        if filter_wc_tickers(mapping.get("tickers") or {}):
            count += 1
    return count


def _tradeable_fixture_mns() -> set[int]:
    try:
        from kalshi_market_discovery import load_discovery_cache
        cache = load_discovery_cache()
        return {int(r["mn"]) for r in cache.get("matched", []) if r.get("mn") is not None}
    except Exception:
        return set()


def _should_fetch_prices(*, mapping: dict, is_live: bool, mn: int | None = None) -> bool:
    tickers = filter_wc_tickers(mapping.get("tickers") or {})
    if not tickers:
        return False
    tradeable = _tradeable_fixture_mns()
    if mn is not None and mn in tradeable:
        return True
    if is_live and float(mapping.get("match_confidence") or 0) >= 0.95:
        return True
    return False


def _prefetch_pricing(
    client: KalshiClient,
    mappings: list[dict],
    live_flags: list[bool],
    mns: list[int | None],
    cfg: Any,
) -> dict[str, dict | None]:
    """Batch-fetch Kalshi prices only for discovery-matched live fixtures."""
    cache: dict[str, dict | None] = {}
    tickers: set[str] = set()
    for mapping, is_live, mn in zip(mappings, live_flags, mns):
        if not _should_fetch_prices(mapping=mapping, is_live=is_live, mn=mn):
            continue
        for ticker in filter_wc_tickers(mapping.get("tickers") or {}).values():
            tickers.add(ticker)
    for i, ticker in enumerate(sorted(tickers)):
        resolve_pricing(client, ticker, cache, cfg)
        if i + 1 < len(tickers):
            time.sleep(0.08)
    return cache


def _kalshi_client_if_configured() -> KalshiClient | None:
    cfg = get_config()
    if not (cfg.kalshi_api_key_id and cfg.kalshi_private_key_path):
        return None
    try:
        return KalshiClient()
    except Exception:
        return None


def run_stale_position_exits(
    fixtures: list[dict],
    *,
    kalshi_client: KalshiClient | None = None,
) -> dict[str, Any]:
    """Exit stale paper and live positions at the start of each trading scan."""
    cli = kalshi_client or _kalshi_client_if_configured()
    paper_closed = exit_stale_positions(fixtures)
    settled_closed = settle_decided_live_positions(fixtures, kalshi_client=cli)
    live_closed = exit_stale_live_positions(fixtures, kalshi_client=cli)
    return {
        "paper_stale_exits": len(paper_closed),
        "live_settled_exits": len(settled_closed),
        "live_stale_exits": len(live_closed),
        "paper_closed_trades": paper_closed,
        "live_settled_positions": settled_closed,
        "live_closed_positions": live_closed,
    }


def build_opportunities(
    *,
    kalshi_markets: list[dict] | None = None,
    client: KalshiClient | None = None,
) -> dict[str, Any]:
    """Scan all fixtures and build trading opportunity rows."""
    with _build_lock:
        return _build_opportunities_inner(kalshi_markets=kalshi_markets, client=client)


def _build_opportunities_inner(
    *,
    kalshi_markets: list[dict] | None = None,
    client: KalshiClient | None = None,
) -> dict[str, Any]:
    """Scan all fixtures and build trading opportunity rows."""
    cfg = get_config()
    ml_data = _load_predictions()
    live_by_team = _load_live_by_teams()
    cli = client or KalshiClient()

    try:
        from kalshi_market_discovery import refresh_wc_discovery_if_stale
        refresh_wc_discovery_if_stale(cli, auto_apply=True)
    except Exception as exc:
        log.debug("WC market discovery skipped: %s", exc)

    if kalshi_markets is None:
        kalshi_markets = fetch_kalshi_markets(cli)

    mappings = all_fixture_mappings(ml_data, kalshi_markets)
    opportunities: list[dict] = []
    fixture_details: list[dict] = []
    live_flags: list[bool] = []
    fixture_mns: list[int | None] = []

    for match, mapping in zip(ml_data, mappings):
        home = match.get("home", "")
        away = match.get("away", "")
        mn = match.get("mn")
        live_row = live_by_team.get((home, away))
        _, _, _, _, is_live = _lambdas_from_match(match, live_row)
        live_flags.append(is_live)
        fixture_mns.append(mn)

    price_cache = _prefetch_pricing(cli, mappings, live_flags, fixture_mns, cfg)

    for match, mapping in zip(ml_data, mappings):
        home = match.get("home", "")
        away = match.get("away", "")
        mn = match.get("mn")
        fkey = fixture_key(home, away, mapping.get("date"), mn)
        live_row = live_by_team.get((home, away))
        lh, la, sh, sa, is_live = _lambdas_from_match(match, live_row)
        conf = _confidence_score(match, live_row)
        _, _, match_status = _score_from_live_row(live_row)
        match_final = match_status in FINAL_STATUSES
        goal_mkts = build_goal_markets(lh, la, score_h=sh, score_a=sa, live=is_live or (sh > 0 or sa > 0))
        fetch_prices = _should_fetch_prices(mapping=mapping, is_live=is_live, mn=mn)

        fixture_opps = scan_opportunities(
            match=match,
            mapping=mapping,
            goal_mkts=goal_mkts,
            live_row=live_row,
            fkey=fkey,
            client=cli,
            cfg=cfg,
            price_cache=price_cache,
            fetch_prices=fetch_prices,
        )
        opportunities.extend(fixture_opps)

        trade_ideas = sum(1 for o in fixture_opps if o.get("recommendation") == "TRADE")
        mapped_markets = sum(
            1 for o in fixture_opps
            if o.get("ticker") and is_wc_ticker(o.get("ticker"))
        )
        fixture_details.append({
            "mn": mn,
            "group": match.get("group"),
            "home": home,
            "away": away,
            "home_flag": match.get("home_flag", ""),
            "away_flag": match.get("away_flag", ""),
            "fixture_key": fkey,
            "match_date": mapping.get("date"),
            "goal_markets": goal_mkts,
            "outcomes": goal_mkts.get("outcomes"),
            "mapping": mapping,
            "live": is_live,
            "score_home": sh,
            "score_away": sa,
            "match_status": match_status,
            "match_final": match_final,
            "confidence": round(conf, 3),
            "trade_ideas": trade_ideas,
            "mapped_markets": mapped_markets,
            "opportunities": fixture_opps,
        })

    run_stale_position_exits(fixture_details, kalshi_client=cli)

    update_paper_marks(fixture_details)

    live_doc = load_live_positions(fixtures=fixture_details, client=cli)
    discovered_count = _discovered_tradeable_count()
    mapped_count = discovered_count or _count_wc_mapped_fixtures(fixture_details)

    payload = {
        "updated_at": time.time(),
        "updated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": config_summary(),
        "opportunities": opportunities,
        "fixtures": fixture_details,
        "mapped_fixture_count": mapped_count,
        "kalshi_discovered_count": discovered_count,
        "paper": paper_stats(),
        "live": live_doc,
        "risk": risk_dashboard(cli),
    }

    OPPORTUNITIES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(OPPORTUNITIES_CACHE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with _cache_lock:
        _cached_opportunities.clear()
        _cached_opportunities.update(payload)

    return payload


def _load_opportunities_from_disk() -> dict[str, Any] | None:
    if not OPPORTUNITIES_CACHE.exists():
        return None
    try:
        with open(OPPORTUNITIES_CACHE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not read opportunities cache: %s", exc)
        return None


def _sync_cache_from_disk_if_newer() -> dict[str, Any] | None:
    data = _load_opportunities_from_disk()
    if not data:
        return None
    file_ts = data.get("updated_at") or 0
    with _cache_lock:
        mem_ts = _cached_opportunities.get("updated_at") or 0
        if file_ts > mem_ts:
            _cached_opportunities.clear()
            _cached_opportunities.update(data)
    return data


def get_opportunities(*, refresh: bool = False) -> dict[str, Any]:
    if not refresh:
        synced = _sync_cache_from_disk_if_newer()
        if synced and _cached_opportunities.get("opportunities"):
            with _cache_lock:
                return dict(_cached_opportunities)

    if refresh or not _cached_opportunities.get("opportunities"):
        if OPPORTUNITIES_CACHE.exists() and not refresh:
            data = _load_opportunities_from_disk()
            if data:
                with _cache_lock:
                    _cached_opportunities.clear()
                    _cached_opportunities.update(data)
                return data
        try:
            return build_opportunities()
        except Exception as exc:
            log.warning("build_opportunities failed, using cached file: %s", exc)
            data = _load_opportunities_from_disk()
            if data:
                with _cache_lock:
                    _cached_opportunities.clear()
                    _cached_opportunities.update(data)
                data = dict(data)
                data["refresh_error"] = str(exc)
                return data
            raise

    with _cache_lock:
        return dict(_cached_opportunities)


def run_live_trading_scan(*, refresh: bool = True) -> dict[str, Any]:
    """Auto live-trade on mapped Kalshi markets when live trading is enabled."""
    cfg = get_config()
    if not cfg.auto_live_trading:
        return {"status": "skipped", "reason": "Auto live trading disabled"}
    if not can_place_live_orders():
        return {
            "status": "skipped",
            "reason": "Live orders blocked — set ENABLE_LIVE_TRADING=true and KALSHI_DRY_RUN=false",
            "config": config_summary(),
        }

    data = get_opportunities(refresh=refresh)
    fixtures = data.get("fixtures") or []
    cli = _kalshi_client_if_configured()
    if cli is None:
        return {"status": "error", "reason": "Kalshi credentials not configured"}

    cycle = run_live_cycle(fixtures, kalshi_client=cli)
    return {
        "status": "success",
        "mode": "live",
        "executed": cycle["entered"],
        "closed": cycle["stale_exits"],
        "live": load_live_positions(fixtures=fixtures, client=cli),
        "config": config_summary(),
    }


def run_paper_trading_scan() -> dict[str, Any]:
    """Auto paper-trade live matches: enter, mark-to-market, exit on model drop."""
    cfg = get_config()
    if not cfg.auto_paper_trading:
        return {"status": "skipped", "reason": "Auto paper trading disabled"}

    data = get_opportunities(refresh=True)
    fixtures = data.get("fixtures") or []
    cli = _kalshi_client_if_configured()

    cycle = run_paper_cycle(fixtures, kalshi_client=cli)
    return {
        "status": "success",
        "executed": cycle["entered"],
        "closed": cycle["closed"],
        "marked": cycle["marked"],
        "paper": load_paper_trades(),
    }


def refresh_trading_cycle() -> None:
    """Called from scheduler during live polling."""
    try:
        build_opportunities()
        cfg = get_config()
        if can_place_live_orders() and cfg.auto_live_trading:
            run_live_trading_scan(refresh=False)
        elif cfg.auto_paper_trading and cfg.dry_run:
            run_paper_trading_scan()
    except Exception as exc:
        log.warning("Trading cycle failed: %s", exc)
