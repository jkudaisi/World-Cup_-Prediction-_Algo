"""Live Kalshi position tracking and hedge exits."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from pathlib import Path

from kalshi_client import KalshiClient, KalshiClientError
from market_pricing import executable_price, parse_market_quotes, parse_orderbook
from position_outcomes import evaluate_position_outcome, side_won
from trading_config import can_place_live_orders, get_config

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
LIVE_POSITIONS_PATH = ROOT / "data" / "live_positions.json"
_lock = threading.Lock()


def _load() -> dict:
    LIVE_POSITIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LIVE_POSITIONS_PATH.exists():
        return {"positions": []}
    with open(LIVE_POSITIONS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _save(data: dict) -> None:
    with open(LIVE_POSITIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def open_live_positions() -> list[dict]:
    return [p for p in _load().get("positions", []) if p.get("status") == "open"]


def register_live_position(
    *,
    order_id: str,
    ticker: str,
    side: str,
    count: int,
    entry_price_cents: float,
    model_probability: float,
    market_probability: float | None,
    edge: float,
    confidence: float,
    fixture_key: str,
    home: str,
    away: str,
    market_type: str,
    live: bool = False,
) -> dict[str, Any]:
    """Record an open live position after a filled/submitted entry order."""
    position_id = str(uuid.uuid4())
    cost = round(count * entry_price_cents / 100.0, 2)
    position = {
        "position_id": position_id,
        "entry_order_id": order_id,
        "ticker": ticker,
        "side": side.lower(),
        "count": count,
        "entry_price_cents": entry_price_cents,
        "cost": cost,
        "stake": cost,
        "model_probability": model_probability,
        "model_probability_at_entry": float(model_probability),
        "market_probability_at_entry": market_probability,
        "edge_at_entry": edge,
        "confidence": confidence,
        "fixture_key": fixture_key,
        "home": home,
        "away": away,
        "market_type": market_type,
        "live": live,
        "status": "open",
        "opened_at": datetime.now(timezone.utc).isoformat(),
    }
    with _lock:
        data = _load()
        data.setdefault("positions", []).append(position)
        _save(data)
    return position


def open_live_positions_for_fixture(fixture_key: str) -> list[dict]:
    return [p for p in open_live_positions() if p.get("fixture_key") == fixture_key]


def load_live_positions(
    *,
    fixtures: list[dict] | None = None,
    client: KalshiClient | None = None,
) -> dict[str, Any]:
    data = _load()
    positions = list(data.get("positions", []))
    if fixtures is not None:
        positions = enrich_live_positions(positions, fixtures, client=client)
    open_list = [p for p in positions if p.get("status") == "open"]
    return {
        "open_positions": sorted(open_list, key=lambda x: x.get("opened_at", ""), reverse=True),
        "open_trades": len(open_list),
        "positions": positions,
    }


def settlement_pnl(*, entry_cents: float, count: int, won: bool) -> float:
    """Realized P/L when a contract settles ($1 payout if won, $0 if lost)."""
    cost = count * entry_cents / 100.0
    if won:
        return round(count - cost, 2)
    return round(-cost, 2)


def unrealized_pnl_from_mark(
    *,
    side: str,
    entry_cents: float,
    count: int,
    side_mark_cents: float,
) -> float:
    return round((side_mark_cents - entry_cents) * count / 100.0, 2)


def fetch_ticker_pricing(client: KalshiClient, ticker: str) -> dict | None:
    """Fresh Kalshi pricing for a single ticker (orderbook with market fallback)."""
    cfg = get_config()
    pricing: dict | None = None
    try:
        raw_ob = client.get_orderbook(ticker)
        parsed = parse_orderbook(ticker, raw_ob, stale_seconds=cfg.stale_price_seconds)
        if parsed.get("implied_probability") is not None:
            pricing = parsed
    except KalshiClientError:
        pass

    if pricing is None:
        try:
            mk = client.get_market(ticker)
            parsed = parse_market_quotes(ticker, mk, stale_seconds=cfg.stale_price_seconds)
            if parsed.get("implied_probability") is not None:
                pricing = parsed
        except KalshiClientError:
            return None
    return pricing


def kalshi_settled_yes_won(client: KalshiClient, ticker: str) -> bool | None:
    """Return True/False if Kalshi market is settled, else None."""
    try:
        resp = client.get_market(ticker)
    except KalshiClientError:
        return None
    m = resp.get("market") or resp
    status = (m.get("status") or "").lower()
    if status not in ("closed", "settled", "finalized", "determined"):
        return None

    result = (m.get("result") or m.get("expiration_value") or "").lower()
    if result in ("yes", "true", "1"):
        return True
    if result in ("no", "false", "0"):
        return False

    yes_bid = m.get("yes_bid")
    if yes_bid is None:
        yes_bid = m.get("yes_bid_dollars")
        if yes_bid is not None:
            yes_bid = float(yes_bid) * 100.0
    if yes_bid is not None:
        if float(yes_bid) >= 99.0:
            return True
        if float(yes_bid) <= 1.0:
            return False
    return None


def _pricing_unreliable(pricing: dict | None) -> bool:
    if not pricing:
        return True
    spread = pricing.get("spread")
    if spread is not None and float(spread) >= 50.0:
        return True
    liq = pricing.get("available_liquidity", 0)
    if liq == 0 and spread is None:
        return True
    return False


def _side_mark_cents(pricing: dict, side: str) -> float | None:
    """Mark open position at best bid (exit price) for the held side."""
    side = (side or "yes").lower()
    bid = executable_price(pricing, side, "sell")
    if bid is not None:
        return float(bid)
    mid = pricing.get("mid_price")
    if mid is None:
        return None
    return float(mid) if side == "yes" else float(100.0 - mid)


def _fixture_for_position(position: dict, fixtures: list[dict]) -> dict | None:
    fkey = position.get("fixture_key")
    for fx in fixtures:
        if fkey and fx.get("fixture_key") == fkey:
            return fx
        if fx.get("home") == position.get("home") and fx.get("away") == position.get("away"):
            return fx
    return None


def enrich_live_positions(
    positions: list[dict],
    fixtures: list[dict],
    *,
    client: KalshiClient | None = None,
    price_cache: dict[str, dict | None] | None = None,
) -> list[dict]:
    """Attach fresh marks and unrealized P/L to open positions for the dashboard."""
    cache = price_cache if price_cache is not None else {}
    cli = client
    enriched: list[dict] = []

    for pos in positions:
        if pos.get("status") != "open":
            enriched.append(pos)
            continue

        row = dict(pos)
        fixture = _fixture_for_position(pos, fixtures)
        outcome = evaluate_position_outcome(pos, fixture)

        if outcome is not None:
            row["outcome_decided"] = True
            row["outcome_won"] = outcome["won"]
            row["outcome_reason"] = outcome["reason"]
            row["mark_yes_cents"] = 100.0 if outcome["yes_won"] else 0.0
            side = (pos.get("side") or "yes").lower()
            row["mark_side_cents"] = row["mark_yes_cents"] if side == "yes" else (100.0 - row["mark_yes_cents"])
            row["unrealized_pnl"] = settlement_pnl(
                entry_cents=float(pos.get("entry_price_cents") or 0),
                count=int(pos.get("count") or 0),
                won=outcome["won"],
            )
            enriched.append(row)
            continue

        ticker = pos.get("ticker") or ""
        pricing = cache.get(ticker) if ticker in cache else None
        if pricing is None and cli and ticker:
            pricing = fetch_ticker_pricing(cli, ticker)
            cache[ticker] = pricing

        if cli and ticker and _pricing_unreliable(pricing):
            settled = kalshi_settled_yes_won(cli, ticker)
            if settled is not None:
                side = (pos.get("side") or "yes").lower()
                won = side_won(side=side, yes_won=settled)
                row["outcome_decided"] = True
                row["outcome_won"] = won
                row["outcome_reason"] = "kalshi_settled"
                row["mark_yes_cents"] = 100.0 if settled else 0.0
                row["mark_side_cents"] = row["mark_yes_cents"] if side == "yes" else (100.0 - row["mark_yes_cents"])
                row["unrealized_pnl"] = settlement_pnl(
                    entry_cents=float(pos.get("entry_price_cents") or 0),
                    count=int(pos.get("count") or 0),
                    won=won,
                )
                enriched.append(row)
                continue

        if pricing and not _pricing_unreliable(pricing):
            yes_cents = pricing.get("mid_price")
            side = (pos.get("side") or "yes").lower()
            side_mark = _side_mark_cents(pricing, side)
            if yes_cents is not None:
                row["mark_yes_cents"] = round(float(yes_cents), 2)
            if side_mark is not None:
                row["mark_side_cents"] = round(float(side_mark), 2)
                row["unrealized_pnl"] = unrealized_pnl_from_mark(
                    side=side,
                    entry_cents=float(pos.get("entry_price_cents") or 0),
                    count=int(pos.get("count") or 0),
                    side_mark_cents=float(side_mark),
                )

        enriched.append(row)
    return enriched


def close_live_position_settlement(
    position_id: str,
    *,
    won: bool,
    reason: str = "event_outcome",
    yes_won: bool | None = None,
) -> dict[str, Any]:
    """Close an open position at settlement without placing a hedge order."""
    with _lock:
        data = _load()
        position = None
        for p in data.get("positions", []):
            if p.get("position_id") == position_id and p.get("status") == "open":
                position = p
                break
        if not position:
            return {"status": "not_found", "position_id": position_id}

        count = int(position["count"])
        entry = float(position.get("entry_price_cents") or 0)
        pnl = settlement_pnl(entry_cents=entry, count=count, won=won)

        position["status"] = "closed"
        position["closed_at"] = datetime.now(timezone.utc).isoformat()
        position["exit_reason"] = reason
        position["exit_method"] = "settlement"
        position["won"] = won
        position["pnl"] = pnl
        if yes_won is not None:
            position["settlement_yes_won"] = yes_won
        _save(data)

    log.info(
        "Live position settled %s (%s vs %s): %s pnl=%.2f",
        position_id,
        position.get("home"),
        position.get("away"),
        "won" if won else "lost",
        pnl,
    )
    return {"status": "closed", "position": position, "pnl": pnl}


def close_live_position_by_hedge(
    position_id: str,
    *,
    opposite_price_cents: float,
    client: KalshiClient | None = None,
    current_model_probability: float | None = None,
    current_market_cents: float | None = None,
    reason: str = "model_reversal",
) -> dict[str, Any]:
    """
    Hedge an open live position by buying the opposite side on Kalshi.

    Locks in $1/contract payout minus entry and hedge cost when both legs fill.
    """
    cfg = get_config()
    if not can_place_live_orders():
        return {"status": "skipped", "reason": "Live orders disabled", "position_id": position_id}

    with _lock:
        data = _load()
        position = None
        for p in data.get("positions", []):
            if p.get("position_id") == position_id and p.get("status") == "open":
                position = p
                break
        if not position:
            return {"status": "not_found", "position_id": position_id}

        count = int(position["count"])
        side = (position.get("side") or "yes").lower()
        opposite_side = "no" if side == "yes" else "yes"
        ticker = position["ticker"]
        cost = float(position["cost"])

    cli = client or KalshiClient()
    hedge_cents = float(opposite_price_cents)
    pricing = fetch_ticker_pricing(cli, ticker)
    if pricing:
        fresh_ask = executable_price(pricing, opposite_side, "buy")
        if fresh_ask is not None:
            hedge_cents = float(fresh_ask)

    hedge_cost = round(count * hedge_cents / 100.0, 2)
    max_price = min(99, int(hedge_cents + cfg.max_slippage_cents))
    order_kwargs: dict[str, Any] = {
        "ticker": ticker,
        "side": opposite_side,
        "action": "buy",
        "count": count,
        "order_type": "limit",
    }
    if opposite_side == "yes":
        order_kwargs["yes_price"] = max_price
    else:
        order_kwargs["no_price"] = max_price

    try:
        resp = cli.create_order(**order_kwargs)
    except KalshiClientError as exc:
        err = str(exc).lower()
        log.warning("Live hedge order failed for %s: %s", position_id, exc)
        if "market_closed" in err or "invalid_price" in err or "market_not_found" in err:
            settled = kalshi_settled_yes_won(cli, ticker)
            if settled is not None:
                won = side_won(side=side, yes_won=settled)
                return close_live_position_settlement(
                    position_id,
                    won=won,
                    reason="kalshi_settled",
                    yes_won=settled,
                )
        return {"status": "error", "error": str(exc), "position_id": position_id}

    order = resp.get("order") or resp
    payout = float(count)
    pnl = round(payout - cost - hedge_cost, 2)

    with _lock:
        data = _load()
        for p in data.get("positions", []):
            if p.get("position_id") != position_id or p.get("status") != "open":
                continue
            p["status"] = "closed"
            p["closed_at"] = datetime.now(timezone.utc).isoformat()
            p["exit_reason"] = reason
            p["exit_method"] = "opposite_side_hedge"
            p["hedge_side"] = opposite_side
            p["hedge_order_id"] = order.get("order_id")
            p["hedge_price_cents"] = round(float(hedge_cents), 2)
            p["pnl"] = pnl
            if current_model_probability is not None:
                p["current_model_probability"] = round(float(current_model_probability), 4)
            if current_market_cents is not None:
                p["current_market_cents"] = round(float(current_market_cents), 2)
            _save(data)
            position = p
            break

    return {
        "status": "closed",
        "position": position,
        "pnl": pnl,
        "hedge_cost": hedge_cost,
        "hedge_order": order,
    }
