"""Convert Kalshi orderbooks into market probabilities and liquidity metrics."""

from __future__ import annotations

import time
from typing import Any


def clamp_prob(p: float) -> float:
    return round(min(1.0, max(0.0, float(p))), 4)


def cents_to_prob(cents: float | int | None) -> float | None:
    if cents is None:
        return None
    return clamp_prob(float(cents) / 100.0)


def prob_to_pct(p: float | None) -> str:
    if p is None:
        return "—"
    return f"{p * 100:.1f}%"


def _price_to_cents(raw: Any) -> float | None:
    """Kalshi cent ints (45) or dollar strings/floats (0.45 or '0.4500') -> cents."""
    if raw is None:
        return None
    p = float(raw)
    if p <= 1.0:
        return p * 100.0
    return p


def _qty_from_level(raw: Any) -> int:
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return 0


def _unwrap_orderbook(orderbook: dict | None) -> dict:
    ob = orderbook or {}
    if "orderbook_fp" in ob:
        return ob["orderbook_fp"] or {}
    if "orderbook" in ob:
        return ob["orderbook"] or {}
    return ob


def _best_level(levels: list | None, *, side: str) -> tuple[float | None, int]:
    """Best bid = highest price; best ask derived from opposite side in Kalshi binary markets."""
    if not levels:
        return None, 0
    parsed = []
    for row in levels:
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            price = _price_to_cents(row[0])
            qty = _qty_from_level(row[1])
            if price is not None:
                parsed.append((price, qty))
        elif isinstance(row, dict):
            price = _price_to_cents(row.get("price", row.get("yes_price")))
            qty = _qty_from_level(row.get("quantity", row.get("count", 0)))
            if price is not None:
                parsed.append((price, qty))
    if not parsed:
        return None, 0
    if side == "bid":
        best = max(parsed, key=lambda x: x[0])
    else:
        best = min(parsed, key=lambda x: x[0])
    return best[0], best[1]


def _pricing_from_sides(
    ticker: str,
    *,
    best_yes_bid: float | None,
    best_yes_ask: float | None,
    best_no_bid: float | None,
    best_no_ask: float | None,
    yes_bid_liq: int = 0,
    no_bid_liq: int = 0,
    fetched_at: float | None = None,
    stale_seconds: int = 120,
    age: float | None = None,
) -> dict[str, Any]:
    now = fetched_at or time.time()
    mid_price = None
    implied_probability = None
    spread = None

    if best_yes_bid is not None and best_yes_ask is not None:
        mid_price = (best_yes_bid + best_yes_ask) / 2.0
        implied_probability = cents_to_prob(mid_price)
        spread = round(best_yes_ask - best_yes_bid, 2)
    elif best_yes_bid is not None:
        mid_price = best_yes_bid
        implied_probability = cents_to_prob(best_yes_bid)
    elif best_yes_ask is not None:
        mid_price = best_yes_ask
        implied_probability = cents_to_prob(best_yes_ask)

    available_liquidity = yes_bid_liq + no_bid_liq
    liquidity_score = min(1.0, available_liquidity / 100.0)
    stale_price_warning = bool(age is not None and age > stale_seconds)

    return {
        "ticker": ticker,
        "best_yes_bid": best_yes_bid,
        "best_yes_ask": best_yes_ask,
        "best_no_bid": best_no_bid,
        "best_no_ask": best_no_ask,
        "mid_price": round(mid_price, 2) if mid_price is not None else None,
        "implied_probability": implied_probability,
        "spread": spread,
        "available_liquidity": available_liquidity,
        "liquidity_score": round(liquidity_score, 3),
        "stale_price_warning": stale_price_warning,
        "fetched_at": now,
    }


def parse_orderbook(
    ticker: str,
    orderbook: dict | None,
    *,
    fetched_at: float | None = None,
    stale_seconds: int = 120,
) -> dict[str, Any]:
    """Parse Kalshi YES/NO orderbook into pricing summary."""
    now = fetched_at or time.time()
    ob = _unwrap_orderbook(orderbook)
    yes_levels = ob.get("yes") or ob.get("yes_dollars") or []
    no_levels = ob.get("no") or ob.get("no_dollars") or []

    best_yes_bid, yes_bid_liq = _best_level(yes_levels, side="bid")
    best_no_bid, no_bid_liq = _best_level(no_levels, side="bid")

    best_yes_ask = (100.0 - best_no_bid) if best_no_bid is not None else None
    best_no_ask = (100.0 - best_yes_bid) if best_yes_bid is not None else None

    age = ob.get("age_seconds")
    if age is None and ob.get("timestamp"):
        try:
            age = now - float(ob["timestamp"])
        except (TypeError, ValueError):
            age = None

    return _pricing_from_sides(
        ticker,
        best_yes_bid=best_yes_bid,
        best_yes_ask=best_yes_ask,
        best_no_bid=best_no_bid,
        best_no_ask=best_no_ask,
        yes_bid_liq=yes_bid_liq,
        no_bid_liq=no_bid_liq,
        fetched_at=now,
        stale_seconds=stale_seconds,
        age=age,
    )


def parse_market_quotes(
    ticker: str,
    market_payload: dict | None,
    *,
    fetched_at: float | None = None,
    stale_seconds: int = 120,
) -> dict[str, Any]:
    """Build pricing from GET /markets/{ticker} bid/ask fields (fallback when orderbook empty)."""
    m = market_payload or {}
    if "market" in m:
        m = m["market"]

    yes_bid = m.get("yes_bid")
    yes_ask = m.get("yes_ask")
    no_bid = m.get("no_bid")
    no_ask = m.get("no_ask")

    if yes_bid is None:
        yes_bid = _price_to_cents(m.get("yes_bid_dollars"))
    if yes_ask is None:
        yes_ask = _price_to_cents(m.get("yes_ask_dollars"))
    if no_bid is None:
        no_bid = _price_to_cents(m.get("no_bid_dollars"))
    if no_ask is None:
        no_ask = _price_to_cents(m.get("no_ask_dollars"))

    yes_bid_liq = _qty_from_level(m.get("yes_bid_size_fp") or m.get("yes_bid_size") or 0)
    no_bid_liq = _qty_from_level(m.get("no_bid_size_fp") or m.get("no_bid_size") or 0)

    return _pricing_from_sides(
        ticker,
        best_yes_bid=yes_bid,
        best_yes_ask=yes_ask,
        best_no_bid=no_bid,
        best_no_ask=no_ask,
        yes_bid_liq=yes_bid_liq,
        no_bid_liq=no_bid_liq,
        fetched_at=fetched_at,
        stale_seconds=stale_seconds,
    )


def implied_prob_for_side(pricing: dict, side: str) -> float | None:
    """YES side uses implied_probability; NO uses complement."""
    p_yes = pricing.get("implied_probability")
    if p_yes is None:
        return None
    side = side.lower()
    if side == "yes":
        return p_yes
    return clamp_prob(1.0 - p_yes)


def executable_price(pricing: dict, side: str, action: str = "buy") -> float | None:
    """Return price in cents for limit order."""
    action = action.lower()
    side = side.lower()
    if action == "buy":
        if side == "yes":
            return pricing.get("best_yes_ask")
        return pricing.get("best_no_ask")
    if side == "yes":
        return pricing.get("best_yes_bid")
    return pricing.get("best_no_bid")
