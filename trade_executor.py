"""Live trading executor with strict safeguards."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from edge_engine import evaluate_edge
from entry_guards import block_live_entry
from kalshi_client import KalshiClient, KalshiClientError
from market_pricing import executable_price, parse_market_quotes, parse_orderbook
from paper_trader import is_duplicate_trade, register_duplicate
from risk_manager import evaluate_risk, kelly_stake, stake_to_contracts
from trade_logger import log_decision, log_order, log_result
from trading_config import can_place_live_orders, get_config

log = logging.getLogger(__name__)

_recent_live: set[str] = set()


def _dup_key(ticker: str, side: str, fixture_key: str) -> str:
    return f"live|{fixture_key}|{ticker}|{side}"


def execute_order(
    *,
    ticker: str,
    side: str,
    count: int | None = None,
    fixture_key: str,
    home: str,
    away: str,
    market_type: str,
    model_probability: float,
    confidence: float,
    live: bool = False,
    match_status: str = "NS",
    mapping_confidence: float = 1.0,
    client: KalshiClient | None = None,
    force_paper: bool = False,
    fixture: dict | None = None,
) -> dict[str, Any]:
    """Full order pipeline: edge → risk → price recheck → limit order."""
    cfg = get_config()
    cli = client or KalshiClient()

    score_home = int((fixture or {}).get("score_home") or 0)
    score_away = int((fixture or {}).get("score_away") or 0)
    match_final = bool((fixture or {}).get("match_final"))

    if fixture and (live or score_home > 0 or score_away > 0):
        from position_exits import model_for_market

        live_model = model_for_market(fixture, market_type)
        if live_model is not None:
            model_probability = float(live_model)

    # Fetch fresh pricing (orderbook_fp or market quotes)
    pricing = None
    try:
        raw_ob = cli.get_orderbook(ticker)
        pricing = parse_orderbook(ticker, raw_ob, stale_seconds=cfg.stale_price_seconds)
        if pricing.get("implied_probability") is None:
            pricing = None
    except KalshiClientError:
        pricing = None

    if pricing is None:
        try:
            mk = cli.get_market(ticker)
            pricing = parse_market_quotes(ticker, mk, stale_seconds=cfg.stale_price_seconds)
        except KalshiClientError as exc:
            entry = log_decision(
                fixture=f"{home} vs {away}",
                market=market_type,
                ticker=ticker,
                model_probability=model_probability,
                kalshi_probability=None,
                edge=None,
                confidence=confidence,
                spread=None,
                liquidity=None,
                decision="SKIP",
                reason=f"SKIP: Orderbook unavailable ({exc})",
                risk_approval=False,
            )
            return {"status": "skipped", "reason": str(exc), "log": entry}

    if pricing.get("implied_probability") is None:
        entry = log_decision(
            fixture=f"{home} vs {away}",
            market=market_type,
            ticker=ticker,
            model_probability=model_probability,
            kalshi_probability=None,
            edge=None,
            confidence=confidence,
            spread=None,
            liquidity=None,
            decision="SKIP",
            reason="SKIP: No Kalshi price on book",
            risk_approval=False,
        )
        return {"status": "skipped", "reason": "No Kalshi price", "log": entry}

    market_p = pricing.get("implied_probability")
    spread = pricing.get("spread")
    liquidity = pricing.get("available_liquidity", 0)

    edge_result = evaluate_edge(
        model_probability=model_probability,
        market_implied_probability=market_p or 0.5,
        confidence=confidence,
        spread=spread,
        liquidity=liquidity,
        market_type=market_type,
        match_status=match_status,
        live=live,
        stale=pricing.get("stale_price_warning", False),
        mapping_confidence=mapping_confidence,
    )

    if edge_result["decision"] != "TRADE":
        entry = log_decision(
            fixture=f"{home} vs {away}",
            market=market_type,
            ticker=ticker,
            model_probability=edge_result["model_probability"],
            kalshi_probability=edge_result["market_probability"],
            edge=edge_result["edge"],
            confidence=confidence,
            spread=spread,
            liquidity=liquidity,
            decision="SKIP",
            reason=edge_result["reason"],
            risk_approval=False,
        )
        return {"status": "skipped", "edge": edge_result, "log": entry}

    block_reason = block_live_entry(
        market_type=market_type,
        score_home=score_home,
        score_away=score_away,
        match_final=match_final,
        model_yes=float(model_probability),
        kalshi_yes=float(market_p) if market_p is not None else None,
        spread=spread,
        is_live=live,
    )
    if block_reason:
        entry = log_decision(
            fixture=f"{home} vs {away}",
            market=market_type,
            ticker=ticker,
            model_probability=edge_result["model_probability"],
            kalshi_probability=edge_result["market_probability"],
            edge=edge_result["edge"],
            confidence=confidence,
            spread=spread,
            liquidity=liquidity,
            decision="SKIP",
            reason=f"SKIP: {block_reason}",
            risk_approval=False,
        )
        return {"status": "skipped", "reason": block_reason, "log": entry}

    trade_side = edge_result["side"]
    entry_cents = executable_price(pricing, trade_side, "buy")
    if entry_cents is None:
        entry = log_decision(
            fixture=f"{home} vs {away}",
            market=market_type,
            ticker=ticker,
            model_probability=edge_result["model_probability"],
            kalshi_probability=edge_result["market_probability"],
            edge=edge_result["edge"],
            confidence=confidence,
            spread=spread,
            liquidity=liquidity,
            decision="SKIP",
            reason="SKIP: No executable price on book",
            risk_approval=False,
        )
        return {"status": "skipped", "reason": "No executable price", "log": entry}

    trade_model_p = float(edge_result["model_probability"])
    trade_edge = float(edge_result["edge"])
    from kalshi_account import resolve_bankroll
    br = resolve_bankroll(client)
    target_stake = kelly_stake(br, trade_edge, trade_model_p, confidence)
    if count is None:
        count = stake_to_contracts(target_stake, entry_cents)
    if count <= 0:
        entry = log_decision(
            fixture=f"{home} vs {away}",
            market=market_type,
            ticker=ticker,
            model_probability=edge_result["model_probability"],
            kalshi_probability=edge_result["market_probability"],
            edge=edge_result["edge"],
            confidence=confidence,
            spread=spread,
            liquidity=liquidity,
            decision="SKIP",
            reason="SKIP: Kelly stake below minimum contract size",
            risk_approval=False,
        )
        return {"status": "skipped", "reason": "Kelly stake too small", "log": entry}

    stake = round(count * entry_cents / 100.0, 2)
    risk = evaluate_risk(
        stake=stake,
        fixture_key=fixture_key,
        bankroll=br,
        edge=trade_edge,
        model_p=trade_model_p,
        confidence=confidence,
        client=client,
    )
    if not risk["approved"]:
        entry = log_decision(
            fixture=f"{home} vs {away}",
            market=market_type,
            ticker=ticker,
            model_probability=edge_result["model_probability"],
            kalshi_probability=edge_result["market_probability"],
            edge=edge_result["edge"],
            confidence=confidence,
            spread=spread,
            liquidity=liquidity,
            decision="SKIP",
            reason=f"SKIP: {risk['reason']}",
            risk_approval=False,
        )
        return {"status": "skipped", "risk": risk, "log": entry}

    dup = _dup_key(ticker, trade_side, fixture_key)
    if dup in _recent_live or is_duplicate_trade(ticker, trade_side, fixture_key):
        entry = log_decision(
            fixture=f"{home} vs {away}",
            market=market_type,
            ticker=ticker,
            model_probability=edge_result["model_probability"],
            kalshi_probability=edge_result["market_probability"],
            edge=edge_result["edge"],
            confidence=confidence,
            spread=spread,
            liquidity=liquidity,
            decision="SKIP",
            reason="SKIP: Duplicate trade protection",
            risk_approval=True,
        )
        return {"status": "skipped", "reason": "Duplicate", "log": entry}

    if cfg.require_manual_approval and not force_paper:
        entry = log_decision(
            fixture=f"{home} vs {away}",
            market=market_type,
            ticker=ticker,
            model_probability=edge_result["model_probability"],
            kalshi_probability=edge_result["market_probability"],
            edge=edge_result["edge"],
            confidence=confidence,
            spread=spread,
            liquidity=liquidity,
            decision="SKIP",
            reason="SKIP: Manual approval required",
            risk_approval=True,
        )
        return {"status": "pending_approval", "edge": edge_result, "log": entry}

    # Paper mode or dry run
    if force_paper or cfg.dry_run or not can_place_live_orders():
        from paper_trader import simulate_fill
        fill = simulate_fill(
            ticker=ticker,
            side=trade_side,
            count=count,
            entry_price_cents=entry_cents,
            model_probability=edge_result["model_probability"],
            market_probability=edge_result["market_probability"],
            edge=edge_result["edge"],
            confidence=confidence,
            fixture_key=fixture_key,
            home=home,
            away=away,
            market_type=market_type,
            live=live,
        )
        entry = log_decision(
            fixture=f"{home} vs {away}",
            market=market_type,
            ticker=ticker,
            model_probability=edge_result["model_probability"],
            kalshi_probability=edge_result["market_probability"],
            edge=edge_result["edge"],
            confidence=confidence,
            spread=spread,
            liquidity=liquidity,
            decision="TRADE" if fill.get("status") == "filled" else "SKIP",
            reason=edge_result["reason"] if fill.get("status") == "filled" else fill.get("reason", ""),
            risk_approval=True,
            order_info=fill.get("trade"),
            extra={"mode": "paper"},
        )
        return {"status": fill.get("status", "skipped"), "fill": fill, "log": entry, "mode": "paper"}

    # Live limit order with slippage cap
    max_price = int(entry_cents + cfg.max_slippage_cents)
    order_kwargs: dict[str, Any] = {
        "ticker": ticker,
        "side": trade_side,
        "action": "buy",
        "count": count,
        "order_type": "limit",
        "client_order_id": str(uuid.uuid4()),
    }
    if trade_side == "yes":
        order_kwargs["yes_price"] = max_price
    else:
        order_kwargs["no_price"] = max_price

    try:
        resp = cli.create_order(**order_kwargs)
    except KalshiClientError as exc:
        entry = log_decision(
            fixture=f"{home} vs {away}",
            market=market_type,
            ticker=ticker,
            model_probability=edge_result["model_probability"],
            kalshi_probability=edge_result["market_probability"],
            edge=edge_result["edge"],
            confidence=confidence,
            spread=spread,
            liquidity=liquidity,
            decision="SKIP",
            reason=f"SKIP: Order failed ({exc})",
            risk_approval=True,
        )
        return {"status": "error", "error": str(exc), "log": entry}

    order = resp.get("order") or resp
    order_id = order.get("order_id")
    if order_id is not None and not isinstance(order_id, str):
        order_id = str(order_id)

    recorded_entry = float(entry_cents)
    avg_fill = order.get("average_fill_price") or resp.get("average_fill_price")
    if avg_fill is not None:
        try:
            fill_val = float(avg_fill)
            fill_cents = fill_val * 100.0 if fill_val <= 1.0 else fill_val
            if trade_side == "no":
                fill_cents = 100.0 - fill_cents
            recorded_entry = round(fill_cents, 2)
        except (TypeError, ValueError):
            pass

    log_order({
        "order_id": order_id,
        "ticker": ticker,
        "side": trade_side,
        "count": count,
        "limit_price_cents": max_price,
        "fixture_key": fixture_key,
        "home": home,
        "away": away,
        "market_type": market_type,
        "status": order.get("status", "submitted"),
        "dry_run": resp.get("dry_run", False),
    })
    _recent_live.add(dup)
    register_duplicate(ticker, trade_side, fixture_key)

    from live_trader import register_live_position

    register_live_position(
        order_id=str(order.get("order_id", "")),
        ticker=ticker,
        side=trade_side,
        count=count,
        entry_price_cents=recorded_entry,
        model_probability=float(edge_result["model_probability"]),
        market_probability=edge_result.get("market_probability"),
        edge=float(edge_result["edge"]),
        confidence=confidence,
        fixture_key=fixture_key,
        home=home,
        away=away,
        market_type=market_type,
        live=live,
    )

    entry = log_decision(
        fixture=f"{home} vs {away}",
        market=market_type,
        ticker=ticker,
        model_probability=edge_result["model_probability"],
        kalshi_probability=edge_result["market_probability"],
        edge=edge_result["edge"],
        confidence=confidence,
        spread=spread,
        liquidity=liquidity,
        decision="TRADE",
        reason=edge_result["reason"],
        risk_approval=True,
        order_info=order,
        extra={"mode": "live"},
    )
    log_result({"order_id": order.get("order_id"), "status": "submitted", "ticker": ticker})
    return {"status": "submitted", "order": order, "log": entry, "mode": "live"}
