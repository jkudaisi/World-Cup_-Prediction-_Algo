"""Edge calculation engine — compare model vs Kalshi implied probability."""

from __future__ import annotations

from typing import Any

from trading_config import get_config


def normalize_prob(p: float | None) -> float | None:
    if p is None:
        return None
    return round(min(1.0, max(0.0, float(p))), 4)


def evaluate_edge(
    *,
    model_probability: float,
    market_implied_probability: float,
    confidence: float,
    spread: float | None,
    liquidity: int | float,
    market_type: str,
    match_status: str = "NS",
    time_remaining: int | None = None,
    live: bool = False,
    stale: bool = False,
    mapping_confidence: float = 1.0,
    paper_mode: bool = False,
) -> dict[str, Any]:
    cfg = get_config()
    model_p = normalize_prob(model_probability) or 0.0
    market_p = normalize_prob(market_implied_probability) or 0.0
    conf = normalize_prob(confidence) or 0.0

    warnings: list[str] = []
    if paper_mode and cfg.paper_aggressive:
        min_edge = cfg.paper_min_edge
        min_conf = cfg.paper_min_confidence
    else:
        min_edge = cfg.min_edge_live if live else cfg.min_edge_prematch
        min_conf = cfg.min_confidence_live if live else cfg.min_confidence

    edge_yes = model_p - market_p
    edge_no = (1.0 - model_p) - (1.0 - market_p)

    if abs(edge_yes) >= abs(edge_no):
        side = "yes" if edge_yes > 0 else "no"
        edge = edge_yes if side == "yes" else -edge_yes
        trade_model_p = model_p if side == "yes" else (1.0 - model_p)
        trade_market_p = market_p if side == "yes" else (1.0 - market_p)
    else:
        side = "no" if edge_no > 0 else "yes"
        edge = edge_no if side == "no" else -edge_no
        trade_model_p = (1.0 - model_p) if side == "no" else model_p
        trade_market_p = (1.0 - market_p) if side == "no" else market_p

    edge = round(edge, 4)
    trade_model_p = normalize_prob(trade_model_p) or 0.0
    trade_market_p = normalize_prob(trade_market_p) or 0.0
    raw_edge = round(trade_model_p - trade_market_p, 4)

    should_trade = True
    reason_parts: list[str] = []

    if stale and not paper_mode:
        should_trade = False
        reason_parts.append("Market price is stale")

    if mapping_confidence < cfg.min_mapping_confidence and not paper_mode:
        should_trade = False
        reason_parts.append("Fixture not confidently mapped")

    if conf < min_conf:
        should_trade = False
        reason_parts.append(f"Model confidence too low ({conf:.0%} < {min_conf:.0%})")

    if not paper_mode:
        if spread is not None and spread > cfg.max_spread_cents:
            should_trade = False
            reason_parts.append(f"Spread too wide ({spread:.1f}¢ > {cfg.max_spread_cents:.0f}¢)")

        if liquidity < cfg.min_liquidity_contracts:
            should_trade = False
            reason_parts.append(f"Market too illiquid ({liquidity} < {cfg.min_liquidity_contracts} contracts)")

    if raw_edge < min_edge:
        should_trade = False
        reason_parts.append(f"Edge too small ({raw_edge:+.1%} < {min_edge:.0%} minimum)")

    if cfg.kill_switch:
        should_trade = False
        reason_parts.append("Kill switch active")

    if should_trade:
        reason = (
            f"TRADE: Model edge {raw_edge:+.1%}, confidence {conf:.0%}"
            + (", paper mode" if paper_mode else ", spread acceptable")
        )
        decision = "TRADE"
    else:
        decision = "SKIP"
        reason = "SKIP: " + (reason_parts[0] if reason_parts else "Edge too small")

    if live and time_remaining is not None and time_remaining < cfg.live_time_remaining_warn_minutes:
        warnings.append("Very little time remaining in match")

    return {
        "should_trade": should_trade,
        "decision": decision,
        "side": side,
        "edge": raw_edge,
        "edge_signed": edge,
        "model_probability": trade_model_p,
        "market_probability": trade_market_p,
        "reason": reason,
        "warnings": warnings,
        "market_type": market_type,
        "match_status": match_status,
        "live": live,
        "min_edge_required": min_edge,
    }
