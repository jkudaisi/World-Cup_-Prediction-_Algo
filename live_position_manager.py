"""Live position management: enter, exit on model drop."""

from __future__ import annotations

import logging
from typing import Any

from edge_engine import evaluate_edge
from kalshi_client import KalshiClient
from live_trader import (
    close_live_position_by_hedge,
    close_live_position_settlement,
    fetch_ticker_pricing,
    kalshi_settled_yes_won,
    open_live_positions,
    open_live_positions_for_fixture,
)
from entry_guards import block_live_entry
from position_outcomes import evaluate_position_outcome, side_won
from position_exits import (
    SCAN_ENTRY_MARKETS as ENTRY_MARKETS,
    build_marks_for_positions,
    market_cents_from_opp as _market_cents_from_opp,
    model_for_market as _model_for_market,
    live_scan_model_p,
    model_probability_drop,
    opposite_side_price_cents,
    ranked_opportunities as _ranked_opportunities,
)
from paper_trader import side_probability
from trade_executor import execute_order
from trade_logger import log_decision
from trading_config import can_place_live_orders, get_config

log = logging.getLogger(__name__)


def build_live_marks_from_fixtures(fixtures: list[dict]) -> dict[str, dict]:
    return build_marks_for_positions(
        fixtures,
        open_live_positions(),
        id_key="position_id",
    )


def _fixture_by_key(fixtures: list[dict], fixture_key: str, home: str, away: str) -> dict | None:
    for fx in fixtures:
        if fixture_key and fx.get("fixture_key") == fixture_key:
            return fx
        if fx.get("home") == home and fx.get("away") == away:
            return fx
    return None


def settle_decided_live_positions(
    fixtures: list[dict],
    *,
    kalshi_client: KalshiClient | None = None,
) -> list[dict]:
    """Close open live positions when match outcome or Kalshi settlement is known."""
    if not open_live_positions():
        return []

    cli = kalshi_client
    closed: list[dict] = []

    for pos in list(open_live_positions()):
        pid = pos["position_id"]
        fixture = _fixture_by_key(
            fixtures,
            pos.get("fixture_key", ""),
            pos.get("home", ""),
            pos.get("away", ""),
        )

        outcome = evaluate_position_outcome(pos, fixture)
        if outcome is not None:
            result = close_live_position_settlement(
                pid,
                won=outcome["won"],
                reason=outcome["reason"],
                yes_won=outcome["yes_won"],
            )
            if result.get("status") == "closed":
                closed.append(result)
                log_decision(
                    fixture=f"{pos.get('home')} vs {pos.get('away')}",
                    market=pos.get("market_type", ""),
                    ticker=pos.get("ticker", ""),
                    model_probability=None,
                    kalshi_probability=None,
                    edge=None,
                    confidence=pos.get("confidence"),
                    spread=None,
                    liquidity=None,
                    decision="EXIT",
                    reason=outcome["reason"],
                    risk_approval=True,
                    order_info=result.get("position"),
                    extra={
                        "mode": "live",
                        "exit_method": "settlement",
                        "pnl": result.get("pnl"),
                        "score": outcome.get("score"),
                    },
                )
            continue

        if cli and pos.get("ticker"):
            settled = kalshi_settled_yes_won(cli, pos["ticker"])
            if settled is not None:
                side = (pos.get("side") or "yes").lower()
                won = side_won(side=side, yes_won=settled)
                result = close_live_position_settlement(
                    pid,
                    won=won,
                    reason="kalshi_settled",
                    yes_won=settled,
                )
                if result.get("status") == "closed":
                    closed.append(result)

    return closed


def exit_stale_live_positions(
    fixtures: list[dict],
    *,
    kalshi_client: KalshiClient | None = None,
) -> list[dict]:
    """
    Exit open live positions when model probability for the held outcome drops
    beyond paper_model_drop_exit. Hedges by buying the opposite side on Kalshi.
    """
    if not open_live_positions():
        return []
    if not can_place_live_orders():
        return []

    cfg = get_config()
    marks = build_live_marks_from_fixtures(fixtures)
    closed: list[dict] = []

    for pos in open_live_positions():
        u = marks.get(pos["position_id"])
        if not u or u.get("model_probability") is None:
            continue

        model_yes = float(u["model_probability"])
        market_cents = u.get("market_cents")
        side = (pos.get("side") or "yes").lower()
        drop = model_probability_drop(pos, model_yes)

        if drop < cfg.paper_model_drop_exit:
            continue

        opposite_cents = opposite_side_price_cents(side, market_cents, model_yes)
        if kalshi_client:
            pricing = fetch_ticker_pricing(kalshi_client, pos.get("ticker", ""))
            if pricing:
                from market_pricing import executable_price
                fresh = executable_price(pricing, "no" if side == "yes" else "yes", "buy")
                if fresh is not None:
                    opposite_cents = float(fresh)

        result = close_live_position_by_hedge(
            pos["position_id"],
            opposite_price_cents=opposite_cents,
            client=kalshi_client,
            current_model_probability=model_yes,
            current_market_cents=market_cents,
            reason="model_reversal",
        )
        if result.get("status") != "closed":
            if result.get("status") == "error":
                fixture = _fixture_by_key(
                    fixtures,
                    pos.get("fixture_key", ""),
                    pos.get("home", ""),
                    pos.get("away", ""),
                )
                outcome = evaluate_position_outcome(pos, fixture)
                if outcome is not None:
                    settle_result = close_live_position_settlement(
                        pos["position_id"],
                        won=outcome["won"],
                        reason=outcome["reason"],
                        yes_won=outcome["yes_won"],
                    )
                    if settle_result.get("status") == "closed":
                        closed.append(settle_result)
            continue

        closed.append(result)
        current_side_p = side_probability(model_yes, side)
        entry_side_p = float(pos.get("model_probability_at_entry") or pos.get("model_probability") or 0.5)
        log.info(
            "Live exit (model_reversal) %s vs %s: model dropped %.1f%% (entry %.1f%% -> %.1f%%)",
            pos.get("home"),
            pos.get("away"),
            drop * 100,
            entry_side_p * 100,
            current_side_p * 100,
        )
        log_decision(
            fixture=f"{pos.get('home')} vs {pos.get('away')}",
            market=pos.get("market_type", ""),
            ticker=pos.get("ticker", ""),
            model_probability=current_side_p,
            kalshi_probability=(market_cents / 100.0) if market_cents is not None else None,
            edge=None,
            confidence=pos.get("confidence"),
            spread=None,
            liquidity=None,
            decision="EXIT",
            reason="model_reversal",
            risk_approval=True,
            order_info=result.get("position"),
            extra={
                "mode": "live",
                "exit_method": "opposite_side_hedge",
                "hedge_price_cents": opposite_cents,
                "model_drop": round(drop, 4),
                "pnl": result.get("pnl"),
                "hedge_order_id": (result.get("hedge_order") or {}).get("order_id"),
            },
        )

    return closed


def _try_enter_fixture_live(
    fixture: dict,
    *,
    kalshi_client: KalshiClient | None = None,
    only_live: bool = True,
) -> list[dict]:
    """Place live Kalshi orders for ranked TRADE signals (up to max per match)."""
    cfg = get_config()
    if not can_place_live_orders():
        return []
    if only_live and not fixture.get("live"):
        return []

    fkey = fixture.get("fixture_key") or f"{fixture.get('home')}|{fixture.get('away')}"
    entered: list[dict] = []

    opps = fixture.get("opportunities") or []
    candidates = [o for o in opps if o.get("market_type") in ENTRY_MARKETS and o.get("ticker")]
    ranked = _ranked_opportunities(candidates, trade_only=True)
    if not ranked:
        return []

    held = {
        (p.get("market_type"), (p.get("side") or "yes").lower())
        for p in open_live_positions_for_fixture(fkey)
    }

    for best in ranked:
        if len(open_live_positions_for_fixture(fkey)) >= cfg.max_trades_per_match:
            break

        mt = best["market_type"]
        ticker = str(best.get("ticker") or "")
        if not ticker or ticker.startswith("PAPER|"):
            continue

        model_p = live_scan_model_p(fixture, mt, best)
        market_cents = _market_cents_from_opp(best)
        if market_cents is None:
            continue
        kalshi_p = market_cents / 100.0

        edge_result = evaluate_edge(
            model_probability=model_p,
            market_implied_probability=kalshi_p,
            confidence=float(best.get("confidence") or fixture.get("confidence") or 0.5),
            spread=best.get("spread"),
            liquidity=best.get("liquidity") or 0,
            market_type=mt,
            live=bool(fixture.get("live")),
            mapping_confidence=float(best.get("mapping_confidence") or 1.0),
        )
        if edge_result["decision"] != "TRADE":
            continue

        side = (edge_result.get("side") or "yes").lower()
        if (mt, side) in held:
            continue

        block_reason = block_live_entry(
            market_type=mt,
            score_home=int(fixture.get("score_home") or 0),
            score_away=int(fixture.get("score_away") or 0),
            match_final=bool(fixture.get("match_final")),
            model_yes=model_p,
            kalshi_yes=kalshi_p,
            spread=best.get("spread"),
            is_live=bool(fixture.get("live")),
        )
        if block_reason:
            log.debug(
                "Live entry blocked %s vs %s %s: %s",
                fixture.get("home"),
                fixture.get("away"),
                mt,
                block_reason,
            )
            continue

        result = execute_order(
            ticker=ticker,
            side=side,
            fixture_key=fkey,
            home=fixture.get("home", ""),
            away=fixture.get("away", ""),
            market_type=mt,
            model_probability=model_p,
            confidence=float(best.get("confidence") or fixture.get("confidence") or 0.5),
            live=bool(fixture.get("live")),
            mapping_confidence=float(best.get("mapping_confidence") or 1.0),
            client=kalshi_client,
            force_paper=False,
            fixture=fixture,
        )
        if result.get("status") in ("submitted", "filled") and result.get("mode") == "live":
            log.info(
                "Live entry %s vs %s %s %s",
                fixture.get("home"),
                fixture.get("away"),
                mt,
                side,
            )
            entered.append(result)
            held.add((mt, side))

    return entered


def enter_live_for_fixtures(
    fixtures: list[dict],
    *,
    kalshi_client: KalshiClient | None = None,
    only_live: bool = True,
) -> list[dict]:
    entered: list[dict] = []
    for fx in fixtures:
        if only_live and not fx.get("live"):
            continue
        entered.extend(
            _try_enter_fixture_live(fx, kalshi_client=kalshi_client, only_live=only_live),
        )
    return entered


def run_live_cycle(
    fixtures: list[dict],
    *,
    kalshi_client: KalshiClient | None = None,
) -> dict[str, Any]:
    """Settle decided positions, exit stale ones, enter new opportunities."""
    settled = settle_decided_live_positions(fixtures, kalshi_client=kalshi_client)
    stale_closed = exit_stale_live_positions(fixtures, kalshi_client=kalshi_client)
    entered = enter_live_for_fixtures(fixtures, kalshi_client=kalshi_client, only_live=True)
    return {
        "settled_exits": len(settled),
        "stale_exits": len(stale_closed),
        "entered": len(entered),
        "settled_positions": settled,
        "stale_closed_positions": stale_closed,
        "entered_orders": entered,
    }


def run_live_exit_scan(
    fixtures: list[dict],
    *,
    kalshi_client: KalshiClient | None = None,
) -> dict[str, Any]:
    """Run settlement and stale-position exits for live trading."""
    settled = settle_decided_live_positions(fixtures, kalshi_client=kalshi_client)
    stale_closed = exit_stale_live_positions(fixtures, kalshi_client=kalshi_client)
    return {
        "settled_exits": len(settled),
        "stale_exits": len(stale_closed),
        "settled_positions": settled,
        "stale_closed_positions": stale_closed,
    }
