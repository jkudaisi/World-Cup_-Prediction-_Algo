"""Active paper trading: enter per match, mark-to-market, exit on model drop."""

from __future__ import annotations

import logging
from typing import Any

from entry_guards import block_live_entry
from edge_engine import evaluate_edge
from market_pricing import executable_price, parse_orderbook
from paper_trader import (
    close_paper_trade,
    close_paper_trade_by_hedge,
    current_bankroll,
    mark_open_trades,
    open_trades,
    open_trades_for_fixture,
    side_probability,
    simulate_fill,
    synthetic_paper_ticker,
)
from position_exits import (
    SCAN_ENTRY_MARKETS as ENTRY_MARKETS,
    build_marks_for_positions,
    entry_side_probability,
    market_cents_from_opp,
    market_cents_from_opp as _market_cents_from_opp,
    model_for_market,
    model_for_market as _model_for_market,
    live_scan_model_p,
    model_probability_drop,
    opposite_side_price_cents,
    ranked_opportunities as _ranked_opportunities,
)
from risk_manager import evaluate_risk, kelly_stake, stake_to_contracts
from trade_logger import log_decision
from trading_config import can_place_live_orders, get_config

log = logging.getLogger(__name__)


def _model_for_market(fixture: dict, market_type: str) -> float | None:
    return model_for_market(fixture, market_type)


def _market_cents_from_opp(opp: dict) -> float | None:
    return market_cents_from_opp(opp)


def build_marks_from_fixtures(fixtures: list[dict]) -> dict[str, dict]:
    """Map trade_id -> {model_probability, market_cents} from latest scan."""
    return build_marks_for_positions(fixtures, open_trades(), id_key="trade_id")


def update_paper_marks(fixtures: list[dict]) -> int:
    return mark_open_trades(build_marks_from_fixtures(fixtures))


def _entry_side_probability(trade: dict) -> float:
    return entry_side_probability(trade)


def exit_stale_positions(fixtures: list[dict]) -> list[dict]:
    """
    Exit open positions when model probability for the held outcome drops
    beyond paper_model_drop_exit. Hedges by simulating an opposite-side buy.
    """
    cfg = get_config()
    marks = build_marks_from_fixtures(fixtures)
    closed: list[dict] = []

    for t in open_trades():
        u = marks.get(t["trade_id"])
        if not u or u.get("model_probability") is None:
            continue

        model_yes = float(u["model_probability"])
        market_cents = u.get("market_cents")
        side = (t.get("side") or "yes").lower()
        entry_side_p = _entry_side_probability(t)
        current_side_p = side_probability(model_yes, side)
        drop = model_probability_drop(t, model_yes)

        if drop < cfg.paper_model_drop_exit:
            continue

        opposite_cents = opposite_side_price_cents(side, market_cents, model_yes)
        result = close_paper_trade_by_hedge(
            t["trade_id"],
            opposite_price_cents=opposite_cents,
            current_model_probability=model_yes,
            current_market_cents=market_cents,
            reason="model_reversal",
        )
        if result.get("status") != "closed":
            continue

        closed.append(result)
        log.info(
            "Paper exit (model_reversal) %s vs %s: model dropped %.1f%% (entry %.1f%% -> %.1f%%)",
            t.get("home"),
            t.get("away"),
            drop * 100,
            entry_side_p * 100,
            current_side_p * 100,
        )
        log_decision(
            fixture=f"{t.get('home')} vs {t.get('away')}",
            market=t.get("market_type", ""),
            ticker=t.get("ticker", ""),
            model_probability=current_side_p,
            kalshi_probability=(market_cents / 100.0) if market_cents is not None else None,
            edge=None,
            confidence=t.get("confidence"),
            spread=None,
            liquidity=None,
            decision="EXIT",
            reason="model_reversal",
            risk_approval=True,
            order_info=result.get("trade"),
            extra={
                "exit_method": "opposite_side_hedge",
                "hedge_price_cents": opposite_cents,
                "model_drop": round(drop, 4),
                "pnl": result.get("pnl"),
            },
        )

    return closed


def _should_exit(trade: dict, model_yes: float, market_cents: float | None) -> tuple[bool, str]:
    cfg = get_config()
    side = (trade.get("side") or "yes").lower()
    current_side_p = side_probability(model_yes, side)

    if market_cents is not None:
        mkt_side_p = (market_cents / 100.0) if side == "yes" else (1.0 - market_cents / 100.0)
        if current_side_p + cfg.paper_exit_edge_reversal < mkt_side_p:
            return True, "Edge reversed vs market"

    return False, ""


def manage_paper_exits(fixtures: list[dict]) -> list[dict]:
    """Close paper positions when edge reverses vs market."""
    marks = build_marks_from_fixtures(fixtures)
    closed: list[dict] = []
    for t in open_trades():
        u = marks.get(t["trade_id"])
        if not u or u.get("model_probability") is None:
            continue
        model_yes = float(u["model_probability"])
        market_cents = u.get("market_cents")
        should, reason = _should_exit(t, model_yes, market_cents)
        if not should:
            continue
        side = (t.get("side") or "yes").lower()
        if market_cents is not None:
            exit_cents = float(market_cents) if side == "yes" else (100.0 - float(market_cents))
        else:
            exit_cents = side_probability(model_yes, side) * 100.0
        exit_cents = max(1.0, min(99.0, exit_cents))
        result = close_paper_trade(
            t["trade_id"],
            exit_price_cents=exit_cents,
            current_model_probability=model_yes,
            current_market_cents=market_cents,
            reason=reason,
        )
        if result.get("status") == "closed":
            closed.append(result)
            log.info("Paper exit %s vs %s: %s", t.get("home"), t.get("away"), reason)
    return closed


def _paper_entry_price(model_p: float, market_cents: float | None, side: str) -> float:
    cfg = get_config()
    side = side.lower()
    if market_cents is not None:
        return float(market_cents) if side == "yes" else (100.0 - float(market_cents))
    if cfg.paper_use_model_pricing:
        fair = side_probability(model_p, side) * 100.0
        return max(1.0, min(99.0, round(fair, 1)))
    return max(1.0, min(99.0, round(model_p * 100.0, 1)))


def _try_enter_fixture(
    fixture: dict,
    *,
    kalshi_client=None,
    only_live: bool = False,
) -> dict | None:
    cfg = get_config()
    if only_live and not fixture.get("live"):
        return None

    fkey = fixture.get("fixture_key") or f"{fixture.get('home')}|{fixture.get('away')}"
    existing = open_trades_for_fixture(fkey)
    if len(existing) >= cfg.paper_trades_per_match:
        return None

    opps = fixture.get("opportunities") or []
    candidates = [o for o in opps if o.get("market_type") in ENTRY_MARKETS]
    ranked = _ranked_opportunities(candidates, trade_only=True)
    if not ranked and cfg.paper_aggressive and fixture.get("live"):
        ranked = _ranked_opportunities(candidates, trade_only=False)
    if not ranked:
        return None

    held = {(t.get("market_type"), (t.get("side") or "yes").lower()) for t in existing}

    for best in ranked:
        mt = best["market_type"]
        model_p = live_scan_model_p(fixture, mt, best)
        market_cents = _market_cents_from_opp(best)
        kalshi_p = (market_cents / 100.0) if market_cents is not None else model_p
        spread = best.get("spread")
        liquidity = best.get("liquidity") or 0

        edge_result = evaluate_edge(
            model_probability=model_p,
            market_implied_probability=kalshi_p,
            confidence=float(best.get("confidence") or fixture.get("confidence") or 0.5),
            spread=spread,
            liquidity=liquidity,
            market_type=mt,
            live=bool(fixture.get("live")),
            paper_mode=True,
            mapping_confidence=float(best.get("mapping_confidence") or 1.0),
        )

        if edge_result["decision"] != "TRADE":
            if not (
                cfg.paper_aggressive
                and fixture.get("live")
                and model_p >= cfg.paper_aggressive_min_model_prob
            ):
                continue
            raw_edge = abs(model_p - kalshi_p)
            if raw_edge < max(cfg.paper_min_edge, cfg.paper_aggressive_min_edge_floor):
                continue
            side = "yes" if model_p >= 0.5 else "no"
            edge_result = {
                **edge_result,
                "decision": "TRADE",
                "side": side,
                "edge": abs(model_p - kalshi_p),
                "model_probability": model_p if side == "yes" else (1.0 - model_p),
            }
        else:
            side = edge_result.get("side") or "yes"

        side = (side or "yes").lower()
        if (mt, side) in held:
            continue

        block_reason = block_live_entry(
            market_type=mt,
            score_home=int(fixture.get("score_home") or 0),
            score_away=int(fixture.get("score_away") or 0),
            match_final=bool(fixture.get("match_final")),
            model_yes=model_p,
            kalshi_yes=kalshi_p,
            spread=spread,
            is_live=bool(fixture.get("live")),
        )
        if block_reason:
            continue

        ticker = best.get("ticker") or synthetic_paper_ticker(
            fixture.get("home", ""), fixture.get("away", ""), mt,
        )

        if kalshi_client and ticker and not str(ticker).startswith("PAPER|"):
            try:
                raw_ob = kalshi_client.get_orderbook(ticker)
                pricing = parse_orderbook(ticker, raw_ob, stale_seconds=cfg.stale_price_seconds)
                px = executable_price(pricing, side, "buy")
                if px is not None:
                    market_cents = float(px) if side == "yes" else (100.0 - float(px))
                    kalshi_p = pricing.get("implied_probability") or kalshi_p
            except Exception:
                pass

        entry_cents = _paper_entry_price(model_p, market_cents, side)
        trade_model_p = float(edge_result["model_probability"])
        trade_edge = float(edge_result.get("edge") or 0.0)
        conf = float(best.get("confidence") or fixture.get("confidence") or 0.5)
        br = current_bankroll()
        target_stake = kelly_stake(br, trade_edge, trade_model_p, conf)
        count = stake_to_contracts(target_stake, entry_cents)
        if count <= 0:
            continue

        stake = round(count * entry_cents / 100.0, 2)
        risk = evaluate_risk(
            stake=stake,
            fixture_key=fkey,
            bankroll=br,
            edge=trade_edge,
            model_p=trade_model_p,
            confidence=conf,
        )
        if not risk["approved"]:
            continue

        fill = simulate_fill(
            ticker=ticker,
            side=side,
            count=count,
            entry_price_cents=entry_cents,
            model_probability=edge_result["model_probability"],
            market_probability=kalshi_p,
            edge=edge_result.get("edge") or 0.0,
            confidence=float(best.get("confidence") or 0.5),
            fixture_key=fkey,
            home=fixture.get("home", ""),
            away=fixture.get("away", ""),
            market_type=mt,
            live=bool(fixture.get("live")),
        )
        if fill.get("status") == "filled":
            return fill
    return None


def enter_paper_for_fixtures(
    fixtures: list[dict],
    *,
    kalshi_client=None,
    only_live: bool = True,
) -> list[dict]:
    """Open paper trades on live (or all) fixtures — up to paper_trades_per_match each."""
    cfg = get_config()
    entered: list[dict] = []
    for fx in fixtures:
        if only_live and not fx.get("live"):
            continue
        fkey = fx.get("fixture_key") or f"{fx.get('home')}|{fx.get('away')}"
        while len(open_trades_for_fixture(fkey)) < cfg.paper_trades_per_match:
            result = _try_enter_fixture(fx, kalshi_client=kalshi_client, only_live=only_live)
            if not result:
                break
            entered.append(result)
    return entered


def run_paper_cycle(
    fixtures: list[dict],
    *,
    kalshi_client=None,
) -> dict[str, Any]:
    """Mark open positions, exit stale/reversed, enter new live matches."""
    from live_position_manager import exit_stale_live_positions

    stale_closed = exit_stale_positions(fixtures)
    live_stale_closed = exit_stale_live_positions(fixtures, kalshi_client=kalshi_client)
    closed = manage_paper_exits(fixtures)
    entered = (
        []
        if can_place_live_orders()
        else enter_paper_for_fixtures(fixtures, kalshi_client=kalshi_client, only_live=True)
    )
    marks = update_paper_marks(fixtures)
    return {
        "marked": marks,
        "stale_exits": len(stale_closed),
        "live_stale_exits": len(live_stale_closed),
        "closed": len(closed) + len(stale_closed) + len(live_stale_closed),
        "entered": len(entered),
        "stale_closed_trades": stale_closed,
        "live_stale_closed_positions": live_stale_closed,
        "closed_trades": closed,
        "entered_trades": entered,
    }
