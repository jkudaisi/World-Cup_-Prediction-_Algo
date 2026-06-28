"""Paper trading simulator with P/L tracking."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
PAPER_TRADES_PATH = ROOT / "data" / "paper_trades.json"
_lock = threading.Lock()

# In-memory duplicate protection for paper trades
_recent_keys: set[str] = set()


def _load() -> dict:
    PAPER_TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not PAPER_TRADES_PATH.exists():
        return {"trades": [], "bankroll": 20.0, "starting_bankroll": 20.0}
    with open(PAPER_TRADES_PATH, encoding="utf-8") as f:
        return json.load(f)


def _save(data: dict) -> None:
    with open(PAPER_TRADES_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _duplicate_key(ticker: str, side: str, fixture_key: str) -> str:
    return f"{fixture_key}|{ticker}|{side}"


def synthetic_paper_ticker(home: str, away: str, market_type: str) -> str:
    slug = f"PAPER|{home}|{away}|{market_type}".replace(" ", "_")
    return slug[:80]


def side_probability(yes_probability: float, side: str) -> float:
    side = (side or "yes").lower()
    p = float(yes_probability)
    return p if side == "yes" else round(1.0 - p, 4)


def is_duplicate_trade(ticker: str, side: str, fixture_key: str) -> bool:
    key = _duplicate_key(ticker, side, fixture_key)
    if key in _recent_keys:
        return True
    data = _load()
    for t in data.get("trades", []):
        if t.get("status") == "open" and t.get("duplicate_key") == key:
            return True
    return False


def register_duplicate(ticker: str, side: str, fixture_key: str) -> None:
    _recent_keys.add(_duplicate_key(ticker, side, fixture_key))


def current_bankroll() -> float:
    """Paper trading bankroll (falls back to config bankroll if unset)."""
    from trading_config import get_config
    data = _load()
    return float(data.get("bankroll", get_config().bankroll))


def simulate_fill(
    *,
    ticker: str,
    side: str,
    count: int,
    entry_price_cents: float,
    model_probability: float,
    market_probability: float,
    edge: float,
    confidence: float,
    fixture_key: str,
    home: str,
    away: str,
    market_type: str,
    live: bool = False,
) -> dict[str, Any]:
    """Open a paper trade with simulated fill at entry price."""
    if is_duplicate_trade(ticker, side, fixture_key):
        return {"status": "skipped", "reason": "Duplicate trade protection"}

    cost = round(count * entry_price_cents / 100.0, 2)
    trade_id = str(uuid.uuid4())
    dup_key = _duplicate_key(ticker, side, fixture_key)
    trade = {
        "trade_id": trade_id,
        "duplicate_key": dup_key,
        "ticker": ticker,
        "side": side,
        "count": count,
        "entry_price_cents": entry_price_cents,
        "closing_line_cents": entry_price_cents,
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
        "pnl": 0.0,
        "current_model_probability": model_probability,
        "current_market_cents": entry_price_cents,
        "unrealized_pnl": 0.0,
    }

    with _lock:
        data = _load()
        data.setdefault("trades", []).append(trade)
        data["bankroll"] = round(float(data.get("bankroll", 20)) - cost, 2)
        _save(data)
    register_duplicate(ticker, side, fixture_key)
    return {"status": "filled", "trade": trade}


def settle_trade(trade_id: str, *, won: bool, settlement_price_cents: float = 100.0) -> dict:
    """Settle paper trade — won pays $1/contract, lost pays $0."""
    with _lock:
        data = _load()
        for t in data.get("trades", []):
            if t.get("trade_id") != trade_id or t.get("status") != "open":
                continue
            count = int(t["count"])
            cost = float(t["cost"])
            payout = count * (settlement_price_cents / 100.0) if won else 0.0
            pnl = round(payout - cost, 2)
            t["status"] = "settled"
            t["won"] = won
            t["pnl"] = pnl
            t["settled_at"] = datetime.now(timezone.utc).isoformat()
            t["settlement_price_cents"] = settlement_price_cents
            data["bankroll"] = round(float(data.get("bankroll", 0)) + payout, 2)
            _save(data)
            return {"status": "settled", "trade": t}
    return {"status": "not_found", "trade_id": trade_id}


def close_paper_trade(
    trade_id: str,
    *,
    exit_price_cents: float,
    current_model_probability: float | None = None,
    current_market_cents: float | None = None,
    reason: str = "Manual close",
) -> dict[str, Any]:
    """Sell/close an open paper position at exit price (per contract, cents)."""
    with _lock:
        data = _load()
        for t in data.get("trades", []):
            if t.get("trade_id") != trade_id or t.get("status") != "open":
                continue
            count = int(t["count"])
            cost = float(t["cost"])
            side = (t.get("side") or "yes").lower()
            payout = count * (float(exit_price_cents) / 100.0)
            pnl = round(payout - cost, 2)
            t["status"] = "closed"
            t["closed_at"] = datetime.now(timezone.utc).isoformat()
            t["exit_price_cents"] = round(float(exit_price_cents), 2)
            t["exit_reason"] = reason
            t["pnl"] = pnl
            t["closing_line_cents"] = round(float(exit_price_cents), 2)
            if current_model_probability is not None:
                t["current_model_probability"] = round(float(current_model_probability), 4)
            if current_market_cents is not None:
                t["current_market_cents"] = round(float(current_market_cents), 2)
            t["unrealized_pnl"] = 0.0
            data["bankroll"] = round(float(data.get("bankroll", 0)) + payout, 2)
            key = t.get("duplicate_key")
            if key:
                _recent_keys.discard(key)
            _save(data)
            return {"status": "closed", "trade": t, "pnl": pnl}
    return {"status": "not_found", "trade_id": trade_id}


def close_paper_trade_by_hedge(
    trade_id: str,
    *,
    opposite_price_cents: float,
    current_model_probability: float | None = None,
    current_market_cents: float | None = None,
    reason: str = "model_reversal",
) -> dict[str, Any]:
    """
    Close an open position by simulating a buy of the opposite side at market.

    Locks in payout of $1/contract minus entry cost and hedge cost (partial loss).
    """
    with _lock:
        data = _load()
        for t in data.get("trades", []):
            if t.get("trade_id") != trade_id or t.get("status") != "open":
                continue
            count = int(t["count"])
            cost = float(t["cost"])
            side = (t.get("side") or "yes").lower()
            opposite_side = "no" if side == "yes" else "yes"
            hedge_cost = round(count * float(opposite_price_cents) / 100.0, 2)
            payout = float(count)  # YES + NO pair resolves to $1 per contract
            pnl = round(payout - cost - hedge_cost, 2)
            t["status"] = "closed"
            t["closed_at"] = datetime.now(timezone.utc).isoformat()
            t["exit_reason"] = reason
            t["exit_method"] = "opposite_side_hedge"
            t["hedge_side"] = opposite_side
            t["hedge_price_cents"] = round(float(opposite_price_cents), 2)
            t["pnl"] = pnl
            t["closing_line_cents"] = round(float(opposite_price_cents), 2)
            if current_model_probability is not None:
                t["current_model_probability"] = round(float(current_model_probability), 4)
            if current_market_cents is not None:
                t["current_market_cents"] = round(float(current_market_cents), 2)
            t["unrealized_pnl"] = 0.0
            data["bankroll"] = round(float(data.get("bankroll", 0)) - hedge_cost + payout, 2)
            key = t.get("duplicate_key")
            if key:
                _recent_keys.discard(key)
            _save(data)
            return {
                "status": "closed",
                "trade": t,
                "pnl": pnl,
                "hedge_cost": hedge_cost,
                "payout": payout,
            }
    return {"status": "not_found", "trade_id": trade_id}


def mark_open_trades(updates: dict[str, dict]) -> int:
    """Update open trades with current model/market prices. Key = trade_id."""
    if not updates:
        return 0
    with _lock:
        data = _load()
        n = 0
        for t in data.get("trades", []):
            if t.get("status") != "open":
                continue
            tid = t.get("trade_id")
            u = updates.get(tid)
            if not u:
                continue
            side = (t.get("side") or "yes").lower()
            entry_cents = float(t.get("entry_price_cents", 0))
            count = int(t.get("count", 1))
            model_yes = u.get("model_probability")
            market_cents = u.get("market_cents")
            if model_yes is not None:
                t["current_model_probability"] = round(float(model_yes), 4)
                side_p = side_probability(float(model_yes), side)
                t["current_side_probability"] = side_p
            if market_cents is not None:
                t["current_market_cents"] = round(float(market_cents), 2)
                mark_cents = float(market_cents) if side == "yes" else (100.0 - float(market_cents))
            elif model_yes is not None:
                mark_cents = side_probability(float(model_yes), side) * 100.0
                t["current_market_cents"] = round(mark_cents, 2)
            else:
                mark_cents = entry_cents
            t["unrealized_pnl"] = round((mark_cents - entry_cents) * count / 100.0, 2)
            t["closing_line_cents"] = round(mark_cents, 2)
            n += 1
        if n:
            _save(data)
        return n


def open_trades() -> list[dict]:
    return [t for t in _load().get("trades", []) if t.get("status") == "open"]


def open_trades_for_fixture(fixture_key: str) -> list[dict]:
    return [t for t in open_trades() if t.get("fixture_key") == fixture_key]


def paper_stats() -> dict[str, Any]:
    data = _load()
    trades = data.get("trades", [])
    settled = [t for t in trades if t.get("status") in ("settled", "closed")]
    open_trades_list = [t for t in trades if t.get("status") == "open"]
    starting = float(data.get("starting_bankroll", 20))
    bankroll = float(data.get("bankroll", starting))
    total_pnl = sum(float(t.get("pnl", 0)) for t in settled)
    unrealized = sum(float(t.get("unrealized_pnl", 0)) for t in open_trades_list)
    wins = sum(1 for t in settled if float(t.get("pnl", 0)) > 0)
    losses = sum(1 for t in settled if float(t.get("pnl", 0)) < 0)
    win_rate = wins / len(settled) if settled else 0.0
    avg_edge = (
        sum(float(t.get("edge_at_entry", 0)) for t in settled) / len(settled)
        if settled else 0.0
    )
    clv_sum = 0.0
    clv_n = 0
    for t in settled:
        entry = float(t.get("entry_price_cents", 0))
        close = float(t.get("closing_line_cents", entry))
        if entry > 0:
            clv_sum += (close - entry) / entry
            clv_n += 1
    clv = clv_sum / clv_n if clv_n else 0.0

    # Max drawdown from equity curve
    equity = starting
    peak = starting
    max_dd = 0.0
    for t in sorted(settled, key=lambda x: x.get("settled_at", "")):
        equity += float(t.get("pnl", 0))
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    roi = (bankroll - starting) / starting if starting > 0 else 0.0

    return {
        "starting_bankroll": starting,
        "bankroll": round(bankroll, 2),
        "total_pnl": round(total_pnl, 2),
        "roi": round(roi, 4),
        "win_rate": round(win_rate, 4),
        "wins": wins,
        "losses": losses,
        "num_trades": len(settled),
        "open_trades": len(open_trades_list),
        "unrealized_pnl": round(unrealized, 2),
        "equity": round(bankroll + unrealized, 2),
        "average_edge": round(avg_edge, 4),
        "closing_line_value": round(clv, 4),
        "max_drawdown": round(max_dd, 4),
        "recent_trades": sorted(trades, key=lambda x: x.get("opened_at", ""), reverse=True)[:20],
        "open_positions": sorted(open_trades_list, key=lambda x: x.get("opened_at", ""), reverse=True),
    }


def load_paper_trades() -> dict:
    return {**paper_stats(), "trades": _load().get("trades", [])}
