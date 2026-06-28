"""Strict bankroll and exposure risk controls."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from trading_config import get_config

ROOT = Path(__file__).parent
PAPER_TRADES_PATH = ROOT / "data" / "paper_trades.json"
ORDERS_PATH = ROOT / "data" / "trade_orders.json"

KELLY_BANKROLL_CAP = 0.15


def kelly_stake(
    bankroll: float,
    edge: float,
    model_p: float,
    confidence: float,
) -> float:
    """
    Fractional Kelly stake for a binary contract.

    Full Kelly fraction is scaled by confidence tier (0.50 / 0.35 / 0.25),
    then capped at 15% of bankroll.
    """
    bankroll = float(bankroll)
    edge = float(edge)
    model_p = float(model_p)
    confidence = float(confidence)

    if bankroll <= 0 or edge <= 0:
        return 0.0

    model_p = min(max(model_p, 0.01), 0.99)
    market_p = min(max(model_p - edge, 0.01), 0.99)
    full_kelly = edge / max(1.0 - market_p, 0.01)

    if confidence >= 0.80:
        fraction = 0.50
    elif confidence >= 0.70:
        fraction = 0.35
    else:
        fraction = 0.25

    stake = bankroll * full_kelly * fraction
    cap = bankroll * KELLY_BANKROLL_CAP
    return round(min(stake, cap), 2)


def stake_to_contracts(stake: float, entry_cents: float) -> int:
    """Convert dollar stake to whole contracts at the given entry price."""
    if stake <= 0 or entry_cents <= 0:
        return 0
    return int(stake * 100.0 / entry_cents)


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _today_str() -> str:
    return date.today().isoformat()


def _open_trades() -> list[dict]:
    paper = _load_json(PAPER_TRADES_PATH, {"trades": []})
    open_paper = [t for t in paper.get("trades", []) if t.get("status") == "open"]
    live_path = ROOT / "data" / "live_positions.json"
    live_doc = _load_json(live_path, {"positions": []})
    open_live = [p for p in live_doc.get("positions", []) if p.get("status") == "open"]
    orders = _load_json(ORDERS_PATH, {"orders": []})
    open_orders = [o for o in orders.get("orders", []) if o.get("status") in ("open", "pending", "resting")]
    return open_paper + open_live + open_orders


def _daily_pnl() -> float:
    paper = _load_json(PAPER_TRADES_PATH, {"trades": []})
    today = _today_str()
    pnl = 0.0
    for t in paper.get("trades", []):
        if t.get("settled_at", "")[:10] == today:
            pnl += float(t.get("pnl", 0))
    return round(pnl, 2)


def _consecutive_losses() -> int:
    paper = _load_json(PAPER_TRADES_PATH, {"trades": []})
    settled = [t for t in paper.get("trades", []) if t.get("status") == "settled"]
    settled.sort(key=lambda x: x.get("settled_at", ""), reverse=True)
    streak = 0
    for t in settled:
        if float(t.get("pnl", 0)) < 0:
            streak += 1
        else:
            break
    return streak


def exposure_by_match() -> dict[str, float]:
    exp: dict[str, float] = {}
    for t in _open_trades():
        key = t.get("fixture_key") or f"{t.get('home', '')}|{t.get('away', '')}"
        exp[key] = exp.get(key, 0) + float(t.get("stake", t.get("cost", 0)))
    return exp


def total_open_exposure() -> float:
    return round(sum(exposure_by_match().values()), 2)


def trades_count_by_match(fixture_key: str) -> int:
    count = 0
    for t in _open_trades():
        key = t.get("fixture_key") or f"{t.get('home', '')}|{t.get('away', '')}"
        if key == fixture_key:
            count += 1
    paper = _load_json(PAPER_TRADES_PATH, {"trades": []})
    today = _today_str()
    for t in paper.get("trades", []):
        if t.get("fixture_key") == fixture_key and t.get("opened_at", "")[:10] == today:
            count += 1
    return count


def evaluate_risk(
    *,
    stake: float,
    fixture_key: str,
    bankroll: float | None = None,
    edge: float | None = None,
    model_p: float | None = None,
    confidence: float | None = None,
    client: Any = None,
) -> dict[str, Any]:
    cfg = get_config()
    if bankroll is None:
        from kalshi_account import resolve_bankroll
        br = resolve_bankroll(client)
    else:
        br = float(bankroll)
    exp_match = exposure_by_match().get(fixture_key, 0.0)
    total_exp = total_open_exposure()
    daily = _daily_pnl()
    open_positions = len(_open_trades())
    streak = _consecutive_losses()
    match_trades = trades_count_by_match(fixture_key)

    kelly_edge = float(edge if edge is not None else 0.08)
    kelly_model_p = float(model_p if model_p is not None else 0.55)
    kelly_conf = float(confidence if confidence is not None else 0.60)
    max_allowed = kelly_stake(br, kelly_edge, kelly_model_p, kelly_conf)

    approved = True
    reasons: list[str] = []

    if cfg.kill_switch:
        approved = False
        reasons.append("Kill switch active")

    if stake > max_allowed:
        approved = False
        reasons.append(f"Stake ${stake:.2f} exceeds Kelly max ${max_allowed:.2f}")

    if exp_match + stake > cfg.max_exposure_per_match:
        approved = False
        reasons.append(f"Match exposure limit (${cfg.max_exposure_per_match:.2f})")

    if total_exp + stake > cfg.max_total_exposure:
        approved = False
        reasons.append(f"Total exposure limit (${cfg.max_total_exposure:.2f})")

    if daily <= -cfg.max_daily_loss:
        approved = False
        reasons.append(f"Daily loss limit reached (${cfg.max_daily_loss:.2f})")

    if match_trades >= cfg.max_trades_per_match:
        approved = False
        reasons.append(f"Max trades per match ({cfg.max_trades_per_match})")

    if streak >= cfg.max_consecutive_losses:
        approved = False
        reasons.append(f"{streak} consecutive losses — trading paused")

    if not approved:
        reason = "Risk rejected: " + reasons[0]
    else:
        reason = "Risk approved"

    return {
        "approved": approved,
        "max_allowed_stake": max_allowed,
        "kelly_stake": max_allowed,
        "reason": reason,
        "current_exposure": round(exp_match, 2),
        "total_exposure": total_exp,
        "daily_pnl": daily,
        "open_positions": open_positions,
        "consecutive_losses": streak,
        "fixture_key": fixture_key,
    }


def risk_dashboard(client: Any = None) -> dict[str, Any]:
    cfg = get_config()
    from kalshi_account import fetch_kalshi_account_summary, should_use_kalshi_balance

    bankroll = float(cfg.bankroll)
    available_cash = None
    in_positions = total_open_exposure()
    account_total = None
    bankroll_source = "config"
    kalshi_account = None

    if should_use_kalshi_balance():
        kalshi_account = fetch_kalshi_account_summary(client)
        if kalshi_account:
            bankroll_source = "kalshi"
            account_total = kalshi_account.get("account_total")
            available_cash = kalshi_account.get("available_cash")
            in_positions = kalshi_account.get("in_positions", in_positions)
            bankroll = float(account_total if account_total is not None else cfg.bankroll)

    return {
        "bankroll": round(bankroll, 2),
        "account_total": account_total,
        "available_cash": available_cash,
        "in_positions": round(float(in_positions), 2) if in_positions is not None else total_open_exposure(),
        "open_exposure": round(float(in_positions), 2) if should_use_kalshi_balance() and kalshi_account else total_open_exposure(),
        "exposure_by_match": exposure_by_match(),
        "daily_pnl": _daily_pnl(),
        "open_positions": len(_open_trades()),
        "kill_switch": cfg.kill_switch,
        "dry_run": cfg.dry_run,
        "live_trading_enabled": cfg.enable_live_trading,
        "can_place_live_orders": cfg.enable_live_trading and not cfg.dry_run and not cfg.kill_switch,
        "consecutive_losses": _consecutive_losses(),
        "bankroll_source": bankroll_source,
        "kalshi_account": kalshi_account,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
