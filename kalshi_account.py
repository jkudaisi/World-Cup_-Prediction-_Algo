"""Fetch live Kalshi account balance and position exposure."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from kalshi_auth import credentials_configured
from kalshi_client import KalshiClient, KalshiClientError
from trading_config import can_place_live_orders, get_config

log = logging.getLogger(__name__)

_CACHE: dict[str, Any] = {"data": None, "ts": 0.0}
DEFAULT_TTL_SECONDS = 45


def _parse_dollars(dollars_str: str | None, cents: int | None) -> float:
    if dollars_str is not None and str(dollars_str).strip():
        try:
            return float(dollars_str)
        except (TypeError, ValueError):
            pass
    if cents is not None:
        try:
            return float(cents) / 100.0
        except (TypeError, ValueError):
            pass
    return 0.0


def _position_exposure(positions_resp: dict) -> tuple[float, int]:
    total = 0.0
    markets = positions_resp.get("market_positions") or []
    for row in markets:
        try:
            total += float(row.get("market_exposure_dollars") or 0)
        except (TypeError, ValueError):
            continue
    return round(total, 4), len(markets)


def fetch_kalshi_account_summary(
    client: KalshiClient | None = None,
    *,
    force: bool = False,
    max_age_seconds: int = DEFAULT_TTL_SECONDS,
) -> dict[str, Any] | None:
    """
    Return Kalshi cash, open exposure, and account total (cash + in positions).

    Cached briefly to avoid hammering the portfolio API during scans.
    """
    if not credentials_configured():
        return None

    now = time.time()
    if (
        not force
        and _CACHE.get("data")
        and now - float(_CACHE.get("ts") or 0) < max_age_seconds
    ):
        return dict(_CACHE["data"])

    cli = client or KalshiClient()
    try:
        bal_resp = cli.get_balance()
        pos_resp = cli.get_positions()
    except KalshiClientError as exc:
        log.warning("Kalshi account fetch failed: %s", exc)
        if _CACHE.get("data"):
            stale = dict(_CACHE["data"])
            stale["stale"] = True
            return stale
        return None

    cash = _parse_dollars(bal_resp.get("balance_dollars"), bal_resp.get("balance"))
    in_positions, open_count = _position_exposure(pos_resp)
    account_total = round(cash + in_positions, 2)

    pv_cents = bal_resp.get("portfolio_value")
    portfolio_api = round(float(pv_cents) / 100.0, 2) if pv_cents is not None else None

    data = {
        "source": "kalshi",
        "balance": round(cash, 2),
        "available_cash": round(cash, 2),
        "in_positions": round(in_positions, 2),
        "account_total": account_total,
        "portfolio_value_api": portfolio_api,
        "open_position_count": open_count,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "updated_at_ts": now,
    }
    _CACHE["data"] = data
    _CACHE["ts"] = now
    return data


def should_use_kalshi_balance() -> bool:
    cfg = get_config()
    return bool(cfg.enable_live_trading and credentials_configured() and not cfg.dry_run)


def resolve_bankroll(
    client: KalshiClient | None = None,
    *,
    bankroll: float | None = None,
) -> float:
    """Bankroll for Kelly sizing: Kalshi account total when live, else config/paper."""
    if bankroll is not None:
        return float(bankroll)

    if should_use_kalshi_balance():
        summary = fetch_kalshi_account_summary(client)
        if summary and summary.get("account_total") is not None:
            return float(summary["account_total"])

    cfg = get_config()
    if cfg.auto_paper_trading or (cfg.dry_run and not cfg.enable_live_trading):
        from paper_trader import current_bankroll
        return current_bankroll()

    return float(cfg.bankroll)
