"""Aggregate realized P/L by calendar day from live and paper ledgers."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
LIVE_POSITIONS_PATH = ROOT / "data" / "live_positions.json"
PAPER_TRADES_PATH = ROOT / "data" / "paper_trades.json"


def _parse_closed_date(ts: str | None) -> date | None:
    if not ts:
        return None
    try:
        normalized = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.date()
    except (TypeError, ValueError):
        return None


def _load_live_closed() -> list[dict]:
    if not LIVE_POSITIONS_PATH.exists():
        return []
    with open(LIVE_POSITIONS_PATH, encoding="utf-8") as f:
        doc = json.load(f)
    return [
        p for p in doc.get("positions", [])
        if p.get("status") == "closed" and p.get("pnl") is not None
    ]


def _load_paper_closed() -> list[dict]:
    if not PAPER_TRADES_PATH.exists():
        return []
    with open(PAPER_TRADES_PATH, encoding="utf-8") as f:
        doc = json.load(f)
    return [
        t for t in doc.get("trades", [])
        if t.get("status") in ("closed", "settled") and t.get("pnl") is not None
    ]


def _closed_date_for_live(position: dict) -> date | None:
    return _parse_closed_date(position.get("closed_at")) or _parse_closed_date(position.get("opened_at"))


def _closed_date_for_paper(trade: dict) -> date | None:
    return (
        _parse_closed_date(trade.get("closed_at"))
        or _parse_closed_date(trade.get("settled_at"))
        or _parse_closed_date(trade.get("opened_at"))
    )


def _week_window(week_offset: int, *, today: date | None = None) -> tuple[date, date]:
    """Seven-day window ending `today - 7*week_offset` (offset 0 = includes today)."""
    today = today or date.today()
    offset = max(0, int(week_offset))
    end = today - timedelta(days=7 * offset)
    start = end - timedelta(days=6)
    return start, end


def weekly_pnl_history(*, week_offset: int = 0, today: date | None = None) -> dict[str, Any]:
    """
    Return seven days of realized P/L for live and paper trading.

    week_offset=0 → last 7 days including today.
    week_offset=1 → the prior 7-day block, etc.
    """
    start, end = _week_window(week_offset, today=today)
    day_keys: list[str] = []
    cursor = start
    while cursor <= end:
        day_keys.append(cursor.isoformat())
        cursor += timedelta(days=1)

    buckets: dict[str, dict[str, Any]] = {
        d: {
            "date": d,
            "live_pnl": 0.0,
            "paper_pnl": 0.0,
            "total_pnl": 0.0,
            "live_trades": 0,
            "paper_trades": 0,
            "trades": 0,
        }
        for d in day_keys
    }

    earliest: date | None = None

    for pos in _load_live_closed():
        closed_day = _closed_date_for_live(pos)
        if closed_day is None:
            continue
        earliest = closed_day if earliest is None else min(earliest, closed_day)
        if closed_day < start or closed_day > end:
            continue
        key = closed_day.isoformat()
        pnl = round(float(pos.get("pnl") or 0), 2)
        buckets[key]["live_pnl"] = round(buckets[key]["live_pnl"] + pnl, 2)
        buckets[key]["live_trades"] += 1
        buckets[key]["trades"] += 1
        buckets[key]["total_pnl"] = round(buckets[key]["live_pnl"] + buckets[key]["paper_pnl"], 2)

    for trade in _load_paper_closed():
        closed_day = _closed_date_for_paper(trade)
        if closed_day is None:
            continue
        earliest = closed_day if earliest is None else min(earliest, closed_day)
        if closed_day < start or closed_day > end:
            continue
        key = closed_day.isoformat()
        pnl = round(float(trade.get("pnl") or 0), 2)
        buckets[key]["paper_pnl"] = round(buckets[key]["paper_pnl"] + pnl, 2)
        buckets[key]["paper_trades"] += 1
        buckets[key]["trades"] += 1
        buckets[key]["total_pnl"] = round(buckets[key]["live_pnl"] + buckets[key]["paper_pnl"], 2)

    days = []
    for key in day_keys:
        row = buckets[key]
        d = date.fromisoformat(key)
        days.append({
            **row,
            "weekday": d.strftime("%a"),
            "label": d.strftime("%b %d"),
        })

    week_total = round(sum(d["total_pnl"] for d in days), 2)
    week_live = round(sum(d["live_pnl"] for d in days), 2)
    week_paper = round(sum(d["paper_pnl"] for d in days), 2)
    week_trades = sum(d["trades"] for d in days)

    has_older = earliest is not None and earliest < start
    has_newer = week_offset > 0

    return {
        "week_offset": max(0, int(week_offset)),
        "week_start": start.isoformat(),
        "week_end": end.isoformat(),
        "week_label": f"{start.strftime('%b %d')} – {end.strftime('%b %d, %Y')}",
        "days": days,
        "week_total": week_total,
        "week_live_total": week_live,
        "week_paper_total": week_paper,
        "week_trades": week_trades,
        "has_older_weeks": has_older,
        "has_newer_weeks": has_newer,
        "earliest_trade_date": earliest.isoformat() if earliest else None,
    }
