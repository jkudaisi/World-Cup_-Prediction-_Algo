"""Append-only trading decision and order logs."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
DECISIONS_PATH = ROOT / "data" / "trading_decisions.json"
ORDERS_PATH = ROOT / "data" / "trade_orders.json"
RESULTS_PATH = ROOT / "data" / "trade_results.json"

_lock = threading.Lock()


def _ensure_file(path: Path, default: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default, f, indent=2)


def _append(path: Path, key: str, entry: dict) -> None:
    _ensure_file(path, {key: []})
    with _lock:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault(key, []).append(entry)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def log_decision(
    *,
    fixture: str,
    market: str,
    ticker: str,
    model_probability: float | None,
    kalshi_probability: float | None,
    edge: float | None,
    confidence: float | None,
    spread: float | None,
    liquidity: int | float | None,
    decision: str,
    reason: str,
    risk_approval: bool | None = None,
    order_info: dict | None = None,
    extra: dict | None = None,
) -> dict:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fixture": fixture,
        "market": market,
        "ticker": ticker,
        "model_probability": model_probability,
        "kalshi_probability": kalshi_probability,
        "edge": edge,
        "confidence": confidence,
        "spread": spread,
        "liquidity": liquidity,
        "decision": decision,
        "reason": reason,
        "risk_approval": risk_approval,
        "order_info": order_info,
        **(extra or {}),
    }
    _append(DECISIONS_PATH, "decisions", entry)
    return entry


def log_order(order: dict) -> dict:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **order,
    }
    _append(ORDERS_PATH, "orders", entry)
    return entry


def log_result(result: dict) -> dict:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **result,
    }
    _append(RESULTS_PATH, "results", entry)
    return entry


def load_decisions(limit: int = 200) -> list[dict]:
    _ensure_file(DECISIONS_PATH, {"decisions": []})
    with open(DECISIONS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    decisions = data.get("decisions", [])
    return decisions[-limit:]


def load_orders(limit: int = 200) -> list[dict]:
    _ensure_file(ORDERS_PATH, {"orders": []})
    with open(ORDERS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("orders", [])[-limit:]


def load_results(limit: int = 200) -> list[dict]:
    _ensure_file(RESULTS_PATH, {"results": []})
    with open(RESULTS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("results", [])[-limit:]


def load_all_logs(limit: int = 200) -> dict[str, Any]:
    return {
        "decisions": load_decisions(limit),
        "orders": load_orders(limit),
        "results": load_results(limit),
    }
