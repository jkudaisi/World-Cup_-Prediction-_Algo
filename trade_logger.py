"""Append-only trading decision and order logs."""

from __future__ import annotations

import json
import logging
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training_store import atomic_write_json

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
DECISIONS_PATH = ROOT / "data" / "trading_decisions.json"
ORDERS_PATH = ROOT / "data" / "trade_orders.json"
RESULTS_PATH = ROOT / "data" / "trade_results.json"

_lock = threading.Lock()


def _ensure_file(path: Path, default: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        atomic_write_json(path, default)


def _load_log_file(path: Path, key: str) -> dict[str, Any]:
    """Load a log JSON file; recover from truncated/corrupt writes."""
    default = {key: []}
    _ensure_file(path, default)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("log root must be object")
        data.setdefault(key, [])
        return data
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        backup = path.with_suffix(path.suffix + ".corrupt")
        try:
            shutil.copy2(path, backup)
        except OSError:
            pass
        log.warning("Reset corrupt log %s (%s); backup at %s", path.name, exc, backup.name)
        atomic_write_json(path, default)
        return dict(default)


def _append(path: Path, key: str, entry: dict) -> None:
    with _lock:
        data = _load_log_file(path, key)
        data.setdefault(key, []).append(entry)
        atomic_write_json(path, data)


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
    # Full scans emit thousands of SKIP rows — only persist actionable decisions.
    if extra and extra.get("scan") and decision != "TRADE":
        return entry
    try:
        _append(DECISIONS_PATH, "decisions", entry)
    except Exception as exc:
        log.warning("Could not append trading decision log: %s", exc)
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
    data = _load_log_file(DECISIONS_PATH, "decisions")
    decisions = data.get("decisions", [])
    return decisions[-limit:]


def load_orders(limit: int = 200) -> list[dict]:
    data = _load_log_file(ORDERS_PATH, "orders")
    return data.get("orders", [])[-limit:]


def load_results(limit: int = 200) -> list[dict]:
    data = _load_log_file(RESULTS_PATH, "results")
    return data.get("results", [])[-limit:]


def load_all_logs(limit: int = 200) -> dict[str, Any]:
    return {
        "decisions": load_decisions(limit),
        "orders": load_orders(limit),
        "results": load_results(limit),
    }
