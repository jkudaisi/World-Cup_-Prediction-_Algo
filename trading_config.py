"""Centralized Kalshi trading configuration (defaults: paper / dry-run only)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv(override=True)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


@dataclass(frozen=True)
class TradingConfig:
    dry_run: bool = True
    enable_live_trading: bool = False
    auto_paper_trading: bool = True
    auto_live_trading: bool = True
    require_manual_approval: bool = False

    min_edge_prematch: float = 0.08
    min_edge_live: float = 0.10
    min_confidence: float = 0.60
    min_confidence_live: float = 0.40
    max_spread_cents: float = 6.0
    min_liquidity_contracts: int = 20
    max_slippage_cents: float = 2.0
    stale_price_seconds: int = 120

    bankroll: float = 20.0
    max_stake_per_trade: float = 2.0
    max_exposure_per_match: float = 4.0
    max_total_exposure: float = 10.0
    max_daily_loss: float = 5.0
    max_trades_per_match: int = 3
    max_consecutive_losses: int = 3
    max_bankroll_pct_per_trade: float = 0.10

    # Paper trading (aggressive demo mode — live rules stay strict)
    paper_aggressive: bool = True
    paper_min_edge: float = 0.02
    paper_min_confidence: float = 0.45
    paper_use_model_pricing: bool = True
    paper_model_drop_exit: float = 0.05
    paper_exit_edge_reversal: float = 0.03
    paper_trades_per_match: int = 3
    paper_contracts: int = 1

    kill_switch: bool = False

    kalshi_api_key_id: str = ""
    kalshi_private_key_path: str = ""
    kalshi_base_url: str = "https://api.elections.kalshi.com/trade-api/v2"


def load_trading_config() -> TradingConfig:
    dry_run = _env_bool("KALSHI_DRY_RUN", True)
    enable_live = _env_bool("ENABLE_LIVE_TRADING", False)
    return TradingConfig(
        dry_run=dry_run,
        enable_live_trading=enable_live,
        auto_paper_trading=_env_bool("AUTO_PAPER_TRADING", True),
        auto_live_trading=_env_bool("AUTO_LIVE_TRADING", True),
        require_manual_approval=_env_bool("REQUIRE_MANUAL_APPROVAL", False),
        min_edge_prematch=_env_float("MIN_EDGE_PREMATCH", 0.08),
        min_edge_live=_env_float("MIN_EDGE_LIVE", 0.10),
        min_confidence=_env_float("MIN_CONFIDENCE", 0.60),
        min_confidence_live=_env_float("MIN_CONFIDENCE_LIVE", 0.40),
        max_spread_cents=_env_float("MAX_SPREAD_CENTS", 6.0),
        min_liquidity_contracts=_env_int("MIN_LIQUIDITY_CONTRACTS", 20),
        max_slippage_cents=_env_float("MAX_SLIPPAGE_CENTS", 2.0),
        stale_price_seconds=_env_int("STALE_PRICE_SECONDS", 120),
        bankroll=_env_float("BANKROLL", 20.0),
        max_stake_per_trade=_env_float("MAX_STAKE_PER_TRADE", 2.0),
        max_exposure_per_match=_env_float("MAX_EXPOSURE_PER_MATCH", 4.0),
        max_total_exposure=_env_float("MAX_TOTAL_EXPOSURE", 10.0),
        max_daily_loss=_env_float("MAX_DAILY_LOSS", 5.0),
        max_trades_per_match=_env_int("MAX_TRADES_PER_MATCH", 3),
        max_consecutive_losses=_env_int("MAX_CONSECUTIVE_LOSSES", 3),
        max_bankroll_pct_per_trade=_env_float("MAX_BANKROLL_PCT_PER_TRADE", 0.10),
        paper_aggressive=_env_bool("PAPER_AGGRESSIVE", True),
        paper_min_edge=_env_float("PAPER_MIN_EDGE", 0.02),
        paper_min_confidence=_env_float("PAPER_MIN_CONFIDENCE", 0.45),
        paper_use_model_pricing=_env_bool("PAPER_USE_MODEL_PRICING", True),
        paper_model_drop_exit=_env_float("PAPER_MODEL_DROP_EXIT", 0.05),
        paper_exit_edge_reversal=_env_float("PAPER_EXIT_EDGE_REVERSAL", 0.03),
        paper_trades_per_match=_env_int("PAPER_TRADES_PER_MATCH", 3),
        paper_contracts=_env_int("PAPER_CONTRACTS", 1),
        kill_switch=_env_bool("KILL_SWITCH", False),
        kalshi_api_key_id=os.getenv("KALSHI_API_KEY_ID", ""),
        kalshi_private_key_path=os.getenv("KALSHI_PRIVATE_KEY_PATH", ""),
        kalshi_base_url=os.getenv(
            "KALSHI_BASE_URL",
            "https://api.elections.kalshi.com/trade-api/v2",
        ),
    )


# Mutable runtime overrides (kill switch, live enable via API)
_runtime: dict[str, bool | None] = {
    "kill_switch": None,
    "enable_live_trading": None,
}


def get_config() -> TradingConfig:
    base = load_trading_config()
    runtime_kill = _runtime.get("kill_switch")
    runtime_live = _runtime.get("enable_live_trading")
    return TradingConfig(
        **{
            **base.__dict__,
            "kill_switch": base.kill_switch or bool(runtime_kill),
            "enable_live_trading": base.enable_live_trading if runtime_live is None else bool(runtime_live),
        }
    )


def set_kill_switch(active: bool) -> None:
    _runtime["kill_switch"] = active


def set_live_trading(enabled: bool) -> None:
    _runtime["enable_live_trading"] = enabled


def can_place_live_orders() -> bool:
    cfg = get_config()
    return (
        cfg.enable_live_trading
        and not cfg.dry_run
        and not cfg.kill_switch
    )


def config_summary() -> dict:
    cfg = get_config()
    return {
        "dry_run": cfg.dry_run,
        "enable_live_trading": cfg.enable_live_trading,
        "auto_paper_trading": cfg.auto_paper_trading,
        "auto_live_trading": cfg.auto_live_trading,
        "kill_switch": cfg.kill_switch,
        "require_manual_approval": cfg.require_manual_approval,
        "can_place_live_orders": can_place_live_orders(),
        "min_edge_prematch": cfg.min_edge_prematch,
        "min_edge_live": cfg.min_edge_live,
        "min_confidence": cfg.min_confidence,
        "min_confidence_live": cfg.min_confidence_live,
        "max_spread_cents": cfg.max_spread_cents,
        "min_liquidity_contracts": cfg.min_liquidity_contracts,
        "bankroll": cfg.bankroll,
        "max_stake_per_trade": cfg.max_stake_per_trade,
        "max_exposure_per_match": cfg.max_exposure_per_match,
        "max_total_exposure": cfg.max_total_exposure,
        "max_daily_loss": cfg.max_daily_loss,
        "kalshi_credentials_configured": bool(cfg.kalshi_api_key_id and cfg.kalshi_private_key_path),
        "paper_aggressive": cfg.paper_aggressive,
        "paper_min_edge": cfg.paper_min_edge,
        "paper_use_model_pricing": cfg.paper_use_model_pricing,
    }
