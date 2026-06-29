"""Fair odds, expected value, Kelly, and trade recommendation metrics."""

from __future__ import annotations

import math
from typing import Any


def prob_to_fair_odds(p: float) -> float | None:
    """Decimal fair odds from probability."""
    if p is None or p <= 0:
        return None
    return round(1.0 / float(p), 4)


def prob_to_american_odds(p: float) -> int | None:
    if p is None or p <= 0 or p >= 1:
        return None
    if p >= 0.5:
        return int(round(-100 * p / (1 - p)))
    return int(round(100 * (1 - p) / p))


def compute_market_trading_metrics(
    *,
    fair_probability: float,
    market_probability: float | None = None,
    confidence: float = 0.5,
    min_edge: float = 0.08,
    min_confidence: float = 0.55,
    kelly_fraction_cap: float = 0.25,
    model_disagreement: float = 0.0,
) -> dict[str, Any]:
    """Trading analytics for a single market (does not place orders)."""
    fair_p = max(0.001, min(0.999, float(fair_probability)))
    fair_odds = prob_to_fair_odds(fair_p)
    out: dict[str, Any] = {
        "fair_probability": round(fair_p, 4),
        "fair_odds_decimal": fair_odds,
        "fair_odds_american": prob_to_american_odds(fair_p),
        "confidence": round(confidence, 3),
        "model_disagreement": round(model_disagreement, 3),
        "probability_uncertainty": round(max(0, 1 - confidence), 3),
    }

    if market_probability is None:
        out["recommendation"] = "NO_MARKET"
        out["reason"] = "No market price available"
        return out

    mkt_p = max(0.001, min(0.999, float(market_probability)))
    edge = round(fair_p - mkt_p, 4)
    ev_yes = round(fair_p * (1 - mkt_p) - (1 - fair_p) * mkt_p, 4)
    implied_odds = prob_to_fair_odds(mkt_p)

    # Kelly for YES: f* = (bp - q) / b where b = (1/mkt_p - 1)
    b = (1.0 / mkt_p) - 1.0
    kelly = (b * fair_p - (1 - fair_p)) / b if b > 0 else 0.0
    kelly = max(0.0, min(kelly_fraction_cap, kelly))

    should_trade = (
        edge >= min_edge
        and confidence >= min_confidence
        and model_disagreement < 0.35
    )

    out.update({
        "market_probability": round(mkt_p, 4),
        "implied_odds_decimal": implied_odds,
        "edge": edge,
        "edge_pct": round(edge * 100, 2),
        "expected_value_yes": ev_yes,
        "kelly_fraction": round(kelly, 4),
        "recommendation": "TRADE" if should_trade else "PASS",
        "reason": (
            f"Edge {edge:+.1%}, confidence {confidence:.0%}"
            if should_trade
            else _pass_reason(edge, confidence, min_edge, min_confidence, model_disagreement)
        ),
    })
    return out


def _pass_reason(
    edge: float,
    confidence: float,
    min_edge: float,
    min_confidence: float,
    disagreement: float,
) -> str:
    parts = []
    if edge < min_edge:
        parts.append(f"edge {edge:+.1%} < {min_edge:.0%}")
    if confidence < min_confidence:
        parts.append(f"confidence {confidence:.0%} < {min_confidence:.0%}")
    if disagreement >= 0.35:
        parts.append("high model disagreement")
    return "; ".join(parts) if parts else "thresholds not met"


def build_kalshi_market_metrics(
    markets: dict[str, float],
    kalshi_prices: dict[str, float | None] | None = None,
    *,
    confidence: float = 0.5,
    min_edge: float = 0.08,
    min_confidence: float = 0.55,
) -> dict[str, dict[str, Any]]:
    """Attach trading metrics for each named market."""
    prices = kalshi_prices or {}
    return {
        key: compute_market_trading_metrics(
            fair_probability=prob,
            market_probability=prices.get(key),
            confidence=confidence,
            min_edge=min_edge,
            min_confidence=min_confidence,
        )
        for key, prob in markets.items()
        if prob is not None
    }
