"""Shared mark-to-market and model-reversal exit logic for paper and live positions."""

from __future__ import annotations

from goal_markets import model_prob_for_market_type
from paper_trader import side_probability


SCAN_ENTRY_MARKETS = (
    "home_win",
    "away_win",
    "draw",
    "btts_yes",
    "over_2_5",
    "over_3_5",
    "home_advance",
    "away_advance",
)


def _edge_magnitude(opp: dict) -> float | None:
    if opp.get("edge") is not None:
        e = float(opp["edge"])
        return abs(e) if abs(e) <= 1.0 else abs(e) / 100.0
    if opp.get("edge_pct") is not None:
        return abs(float(opp["edge_pct"])) / 100.0
    return None


def ranked_opportunities(
    opportunities: list[dict],
    *,
    trade_only: bool = True,
) -> list[dict]:
    """Rank by edge (highest first). Prefer rows already marked TRADE from the scan."""
    ranked: list[tuple[float, dict]] = []
    for o in opportunities:
        if trade_only and o.get("recommendation") != "TRADE":
            continue
        edge = _edge_magnitude(o)
        if edge is None:
            continue
        ranked.append((edge, o))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [o for _, o in ranked]


def best_opportunity(opportunities: list[dict]) -> dict | None:
    ranked = ranked_opportunities(opportunities, trade_only=True)
    if ranked:
        return ranked[0]
    # Fallback: highest edge among all candidates (paper aggressive paths).
    fallback: list[tuple[float, dict]] = []
    for o in opportunities:
        edge = _edge_magnitude(o)
        if edge is not None:
            fallback.append((edge, o))
    if not fallback:
        return None
    fallback.sort(key=lambda x: x[0], reverse=True)
    return fallback[0][1]


def model_for_market(fixture: dict, market_type: str) -> float | None:
    gm = fixture.get("goal_markets") or {}
    p = model_prob_for_market_type(gm, market_type)
    if p is not None:
        return float(p)
    for o in fixture.get("opportunities") or []:
        if o.get("market_type") == market_type and o.get("model_probability") is not None:
            return float(o["model_probability"])
    return None


def live_scan_model_p(fixture: dict, market_type: str, scan_opp: dict) -> float:
    """Use the scan row's YES probability so entry matches the displayed edge."""
    if scan_opp.get("model_yes_probability") is not None:
        return float(scan_opp["model_yes_probability"])
    if scan_opp.get("model_probability") is not None:
        return float(scan_opp["model_probability"])
    live_p = model_for_market(fixture, market_type)
    if live_p is not None:
        return float(live_p)
    return 0.5


def market_cents_from_opp(opp: dict) -> float | None:
    if opp.get("kalshi_pct") is not None:
        return float(opp["kalshi_pct"])
    if opp.get("kalshi_probability") is not None:
        return float(opp["kalshi_probability"]) * 100.0
    return None


def build_marks_for_positions(
    fixtures: list[dict],
    positions: list[dict],
    *,
    id_key: str = "trade_id",
) -> dict[str, dict]:
    """Map position id -> {model_probability, market_cents} from latest scan."""
    by_fixture: dict[str, dict] = {}
    for fx in fixtures:
        fkey = fx.get("fixture_key")
        if not fkey:
            continue
        opps_by_type = {o.get("market_type"): o for o in (fx.get("opportunities") or [])}
        by_fixture[fkey] = {"opps": opps_by_type, "fixture": fx}

    marks: dict[str, dict] = {}
    for pos in positions:
        pid = pos.get(id_key)
        if not pid:
            continue
        fkey = pos.get("fixture_key")
        mt = pos.get("market_type")
        bundle = by_fixture.get(fkey)
        if not bundle:
            continue
        opp = bundle["opps"].get(mt, {})
        model_p = model_for_market(bundle["fixture"], mt)
        if model_p is None:
            model_p = opp.get("model_yes_probability") or opp.get("model_probability")
        market_cents = market_cents_from_opp(opp)
        if model_p is None and market_cents is None:
            continue
        marks[pid] = {
            "model_probability": model_p,
            "market_cents": market_cents,
        }
    return marks


def opposite_side_price_cents(
    side: str,
    market_cents: float | None,
    model_yes: float,
) -> float:
    """Price (cents) to buy the opposite side for a hedge exit."""
    side = (side or "yes").lower()
    if market_cents is not None:
        yes_c = float(market_cents)
        if side == "yes":
            return max(1.0, min(99.0, 100.0 - yes_c))
        return max(1.0, min(99.0, yes_c))
    opp_p = side_probability(model_yes, "no" if side == "yes" else "yes")
    return max(1.0, min(99.0, opp_p * 100.0))


def entry_side_probability(position: dict) -> float:
    if position.get("model_probability_at_entry") is not None:
        return float(position["model_probability_at_entry"])
    return float(position.get("model_probability") or 0.5)


def model_probability_drop(position: dict, model_yes: float) -> float:
    """How much the held-side model probability fell since entry."""
    side = (position.get("side") or "yes").lower()
    entry_side_p = entry_side_probability(position)
    current_side_p = side_probability(float(model_yes), side)
    return entry_side_p - current_side_p
