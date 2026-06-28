"""Local web server: serves the dashboard and runs the ML pipeline on demand."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

import config
import scheduler
from apifootball_client import DAILY_LIMIT, calls_remaining
from live_trainer import get_all_live_states
from live_updater import build_live_api_response, get_live_status, run_live_cycle
from live_snapshot_store import load_snapshots
from incremental_trainer import run_incremental_training
from training_store import load_training_state
from wc2026_ml_pipeline import save_predictions

try:
    from kalshi_auth import auth_status
    from kalshi_client import KalshiClient, KalshiClientError
    from market_pricing import parse_orderbook
    from risk_manager import risk_dashboard
    from trade_executor import execute_order
    from trade_logger import load_all_logs
    from trading_config import can_place_live_orders, config_summary, set_kill_switch, set_live_trading
    from trading_service import build_opportunities, get_opportunities, run_live_trading_scan, run_paper_trading_scan
    from kalshi_market_mapper import list_unmapped_fixtures
except ImportError:
    auth_status = None  # type: ignore
    list_unmapped_fixtures = None  # type: ignore

ROOT = Path(__file__).parent
PREDICTIONS_FILE = ROOT / "predictions.json"

app = Flask(__name__, static_folder=str(ROOT), static_url_path="")
_scheduler_booted = False


def _load_saved():
    if PREDICTIONS_FILE.exists():
        with open(PREDICTIONS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"ml_data": [], "team_elo": {}, "stats": None, "training": None}


def _ensure_scheduler() -> None:
    global _scheduler_booted
    if _scheduler_booted:
        return
    if config.APIFOOTBALL_KEY:
        if not scheduler.is_scheduler_running():
            scheduler.start_scheduler()
    _scheduler_booted = True


@app.before_request
def _boot_scheduler():
    _ensure_scheduler()


@app.route("/")
def index():
    return send_from_directory(ROOT, "wc2026_ml_predictions.html")


@app.route("/api/predictions")
def get_predictions():
    return jsonify(_load_saved())


@app.route("/api/run", methods=["POST"])
def run_models():
    try:
        data = save_predictions(PREDICTIONS_FILE, verbose=False, incremental=True)
        if data.get("status") == "skipped":
            saved = _load_saved()
            saved["incremental_result"] = data
            return jsonify(saved)
        return jsonify(data)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/training-state")
def api_training_state():
    state = load_training_state()
    state["live_update"] = get_live_status()
    return jsonify(state)


@app.route("/api/train-incremental", methods=["POST"])
def api_train_incremental():
    try:
        result = run_incremental_training(force=False, fetch_from_api=True, verbose=False,
                                          predictions_path=PREDICTIONS_FILE)
        if result.get("status") == "skipped":
            return jsonify(result)
        return jsonify({
            "status": "success",
            "new_matches_used": result.get("new_matches_used", 0),
            "total_world_cup_matches_used": result.get("total_world_cup_matches_used", 0),
            "last_trained_at": result.get("last_trained_at"),
        })
    except Exception as exc:
        logging.exception("train-incremental failed")
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/api/live")
def get_live():
    """Refresh live data if key configured, then return merged predictions."""
    try:
        if config.APIFOOTBALL_KEY:
            run_live_cycle()
        return jsonify(build_live_api_response())
    except Exception as exc:
        logging.exception("api/live failed")
        return jsonify({**_load_saved(), "error": str(exc)}), 500


@app.route("/api/live-snapshots/<int:fixture_id>")
def api_live_snapshots(fixture_id: int):
    return jsonify({
        "fixture_id": fixture_id,
        "snapshots": load_snapshots(fixture_id),
        "count": len(load_snapshots(fixture_id)),
    })


@app.route("/api/today")
def api_today():
    try:
        refresh = request.args.get("refresh") == "1"
        return jsonify(scheduler.get_today_view(refresh=refresh))
    except Exception as exc:
        logging.exception("api_today failed")
        return jsonify({"error": str(exc), "matches": [], "n_matches": 0}), 500


@app.route("/api/scheduler")
def api_scheduler():
    return jsonify(scheduler.get_scheduler_status())


@app.route("/api/live-states")
def api_live_states():
    return jsonify(get_all_live_states())


@app.route("/api/status")
def api_status():
    predictions_exist = PREDICTIONS_FILE.exists()
    predictions_age_seconds = None
    if predictions_exist:
        data = _load_saved()
        generated = (data.get("stats") or {}).get("generated_at")
        if generated:
            try:
                ts = datetime.fromisoformat(generated.replace("Z", "+00:00"))
                predictions_age_seconds = (datetime.now(timezone.utc) - ts).total_seconds()
            except ValueError:
                pass
    live_status = get_live_status()
    return jsonify({
        "server_version": "apifootball-v3-live",
        "predictions_exist": predictions_exist,
        "predictions_age_seconds": predictions_age_seconds,
        "apifootball_key_configured": bool(config.APIFOOTBALL_KEY),
        "apifootball_calls_remaining": calls_remaining(),
        "apifootball_daily_limit": DAILY_LIMIT,
        "scheduler_running": scheduler.is_scheduler_running(),
        "live_update": live_status,
        "server_time": datetime.now(timezone.utc).isoformat(),
        "trading": config_summary() if auth_status else {"available": False},
    })


# ── Kalshi & Trading API ──────────────────────────────────────────────────────

@app.route("/api/kalshi/markets")
def api_kalshi_markets():
    try:
        client = KalshiClient()
        data = client.get_markets(limit=200, status="open")
        return jsonify(data)
    except KalshiClientError as exc:
        return jsonify({"markets": [], "error": str(exc), "configured": auth_status() if auth_status else {}})
    except Exception as exc:
        return jsonify({"markets": [], "error": str(exc)}), 500


@app.route("/api/kalshi/orderbook/<ticker>")
def api_kalshi_orderbook(ticker: str):
    try:
        client = KalshiClient()
        raw = client.get_orderbook(ticker)
        from trading_config import get_config
        pricing = parse_orderbook(ticker, raw, stale_seconds=get_config().stale_price_seconds)
        return jsonify({"orderbook": raw, "pricing": pricing})
    except KalshiClientError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/kalshi/unmapped")
def api_kalshi_unmapped():
    if list_unmapped_fixtures is None:
        return jsonify({"unmapped": [], "count": 0, "error": "Trading module unavailable"}), 503
    data = _load_saved()
    unmapped = list_unmapped_fixtures(data.get("ml_data") or [])
    return jsonify({"unmapped": unmapped, "count": len(unmapped)})


@app.route("/api/kalshi/discover", methods=["GET", "POST"])
def api_kalshi_discover():
    """Discover FIFA World Cup tickers from Kalshi series (KXWCGAME, etc.)."""
    try:
        from kalshi_market_discovery import (
            KALSHI_WC_GAMES_URL,
            apply_discoveries_to_mapping,
            discover_wc_markets,
            load_discovery_cache,
            refresh_wc_discovery_if_stale,
        )
    except ImportError as exc:
        return jsonify({"error": str(exc), "status": "unavailable"}), 503

    force = request.method == "POST" or request.args.get("refresh") == "1"
    auto_apply = request.args.get("apply", "1") != "0"
    try:
        if force:
            client = KalshiClient()
            data = _load_saved()
            result = discover_wc_markets(client, ml_data=data.get("ml_data") or [])
            apply_result = apply_discoveries_to_mapping(result) if auto_apply else {}
            result["apply"] = apply_result
            result["status"] = "refreshed"
        else:
            result = refresh_wc_discovery_if_stale(auto_apply=auto_apply)
        result["kalshi_wc_url"] = KALSHI_WC_GAMES_URL
        return jsonify(result)
    except KalshiClientError as exc:
        return jsonify({"status": "error", "error": str(exc), "cache": load_discovery_cache()}), 400
    except Exception as exc:
        logging.exception("kalshi discover failed")
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/api/trading/opportunities")
def api_trading_opportunities():
    refresh = request.args.get("refresh") == "1"
    try:
        return jsonify(get_opportunities(refresh=refresh))
    except Exception as exc:
        logging.exception("trading opportunities failed")
        return jsonify({"error": str(exc), "opportunities": []}), 500


@app.route("/api/trading/paper")
def api_trading_paper():
    from paper_trader import load_paper_trades
    return jsonify(load_paper_trades())


@app.route("/api/trading/paper/run", methods=["POST"])
def api_trading_paper_run():
    try:
        return jsonify(run_paper_trading_scan())
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/api/trading/logs")
def api_trading_logs():
    limit = int(request.args.get("limit", 200))
    return jsonify(load_all_logs(limit=limit))


@app.route("/api/trading/exposure")
def api_trading_exposure():
    return jsonify(risk_dashboard())


@app.route("/api/trading/live/run", methods=["POST"])
def api_trading_live_run():
    try:
        return jsonify(run_live_trading_scan())
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/api/trading/live/positions")
def api_trading_live_positions():
    from live_trader import load_live_positions
    from trading_service import get_opportunities, _kalshi_client_if_configured

    data = get_opportunities(refresh=False)
    fixtures = data.get("fixtures") or []
    cli = _kalshi_client_if_configured()
    return jsonify(load_live_positions(fixtures=fixtures, client=cli))


@app.route("/api/trading/live/enable", methods=["POST"])
def api_trading_live_enable():
    set_live_trading(True)
    return jsonify({
        "status": "enabled",
        "warning": "Live trading enabled in runtime — real orders still blocked unless KALSHI_DRY_RUN=false",
        "config": config_summary(),
    })


@app.route("/api/trading/live/disable", methods=["POST"])
def api_trading_live_disable():
    set_live_trading(False)
    return jsonify({"status": "disabled", "config": config_summary()})


@app.route("/api/trading/kill-switch", methods=["POST"])
def api_trading_kill_switch():
    body = request.get_json(silent=True) or {}
    active = body.get("active", True)
    set_kill_switch(bool(active))
    return jsonify({"kill_switch": bool(active), "config": config_summary()})


@app.route("/api/trading/order", methods=["POST"])
def api_trading_order():
    body = request.get_json(silent=True) or {}
    required = ("ticker", "fixture_key", "home", "away", "market_type", "model_probability")
    missing = [k for k in required if k not in body]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    force_paper = bool(body.get("paper", True))
    if not force_paper and not can_place_live_orders():
        return jsonify({
            "error": "Live orders blocked. Set ENABLE_LIVE_TRADING=true and KALSHI_DRY_RUN=false",
            "config": config_summary(),
        }), 403

    try:
        result = execute_order(
            ticker=body["ticker"],
            side=body.get("side", "yes"),
            count=int(body.get("count", 1)),
            fixture_key=body["fixture_key"],
            home=body["home"],
            away=body["away"],
            market_type=body["market_type"],
            model_probability=float(body["model_probability"]),
            confidence=float(body.get("confidence", 0.5)),
            live=bool(body.get("live", False)),
            mapping_confidence=float(body.get("mapping_confidence", 1.0)),
            force_paper=force_paper,
        )
        return jsonify(result)
    except Exception as exc:
        logging.exception("trading order failed")
        return jsonify({"status": "error", "error": str(exc)}), 500


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if not PREDICTIONS_FILE.exists():
        print("No saved predictions found - running initial pipeline (this may take a few minutes)...")
        save_predictions(PREDICTIONS_FILE, verbose=False)
        print("Initial predictions saved.")
    if config.APIFOOTBALL_KEY:
        scheduler.start_scheduler()
    else:
        print("APIFOOTBALL_KEY not set — live scheduler disabled")
    print("\nOpen http://127.0.0.1:5000 in your browser")
    app.run(host="127.0.0.1", port=5000, debug=False)
