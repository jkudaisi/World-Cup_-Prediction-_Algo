"""Local web server: serves the dashboard and runs the ML pipeline on demand."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, send_from_directory

import config
import scheduler
from apifootball_client import DAILY_LIMIT, calls_remaining
from live_trainer import get_all_live_states
from live_updater import build_live_api_response, get_live_status, run_live_cycle
from live_snapshot_store import load_snapshots
from incremental_trainer import run_incremental_training
from training_store import load_training_state
from wc2026_ml_pipeline import save_predictions

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
        return jsonify(scheduler.get_today_view())
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
    })


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
