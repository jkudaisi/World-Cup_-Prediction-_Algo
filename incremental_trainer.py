"""Incremental World Cup match training using API-Football completed fixtures."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from apifootball_client import (
    APIFootballError,
    WC_LEAGUE_ID,
    WC_SEASON,
    get_fixture_full,
    get_today_fixtures,
)
from model_store import load_artifacts, models_exist, save_artifacts
from team_names import resolve_team_name
from training_store import (
    append_wc_matches,
    atomic_write_json,
    dataset_checksum,
    load_base_cache,
    load_training_state,
    load_wc_matches,
    save_base_cache,
    save_training_state,
    utc_now_iso,
)
from wc2026_ml_pipeline import (
    FIXTURES,
    TEAM_STATS,
    generate_synthetic_dataset,
    get_feature_cols,
    predict_all_fixtures,
    train_models_from_frame,
    update_team_stats_from_match,
)
from feature_builder import build_features, calc_xg_proxy, sample_weight_for_row
from knockout_outcomes import is_knockout_fixture, parse_knockout_outcome

log = logging.getLogger(__name__)

COMPLETED_STATUSES = frozenset({"FT", "AET", "PEN"})
SKIP_REASON = "No new completed World Cup matches since last training run"

PREDICTIONS_PATH = Path(__file__).parent / "predictions.json"

_pending_training_ids: set[int] = set()


def _apply_team_stats_updates(rows: list[dict[str, Any]]) -> None:
    """Refresh TEAM_STATS form/xg/xga after each newly ingested WC match."""
    for row in rows:
        update_team_stats_from_match(row)


def _parse_possession(val: Any) -> float:
    if val is None:
        return 0.5
    if isinstance(val, str) and val.endswith("%"):
        try:
            return float(val.replace("%", "").strip()) / 100.0
        except ValueError:
            return 0.5
    try:
        v = float(val)
        return v / 100.0 if v > 1 else v
    except (TypeError, ValueError):
        return 0.5


def _int_or(val: Any, default: int = 0) -> int:
    try:
        return int(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def _float_or(val: Any, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def completed_match_to_training_row(
    fixture: dict,
    stats: dict | None = None,
    events: list | None = None,
) -> dict[str, Any] | None:
    """Convert API-Football fixture + stats into a training row."""
    home_raw = fixture.get("teams", {}).get("home", {}).get("name", "")
    away_raw = fixture.get("teams", {}).get("away", {}).get("name", "")
    home = resolve_team_name(home_raw)
    away = resolve_team_name(away_raw)

    if home not in TEAM_STATS or away not in TEAM_STATS:
        log.warning("Unknown teams for training row: %s vs %s", home_raw, away_raw)
        return None

    goals = fixture.get("goals") or {}
    ft = (fixture.get("score") or {}).get("fulltime") or {}
    gh = goals.get("home") if goals.get("home") is not None else ft.get("home")
    ga = goals.get("away") if goals.get("away") is not None else ft.get("away")
    if gh is None or ga is None:
        return None

    gh, ga = int(gh), int(ga)
    feats = build_features(home, away)
    hs = (stats or {}).get("home") or {}
    aws = (stats or {}).get("away") or {}

    home_sot = _int_or(hs.get("shots_on_goal"))
    away_sot = _int_or(aws.get("shots_on_goal"))
    home_corners = _int_or(hs.get("corner_kicks"))
    away_corners = _int_or(aws.get("corner_kicks"))
    home_id = fixture.get("teams", {}).get("home", {}).get("id")
    away_id = fixture.get("teams", {}).get("away", {}).get("id")

    row: dict[str, Any] = {
        **feats,
        "goals_h": gh,
        "goals_a": ga,
        "fixture_id": fixture["fixture"]["id"],
        "date": (fixture["fixture"].get("date") or "")[:10],
        "home_team": home,
        "away_team": away,
        "home_win": int(gh > ga),
        "draw": int(gh == ga),
        "away_win": int(gh < ga),
        "scoreline": f"{gh}-{ga}",
        "used_for_training": False,
        "source_timestamp": utc_now_iso(),
        "home_possession": _parse_possession(hs.get("ball_possession")),
        "away_possession": _parse_possession(aws.get("ball_possession")),
        "home_shots_total": _int_or(hs.get("shots_total")),
        "away_shots_total": _int_or(aws.get("shots_total")),
        "home_shots_on_target": home_sot,
        "away_shots_on_target": away_sot,
        "home_corners": home_corners,
        "away_corners": away_corners,
        "home_fouls": _int_or(hs.get("fouls")),
        "away_fouls": _int_or(aws.get("fouls")),
        "home_yellow_cards": _int_or(hs.get("yellow_cards")),
        "away_yellow_cards": _int_or(aws.get("yellow_cards")),
        "home_red_cards": _int_or(hs.get("red_cards")),
        "away_red_cards": _int_or(aws.get("red_cards")),
        "home_xg_proxy": round(calc_xg_proxy(hs, events, home_id), 3),
        "away_xg_proxy": round(calc_xg_proxy(aws, events, away_id), 3),
        "home_expected_goals": _float_or(hs.get("expected_goals"), calc_xg_proxy(hs, events, home_id)),
        "away_expected_goals": _float_or(aws.get("expected_goals"), calc_xg_proxy(aws, events, away_id)),
        "source": "world_cup",
    }
    if is_knockout_fixture(fixture):
        ko = parse_knockout_outcome(fixture)
        if ko:
            row.update({
                "knockout_round": ko.get("round", ""),
                "draw_at_90": ko.get("draw_at_90"),
                "went_to_et": ko.get("went_to_et"),
                "went_to_pens": ko.get("went_to_pens"),
                "home_won_et": ko.get("home_won_et"),
                "away_won_et": ko.get("away_won_et"),
                "home_won_pens": ko.get("home_won_pens"),
                "away_won_pens": ko.get("away_won_pens"),
                "home_qualifies": ko.get("home_qualifies"),
            })
    return row


def snapshot_to_training_row(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a final live polling snapshot into a WC training row."""
    home_raw = snapshot.get("home_name") or ""
    away_raw = snapshot.get("away_name") or ""
    home = resolve_team_name(home_raw)
    away = resolve_team_name(away_raw)

    if home not in TEAM_STATS or away not in TEAM_STATS:
        log.warning("Unknown teams in snapshot: %s vs %s", home_raw, away_raw)
        return None

    score = snapshot.get("score") or {}
    gh = score.get("home")
    ga = score.get("away")
    if gh is None or ga is None:
        return None

    gh, ga = int(gh), int(ga)
    fixture_id = snapshot.get("fixture_id")
    if fixture_id is None:
        return None

    stats = snapshot.get("stats") or {}
    events = snapshot.get("events") or []
    hs = stats.get("home") or {}
    aws = stats.get("away") or {}
    home_id = snapshot.get("home_team_id")
    away_id = snapshot.get("away_team_id")
    snap_time = snapshot.get("snapshot_time") or ""

    feats = build_features(home, away)
    row: dict[str, Any] = {
        **feats,
        "goals_h": gh,
        "goals_a": ga,
        "fixture_id": int(fixture_id),
        "date": snap_time[:10] if len(snap_time) >= 10 else "",
        "home_team": home,
        "away_team": away,
        "home_win": int(gh > ga),
        "draw": int(gh == ga),
        "away_win": int(gh < ga),
        "scoreline": f"{gh}-{ga}",
        "used_for_training": False,
        "source_timestamp": utc_now_iso(),
        "home_possession": _parse_possession(hs.get("ball_possession")),
        "away_possession": _parse_possession(aws.get("ball_possession")),
        "home_shots_total": _int_or(hs.get("shots_total")),
        "away_shots_total": _int_or(aws.get("shots_total")),
        "home_shots_on_target": _int_or(hs.get("shots_on_goal")),
        "away_shots_on_target": _int_or(aws.get("shots_on_goal")),
        "home_corners": _int_or(hs.get("corner_kicks")),
        "away_corners": _int_or(aws.get("corner_kicks")),
        "home_fouls": _int_or(hs.get("fouls")),
        "away_fouls": _int_or(aws.get("fouls")),
        "home_yellow_cards": _int_or(hs.get("yellow_cards")),
        "away_yellow_cards": _int_or(aws.get("yellow_cards")),
        "home_red_cards": _int_or(hs.get("red_cards")),
        "away_red_cards": _int_or(aws.get("red_cards")),
        "home_xg_proxy": round(calc_xg_proxy(hs, events, home_id), 3),
        "away_xg_proxy": round(calc_xg_proxy(aws, events, away_id), 3),
        "home_expected_goals": _float_or(hs.get("expected_goals"), calc_xg_proxy(hs, events, home_id)),
        "away_expected_goals": _float_or(aws.get("expected_goals"), calc_xg_proxy(aws, events, away_id)),
        "source": "live_snapshot",
    }
    return row


def scan_live_snapshots_for_completed_matches(
    snapshot_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Scan live snapshot files for newly completed matches and append them to
    world_cup_completed_matches.json (skipping fixture_ids already present).
    """
    from live_snapshot_store import (
        SNAPSHOT_DIR,
        get_final_completed_snapshot,
        load_snapshot_file,
    )

    directory = snapshot_dir or SNAPSHOT_DIR
    existing_ids = {
        int(m["fixture_id"])
        for m in load_wc_matches()
        if m.get("fixture_id") is not None
        and m.get("goals_h") is not None
        and m.get("goals_a") is not None
    }
    new_rows: list[dict[str, Any]] = []

    if not directory.exists():
        log.info("No snapshot directory at %s", directory)
        return new_rows

    for path in sorted(directory.glob("*.json")):
        snapshots = load_snapshot_file(path)
        final = get_final_completed_snapshot(snapshots)
        if final is None:
            continue

        try:
            fid = int(final.get("fixture_id") or path.stem)
        except (TypeError, ValueError):
            log.warning("Could not parse fixture id from %s", path.name)
            continue

        if fid in existing_ids:
            continue

        row = snapshot_to_training_row(final)
        if row is None:
            continue

        row["fixture_id"] = fid
        new_rows.append(row)
        existing_ids.add(fid)
        log.info(
            "Completed match from snapshot: %s vs %s (%s) fixture %s",
            row.get("home_team"),
            row.get("away_team"),
            row.get("scoreline"),
            fid,
        )

    if new_rows:
        append_wc_matches(new_rows)
        _apply_team_stats_updates(new_rows)
        log.info("Appended %s completed match(es) from live snapshots", len(new_rows))
    else:
        log.info("No new completed matches found in live snapshots")

    return new_rows


def fetch_completed_world_cup_fixtures() -> list[dict]:
    """Fetch all completed WC fixtures for the season (one API call)."""
    from apifootball_client import _get

    try:
        result = _get("/fixtures", {"league": WC_LEAGUE_ID, "season": WC_SEASON})
    except APIFootballError as exc:
        log.error("fetch_completed_world_cup_fixtures failed: %s", exc)
        return []
    if not isinstance(result, list):
        return []
    return [
        f for f in result
        if f.get("league", {}).get("id") == WC_LEAGUE_ID
        and f["fixture"]["status"]["short"] in COMPLETED_STATUSES
    ]


def fetch_new_completed_world_cup_matches(
    trained_fixture_ids: set[int] | list[int] | None = None,
    fetch_stats: bool = True,
) -> list[dict[str, Any]]:
    """Return training rows for completed WC matches not yet trained."""
    trained = set(trained_fixture_ids or [])
    fixtures = fetch_completed_world_cup_fixtures()
    new_fixtures = [f for f in fixtures if f["fixture"]["id"] not in trained]
    rows: list[dict[str, Any]] = []

    for fx in new_fixtures:
        fid = fx["fixture"]["id"]
        home_id = fx["teams"]["home"]["id"]
        away_id = fx["teams"]["away"]["id"]
        stats, events = {}, []
        if fetch_stats:
            try:
                full = get_fixture_full(fid, home_id, away_id)
                stats = full.get("stats") or {}
                events = full.get("events") or []
            except APIFootballError as exc:
                log.warning("Stats fetch failed for fixture %s: %s", fid, exc)
        row = completed_match_to_training_row(fx, stats, events)
        if row:
            rows.append(row)
    return rows


def ensure_base_cache(seed: int = 42, n_synthetic: int = 5000) -> pd.DataFrame:
    """Load or create cached synthetic base training data."""
    cached = load_base_cache()
    feature_cols = get_feature_cols()
    if cached and cached.get("feature_cols") == feature_cols and cached.get("rows"):
        return pd.DataFrame(cached["rows"])

    log.info("Generating base synthetic cache (%s rows)...", n_synthetic)
    df = generate_synthetic_dataset(n_synthetic=n_synthetic, seed=seed)
    save_base_cache({
        "seed": seed,
        "n_synthetic": n_synthetic,
        "feature_cols": feature_cols,
        "rows": df.to_dict(orient="records"),
        "created_at": utc_now_iso(),
    })
    return df


def wc_rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    from feature_builder import build_features, sanitize_training_frame

    feature_cols = get_feature_cols()
    records: list[dict[str, Any]] = []
    raw_weights: list[float] = []
    for r in rows:
        home = r.get("home_team")
        away = r.get("away_team")
        if home and away:
            try:
                feats = build_features(home, away)
            except KeyError:
                feats = {k: r[k] for k in feature_cols if k in r}
        else:
            feats = {k: r[k] for k in feature_cols if k in r}
        record = {k: feats.get(k, r.get(k)) for k in feature_cols}
        record["goals_h"] = r["goals_h"]
        record["goals_a"] = r["goals_a"]
        records.append(record)

        w = float(r.get("sample_weight", 1.0))
        knockout = bool(r.get("knockout_round")) or is_knockout_fixture(
            {"league": {"round": r.get("knockout_round", "")}},
        )
        if knockout:
            w *= 1.3
        elif int(r.get("league_id") or 0) == 1:
            w *= 1.1
        raw_weights.append(w)

    df = sanitize_training_frame(pd.DataFrame(records), feature_cols)
    if raw_weights:
        weights = np.array(raw_weights, dtype=float)
        mean_w = float(weights.mean()) if weights.mean() > 0 else 1.0
        df["sample_weight"] = weights / mean_w
    return df


def build_combined_training_frame(
    seed: int = 42,
    n_synthetic: int = 5000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from feature_builder import sanitize_training_frame

    feature_cols = get_feature_cols()
    base_df = sanitize_training_frame(ensure_base_cache(seed=seed, n_synthetic=n_synthetic), feature_cols)
    wc_rows = load_wc_matches()
    wc_df = wc_rows_to_frame(wc_rows)
    if wc_df.empty:
        return base_df, wc_df
    combined = sanitize_training_frame(pd.concat([base_df, wc_df], ignore_index=True), feature_cols)
    return combined, wc_df


def mark_fixture_for_training(fixture_id: int) -> None:
    _pending_training_ids.add(int(fixture_id))


def get_pending_training_ids() -> set[int]:
    return set(_pending_training_ids)


def clear_pending_training_ids(ids: set[int] | None = None) -> None:
    global _pending_training_ids
    if ids is None:
        _pending_training_ids = set()
    else:
        _pending_training_ids -= ids


def _build_predictions_payload(
    ml_data: list,
    training_meta: dict,
    seed: int = 42,
) -> dict[str, Any]:
    agree_count = sum(
        1 for m in ml_data
        if len({f"{p['gh']}-{p['ga']}" for p in m["models"].values()}) == 1
    )
    home_wins = sum(1 for m in ml_data if m["ens_h"] > m["ens_a"])
    draws = sum(1 for m in ml_data if m["ens_h"] == m["ens_a"])
    away_wins = sum(1 for m in ml_data if m["ens_h"] < m["ens_a"])
    total_goals = sum(m["ens_h"] + m["ens_a"] for m in ml_data)

    return {
        "ml_data": ml_data,
        "team_elo": {team: stats["elo"] for team, stats in TEAM_STATS.items()},
        "stats": {
            "total_goals": total_goals,
            "goals_per_match": round(total_goals / max(len(ml_data), 1), 2),
            "full_agree": agree_count,
            "home_wins": home_wins,
            "draws": draws,
            "away_wins": away_wins,
            "generated_at": utc_now_iso(),
        },
        "training": training_meta,
    }


def run_incremental_training(
    force: bool = False,
    fetch_from_api: bool = True,
    verbose: bool = False,
    seed: int = 42,
    n_synthetic: int = 5000,
    predictions_path: Path | None = None,
) -> dict[str, Any]:
    """
    Incrementally retrain models using cached base data + new WC completed matches.
    Skips if no new completed matches unless force=True (bootstrap when no models).
    """
    predictions_path = predictions_path or PREDICTIONS_PATH
    state = load_training_state()
    trained_ids = set(state.get("trained_fixture_ids", []))

    new_rows: list[dict[str, Any]] = []
    if fetch_from_api:
        new_rows = fetch_new_completed_world_cup_matches(trained_ids, fetch_stats=True)

    if not new_rows and not force and models_exist():
        state["last_incremental_run_status"] = "skipped"
        return {
            "status": "skipped",
            "reason": SKIP_REASON,
            "training_state": state,
        }

    bootstrap = not models_exist()
    if not new_rows and not bootstrap and not force:
        state["last_incremental_run_status"] = "skipped"
        return {
            "status": "skipped",
            "reason": SKIP_REASON,
            "training_state": state,
        }

    if new_rows:
        append_wc_matches(new_rows)
        _apply_team_stats_updates(new_rows)
        for row in new_rows:
            trained_ids.add(int(row["fixture_id"]))

    combined_df, wc_df = build_combined_training_frame(seed=seed, n_synthetic=n_synthetic)
    feature_cols = get_feature_cols()

    if verbose:
        log.info(
            "Training on %s rows (%s base + %s WC)",
            len(combined_df), len(combined_df) - len(wc_df), len(wc_df),
        )

    try:
        weights = np.ones(len(combined_df), dtype=float)
        if len(wc_df) > 0:
            wc_rows = load_wc_matches()
            base_len = len(combined_df) - len(wc_df)
            for i, row in enumerate(wc_rows[-len(wc_df):]):
                weights[base_len + i] = sample_weight_for_row({**row, "source": "world_cup"})
        trained, scaler = train_models_from_frame(
            combined_df, feature_cols, verbose=verbose, sample_weight=weights,
        )
        ml_data = predict_all_fixtures(trained, scaler, feature_cols, verbose=verbose)
        model_versions = {name: "1.0.0" for name in trained}
        save_artifacts(trained, scaler, feature_cols, model_versions)

        wc_all = load_wc_matches()
        new_count = len(new_rows)
        training_meta = {
            "mode": "incremental",
            "last_trained_at": utc_now_iso(),
            "new_matches_used": new_count,
            "total_world_cup_matches_used": len(wc_all),
            "trained_fixture_ids": sorted(trained_ids),
            "model_versions": model_versions,
            "training_rows_count": len(combined_df),
            "n_features": len(feature_cols),
            "seed": seed,
            "bootstrap": bootstrap,
        }

        payload = _build_predictions_payload(ml_data, training_meta, seed=seed)
        atomic_write_json(predictions_path, payload)

        last_date = max((m.get("date") for m in wc_all if m.get("date")), default=None)
        state.update({
            "last_trained_at": training_meta["last_trained_at"],
            "last_trained_fixture_date": last_date,
            "trained_fixture_ids": sorted(trained_ids),
            "model_versions": model_versions,
            "training_rows_count": len(combined_df),
            "new_matches_added": new_count,
            "last_incremental_run_status": "success",
            "errors": [],
            "dataset_checksum": dataset_checksum(wc_all),
            "total_world_cup_matches_used": len(wc_all),
        })
        save_training_state(state)

        wc_updated = load_wc_matches()
        for row in wc_updated:
            row["used_for_training"] = int(row.get("fixture_id", 0)) in trained_ids
        from training_store import save_wc_matches
        save_wc_matches(wc_updated)

        clear_pending_training_ids(set(r["fixture_id"] for r in new_rows))

        return {
            "status": "success",
            "new_matches_used": new_count,
            "total_world_cup_matches_used": len(wc_all),
            "last_trained_at": state["last_trained_at"],
            "training_state": state,
            **payload,
        }
    except Exception as exc:
        log.exception("Incremental training failed")
        state["last_incremental_run_status"] = "error"
        state.setdefault("errors", []).append({"at": utc_now_iso(), "message": str(exc)})
        save_training_state(state)
        raise


def process_pending_training(force_if_pending: bool = True) -> dict[str, Any] | None:
    """Run incremental training if scheduler queued final fixtures."""
    if not _pending_training_ids and not force_if_pending:
        return None
    pending = get_pending_training_ids()
    if not pending:
        return None

    state = load_training_state()
    trained = set(state.get("trained_fixture_ids", []))
    already_trained = pending & trained
    if already_trained:
        clear_pending_training_ids(already_trained)
        pending -= already_trained
    if not pending:
        return None

    log.info("Processing %s pending fixture(s) for incremental training", len(pending))
    result = run_incremental_training(force=False, fetch_from_api=True, verbose=False)

    if result and result.get("status") == "skipped":
        # Nothing new from API — drop queue entries already in training state
        clear_pending_training_ids(pending & trained)
        return result

    if result and result.get("status") == "success":
        updated_trained = set(result.get("training_state", {}).get("trained_fixture_ids", []))
        clear_pending_training_ids(pending & updated_trained)

    return result


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    added = scan_live_snapshots_for_completed_matches()
    print(f"Added {len(added)} match(es) from live snapshots")
    sys.exit(0)
