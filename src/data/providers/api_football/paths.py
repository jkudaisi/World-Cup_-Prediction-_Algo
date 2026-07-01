"""Provider paths and constants."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PROVIDER = "api_football"

RAW_ROOT = ROOT / "data" / "raw" / "providers" / PROVIDER
CACHE_ROOT = ROOT / "data" / "cache" / "providers" / PROVIDER
PROCESSED_ROOT = ROOT / "data" / "processed" / "providers" / PROVIDER
MANIFEST_ROOT = ROOT / "data" / "manifests" / "providers" / PROVIDER

CONFIG_BACKFILL = ROOT / "config" / "api_football_backfill.yaml"
CONFIG_TEAMS = ROOT / "config" / "world_cup_teams.yaml"
RESOLVED_TEAMS = MANIFEST_ROOT / "resolved_teams.json"

COMPLETED_STATUSES = frozenset({"FT", "AET", "PEN", "AWD", "WO"})
LIVE_STATUSES = frozenset({"1H", "HT", "2H", "ET", "P", "LIVE", "BT", "INT", "SUSP"})
SCHEDULED_STATUSES = frozenset({"NS", "TBD", "PST"})

ENDPOINT_FIXTURES = "/fixtures"
ENDPOINT_EVENTS = "/fixtures/events"
ENDPOINT_STATISTICS = "/fixtures/statistics"
ENDPOINT_PLAYERS = "/fixtures/players"
ENDPOINT_LINEUPS = "/fixtures/lineups"
ENDPOINT_INJURIES = "/injuries"
