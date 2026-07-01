"""Config-driven weights and paths for the real-history pipeline."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Data layout (provider-agnostic)
DATA_RAW_PROVIDER = ROOT / "data" / "raw" / "providers" / "api_football"
DATA_CACHE_PROVIDER = ROOT / "data" / "cache" / "providers" / "api_football"
DATA_MANIFESTS_PROVIDER = ROOT / "data" / "manifests" / "providers" / "api_football"
DATA_PROCESSED_PROVIDER = ROOT / "data" / "processed" / "providers" / "api_football"

# Legacy flat paths (transitional)
DATA_RAW = DATA_RAW_PROVIDER
DATA_CACHE = DATA_CACHE_PROVIDER
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_MANIFESTS = ROOT / "data" / "manifests"
DATA_ARCHIVE = ROOT / "data" / "archive" / "pre_real_history_migration"
MODELS_REAL = ROOT / "models" / "real_history"

# Legacy paths (still used during migration)
LEGACY_WC_MATCHES = ROOT / "data" / "world_cup_completed_matches.json"
LEGACY_BASE_CACHE = ROOT / "data" / "base_training_cache.json"
LEGACY_MODELS = ROOT / "models"

# Recency weighting
RECENCY_WEIGHT_TABLE = [
    (0, 183, 1.50),       # 0–6 months
    (183, 365, 1.40),     # 6–12 months
    (365, 730, 1.25),     # 1–2 years
    (730, 1460, 1.00),    # 2–4 years
    (1460, 2190, 0.75),   # 4–6 years
    (2190, 3650, 0.45),   # 6–10 years
    (3650, 10_000_000, 0.20),  # 10+ years
]
RECENCY_DECAY_DAYS = 1460
RECENCY_METHOD = "table"  # "table" | "exponential"

# World Cup cycle weighting (reference cycle: 2026 WC)
WC_CYCLE_WEIGHTS = [
    (2023, 2026, 1.30),
    (2019, 2022, 0.85),
    (2015, 2018, 0.60),
    (2011, 2014, 0.35),
    (0, 2010, 0.20),
]

# Competition weights (league_id or competition_type key)
COMPETITION_WEIGHTS = {
    "world_cup_final": 2.00,
    "world_cup_knockout": 1.80,
    "world_cup_group": 1.60,
    "continental_knockout": 1.50,
    "continental_group": 1.30,
    "world_cup_qualifier": 1.25,
    "continental_qualifier": 1.15,
    "nations_league": 1.10,
    "friendly": 0.55,
    "default": 1.00,
}

# Continuity weights
COACH_SAME = 1.00
COACH_DIFFERENT = 0.75
COACH_UNKNOWN = 0.90
GK_SAME = 1.00
GK_DIFFERENT = 0.80
GK_UNKNOWN = 0.90

# Data quality tiers
DATA_QUALITY_FULL = 1.00
DATA_QUALITY_MISSING_PLAYER_STATS = 0.85
DATA_QUALITY_MISSING_LINEUPS = 0.90
DATA_QUALITY_MISSING_TEAM_STATS = 0.75
DATA_QUALITY_SCORE_ONLY = 0.55

# Synthetic / production flags
PRODUCTION_ALLOW_SYNTHETIC = False
SYNTHETIC_TEST_FIXTURE_PATH = ROOT / "tests" / "fixtures" / "synthetic_matches.json"
