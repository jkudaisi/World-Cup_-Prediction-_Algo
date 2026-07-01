# API-Football Backfill

## Provider

This project uses **API-Football** from **API-Sports** (v3):

- Dashboard: https://dashboard.api-football.com
- Base URL: `https://v3.football.api-sports.io`

## Required environment variables

Set one of:

```bash
APIFOOTBALL_KEY=your_key
# or
API_FOOTBALL_KEY=your_key
# or
API_SPORTS_KEY=your_key
```

Never commit `.env` or API keys.

## Architecture

Provider-agnostic layout:

```text
data/raw/providers/api_football/     # raw data lake (inspectable per endpoint)
data/cache/providers/api_football/   # GET response cache with TTL
data/manifests/providers/api_football/  # run manifests + resume state
config/api_football_backfill.yaml    # backfill settings
config/world_cup_teams.yaml          # team list (hand-editable)
```

Resolved team IDs are written to:

```text
data/manifests/providers/api_football/resolved_teams.json
```

The YAML config is never overwritten automatically.

## Commands

### Coverage discovery (run first)

```bash
python scripts/api_football_discover_coverage.py --from-year 2000 --to-year 2026 --sample-fixtures 5
```

Dry run:

```bash
python scripts/api_football_discover_coverage.py --dry-run
```

### Full backfill

```bash
python scripts/api_football_full_backfill.py --from 2000-01-01 --to today --resume
```

Skip discovery if already run:

```bash
python scripts/api_football_full_backfill.py --skip-coverage-discovery
```

### Resume interrupted run

```bash
python scripts/api_football_resume_backfill.py
```

## Core rule

**Never discard a fixture because detail endpoints are missing.**

Every fixture is stored. Missing endpoints are logged in:

- `data/raw/providers/api_football/coverage/missing_endpoint_log.json`
- failed wrappers under each endpoint folder

## Raw wrapper format

Each saved response uses a standard wrapper with `provider`, `endpoint`, `params`, `success`, `error`, and `api_response`.

## Rate limits

The client sleeps between requests (default 0.25s), retries on 429/5xx, and caches responses to minimize duplicate calls.

## Feeds later pipeline

Part 2.1 stores raw data only. The feature store (Part 2+) reads from `data/raw/providers/api_football/` with explicit missing-data indicators.

## Safety

Reset/backfill scripts do **not** delete `.env`, credentials, or source code.
