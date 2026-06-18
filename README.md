# WC 2026 ML Prediction Engine

Machine-learning score predictions for all 72 World Cup 2026 group-stage matches, with live updates via API-Football.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Add your API-Football key to `.env`:

```
APIFOOTBALL_KEY=your_key_from_dashboard.api-football.com
```

Free tier: **7,500 requests/day** (Pro plan).

## Run

```bash
python server.py
```

Open **http://127.0.0.1:5000**

The background scheduler starts automatically when `APIFOOTBALL_KEY` is set.

## API routes

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Dashboard HTML |
| `/api/predictions` | GET | Cached ML predictions JSON |
| `/api/run` | POST | Retrain all 7 models and save |
| `/api/live` | GET | Predictions with live_trainer updates |
| `/api/today` | GET | Today's WC fixtures + live stats |
| `/api/scheduler` | GET | Scheduler budget and slot status |
| `/api/live-states` | GET | In-memory live match states |
| `/api/status` | GET | Server, API budget, scheduler status |

## How live training works

1. **Morning init** — 1 API call fetches today's WC fixtures.
2. **Budget math** — 7,494 calls available (7,500 − 1 init − 5 reserve), split evenly across matches.
3. **Polling slots** — spaced across each match's 103-minute window; skipped during HT or outside kickoff windows.
4. **Each slot** — `get_fixture_full` (stats + events, 2 requests) feeds `live_trainer.ingest_live_snapshot`.
5. **Lambda update** — blends live xG proxy, possession, cards into adjusted Poisson lambdas written to `predictions.json`.

## Incremental training

Models learn from completed World Cup matches without retraining from scratch on every refresh.

- `training_state.json` — tracks trained fixture IDs and last run
- `data/world_cup_completed_matches.json` — durable WC match training rows
- `data/base_training_cache.json` — cached 5000 synthetic base rows (generated once)
- `models/` — saved model artifacts per algorithm

| Route | Method | Description |
|-------|--------|-------------|
| `/api/training-state` | GET | Current training state |
| `/api/train-incremental` | POST | Run incremental training manually |
| `/api/run` | POST | Incremental train (skips if no new FT matches) |

Refresh and `/api/predictions` **never** retrain — only read saved predictions.

## Tests

```bash
# Fast unit tests (mocked — no API calls)
pytest tests/ -m "not integration"

# Live tests against API-Football (requires APIFOOTBALL_KEY in .env)
pytest tests/test_api_integration.py -m integration -v

# Everything
python test_all_apis.py -v
```

Coverage:
- **API-Football client**: `get_today_fixtures`, `get_live_fixtures`, `get_fixture_stats`, `get_fixture_events`, `get_fixture_lineups`, `get_fixture_full`, budget counter, error handling
- **Flask routes**: `/`, `/api/predictions`, `/api/run`, `/api/live`, `/api/today`, `/api/scheduler`, `/api/live-states`, `/api/status`
- **Scheduler**: `build_day_schedule`, `should_call_now`, `get_today_view`, `get_scheduler_status`
- **Live trainer**: snapshot ingestion and lambda updates
