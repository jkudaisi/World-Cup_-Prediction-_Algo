# WC 2026 ML Prediction Engine

Machine-learning score predictions for all 72 World Cup 2026 group-stage matches, with live updates via API-Football and an optional **Kalshi auto-trading layer** (paper by default, live opt-in).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your keys. **Never commit `.env`** — it is listed in `.gitignore` (along with `*.pem` Kalshi private keys). Use `.env.example` as the template.

```
APIFOOTBALL_KEY=your_key_from_dashboard.api-football.com

# Optional — Kalshi trading (see below)
KALSHI_API_KEY_ID=...
KALSHI_PRIVATE_KEY_PATH=C:/path/to/kalshi-private-key.pem
```

Free API-Football tier: **7,500 requests/day** (Pro plan).

## Run

```bash
python server.py
```

Open **http://127.0.0.1:5000**

The background scheduler starts automatically when `APIFOOTBALL_KEY` is set. If Kalshi credentials are configured, it also refreshes trading opportunities and runs paper or live trading cycles on live match days.

---

## What’s in the dashboard

| Tab | Purpose |
|-----|---------|
| **Predictions** | Group-stage cards, model agreement, projected goals |
| **Today** | Live scores, per-match trades/signals, Kalshi-linked fixtures |
| **Trading** | Open positions, buy signals, mapped markets, account/risk summary |

The **Trading** tab shows:

- **Trading now** — bot-tracked open live/paper positions with unrealized P/L
- **Watching** — TRADE signals not yet entered
- **Recent live trades** — open + closed positions with settlement P/L
- Per-match Kalshi prices, edge, spread, liquidity, TRADE/SKIP reasons

---

## API routes

### Predictions & live data

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Dashboard HTML |
| `/api/predictions` | GET | Cached ML predictions JSON |
| `/api/run` | POST | Incremental retrain (skips if no new FT matches) |
| `/api/live` | GET | Predictions with live_trainer updates |
| `/api/today` | GET | Today's WC fixtures + live stats + trading overlay |
| `/api/scheduler` | GET | Scheduler budget and slot status |
| `/api/live-states` | GET | In-memory live match states |
| `/api/live-snapshots/<fixture_id>` | GET | Stored live snapshot history |
| `/api/status` | GET | Server, API budget, scheduler status |
| `/api/training-state` | GET | Incremental training state |
| `/api/train-incremental` | POST | Run incremental training manually |

### Kalshi market data

| Route | Method | Description |
|-------|--------|-------------|
| `/api/kalshi/markets` | GET | Open Kalshi markets |
| `/api/kalshi/orderbook/<ticker>` | GET | Orderbook + implied probability |
| `/api/kalshi/unmapped` | GET | Fixtures missing Kalshi tickers |
| `/api/kalshi/discover` | GET/POST | Auto-discover WC game/BTTS/total markets |

### Trading

| Route | Method | Description |
|-------|--------|-------------|
| `/api/trading/opportunities` | GET | All fixtures × markets with edge (cached) |
| `/api/trading/paper` | GET | Paper trading stats |
| `/api/trading/paper/run` | POST | Run paper trading scan |
| `/api/trading/live/positions` | GET | Open/closed live positions + marks |
| `/api/trading/live/run` | POST | Run live trading cycle |
| `/api/trading/live/enable` | POST | Enable live trading (runtime) |
| `/api/trading/live/disable` | POST | Disable live trading |
| `/api/trading/kill-switch` | POST | Emergency stop (`{"active": true}`) |
| `/api/trading/logs` | GET | Decisions, orders, results |
| `/api/trading/exposure` | GET | Risk dashboard (Kalshi balance when live) |
| `/api/trading/order` | POST | Place order (paper unless live enabled) |

---

## How live training works

1. **Morning init** — 1 API call fetches today's WC fixtures.
2. **Budget math** — calls split across matches in 103-minute windows; skipped during HT.
3. **Each slot** — `get_fixture_full` feeds `live_trainer.ingest_live_snapshot`.
4. **Lambda update** — live xG, possession, cards → adjusted Poisson lambdas in `predictions.json` / `live_predictions.json`.

## Incremental training

Models learn from completed World Cup matches without full retraining on every refresh.

| File | Role |
|------|------|
| `training_state.json` | Trained fixture IDs, last run |
| `data/world_cup_completed_matches.json` | Durable WC training rows |
| `data/base_training_cache.json` | Cached synthetic base rows |
| `models/` | Saved per-algorithm artifacts |

Refresh and `/api/predictions` **never** retrain — they read saved predictions.

---

## Kalshi trading layer

The trading system sits **on top of** the ML pipeline. Predictions and live updates always run; trading is optional.

| Mode | Description |
|------|-------------|
| **Predictions** | ML probabilities (always on) |
| **Paper trading** | Simulated fills, P/L — **default when `KALSHI_DRY_RUN=true`** |
| **Live trading** | Real Kalshi limit orders — **opt-in only** |

### Architecture (key modules)

| Module | Role |
|--------|------|
| `trading_service.py` | Scan fixtures, cache opportunities, orchestrate cycles |
| `kalshi_market_discovery.py` | Auto-match WC fixtures to Kalshi tickers |
| `kalshi_market_mapper.py` | Manual + discovered ticker mapping |
| `kalshi_client.py` | REST client — **Kalshi V2 event orders** |
| `market_pricing.py` | `orderbook_fp` parsing, executable prices |
| `edge_engine.py` | Model vs market edge, TRADE/SKIP |
| `entry_guards.py` | Block entries on decided outcomes / stale quotes |
| `position_outcomes.py` | Score-based settlement detection |
| `trade_executor.py` | Edge → risk → fresh price → limit order |
| `live_trader.py` | Live position ledger, hedge exits, settlement closes |
| `live_position_manager.py` | Auto enter/exit on live matches |
| `paper_position_manager.py` | Paper enter/exit + mark-to-market |
| `risk_manager.py` | Kelly sizing, exposure caps, daily loss |
| `kalshi_account.py` | Live bankroll from Kalshi balance |

### Auto market discovery

On startup / refresh, the bot discovers Kalshi WC series:

- `KXWCGAME-*` — match result (home / draw / away)
- `KXWCBTTS-*` — both teams to score
- `KXWCTOTAL-*` — over/under goal lines

Discovered mappings are cached in `data/kalshi_discovered_markets.json` and merged into `data/kalshi_market_mapping.json`. Manual overrides still work.

### Kalshi V2 orders

Orders use `POST /portfolio/events/orders` (V2). Legacy yes/no cent prices are mapped automatically:

- Buy YES → bid on YES leg
- Buy NO → ask on YES leg at `(1 − NO price)`

Cancel uses `DELETE /portfolio/events/orders/{order_id}`.

### Live auto-trading cycle (~every 20s on live days)

When `ENABLE_LIVE_TRADING=true`, `KALSHI_DRY_RUN=false`, and `AUTO_LIVE_TRADING=true`:

1. **Settle** positions when score already decides the market (BTTS both scored, total past line, etc.)
2. **Exit** stale positions when live model probability drops (hedge or settlement if market closed)
3. **Enter** ranked TRADE signals (highest edge first, up to `MAX_TRADES_PER_MATCH` per game)

Entry ranking picks the best **edge**, not the most extreme model probability.

### Entry guards (live)

The bot **will not enter** when:

- **Outcome already decided** from live score (e.g. 2–2 → no new Over 3.5 NO or BTTS NO)
- **Kalshi ≈ 99% YES but live model ≈ ≤50%** — stale/dead market quote
- **Spread ≥ 50¢** on BTTS / over markets during live play
- Normal rules: edge too small, illiquid, unmapped, kill switch, exposure caps

During live play, model probabilities come from **score-aware `goal_markets`**, not pre-match envelope.

### Position tracking & P/L

| File | Contents |
|------|----------|
| `data/live_positions.json` | Bot-tracked live positions (open + closed) |
| `data/paper_trades.json` | Paper trade ledger |
| `data/trading_decisions.json` | Every scan TRADE/SKIP/EXIT decision |
| `data/trade_orders.json` | Submitted order log |
| `data/trading_opportunities.json` | Cached scan payload for dashboard |

Open positions are marked with **fresh ticker bids** or **settlement P/L** when the score decides the bet. Closed positions record `exit_method`: `settlement`, `opposite_side_hedge`, etc.

Live sizing uses your **real Kalshi cash balance** (via API), not the `BANKROLL` fallback in `.env`.

### Safety defaults

- `KALSHI_DRY_RUN=true` — no real orders sent
- `ENABLE_LIVE_TRADING=false` — runtime gate for live orders
- Kill switch, max daily loss, per-match/total exposure caps
- Duplicate trade protection per ticker+side+fixture
- Stale/illiquid/low-confidence markets skipped

### Kalshi API setup

1. Create an API key at [kalshi.com/account/profile](https://kalshi.com/account/profile)
2. Save your RSA private key locally (keep out of git — `*.pem` is gitignored)
3. Configure `.env`:

```
KALSHI_API_KEY_ID=your_key_id
KALSHI_PRIVATE_KEY_PATH=C:/path/to/kalshi-private-key.pem
KALSHI_BASE_URL=https://api.elections.kalshi.com/trade-api/v2
KALSHI_DRY_RUN=true
ENABLE_LIVE_TRADING=false
AUTO_LIVE_TRADING=true
AUTO_PAPER_TRADING=true
```

Demo environment: `KALSHI_BASE_URL=https://demo-api.kalshi.co/trade-api/v2`

### Enable live trading (use with caution)

Live orders require **both**:

```
KALSHI_DRY_RUN=false
ENABLE_LIVE_TRADING=true
```

Or enable at runtime (still blocked if dry-run):

```bash
curl -X POST http://127.0.0.1:5000/api/trading/live/enable
```

Emergency stop:

```bash
curl -X POST http://127.0.0.1:5000/api/trading/kill-switch -H "Content-Type: application/json" -d "{\"active\": true}"
```

**Risk warning:** Prediction markets involve real financial risk. Start with paper trading, verify mappings on [Kalshi WC markets](https://kalshi.com), and use small stakes.

### Manual market mapping (optional)

Override or supplement discovery in `data/kalshi_market_mapping.json`:

```json
{
  "Portugal|DRC|2026-06-20": {
    "home_win": "KXWCGAME-26JUN20PORDRC-POR",
    "draw": "KXWCGAME-26JUN20PORDRC-TIE",
    "away_win": "KXWCGAME-26JUN20PORDRC-DRC",
    "over_2_5": "KXWCTOTAL-26JUN20PORDRC-3",
    "btts_yes": "KXWCBTTS-26JUN20PORDRC-BTTS"
  }
}
```

Keys also accept `mn:21` format. Team aliases (USA, DRC, South Korea, etc.) are handled automatically.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `APIFOOTBALL_KEY` | — | API-Football key |
| `KALSHI_API_KEY_ID` | — | Kalshi API key ID |
| `KALSHI_PRIVATE_KEY_PATH` | — | Path to RSA private key PEM |
| `KALSHI_BASE_URL` | production | Kalshi API base URL |
| `KALSHI_DRY_RUN` | `true` | Block real order submission |
| `ENABLE_LIVE_TRADING` | `false` | Allow live orders |
| `AUTO_LIVE_TRADING` | `true` | Auto cycle on live match days |
| `AUTO_PAPER_TRADING` | `true` | Auto paper scan when dry-run |
| `KILL_SWITCH` | `false` | Stop all trading |
| `MIN_EDGE_PREMATCH` | `0.08` | Min pre-match edge (8%) |
| `MIN_EDGE_LIVE` | `0.08–0.10` | Min live edge |
| `MIN_CONFIDENCE` | `0.55–0.60` | Min model confidence |
| `MAX_SPREAD_CENTS` | `6` | Max bid-ask spread |
| `MIN_LIQUIDITY_CONTRACTS` | `20` | Min book depth |
| `MAX_STAKE_PER_TRADE` | `2–5` | Max $ per trade (Kelly capped) |
| `MAX_TRADES_PER_MATCH` | `3` | Max open entries per game |
| `MAX_EXPOSURE_PER_MATCH` | `4` | Max $ at risk per match |
| `MAX_TOTAL_EXPOSURE` | `10` | Max $ at risk total |
| `MAX_DAILY_LOSS` | `5` | Daily loss halt |
| `BANKROLL` | `20` | Paper / display fallback; live uses Kalshi balance |
| `PAPER_*` | see `.env.example` | Aggressive paper-only overrides |

Copy from `.env.example` — do not commit your real `.env`.

### Edge & skip rules

- Pre-match / live minimum edge (configurable)
- Minimum confidence, max spread, min liquidity
- Entry guards for decided markets and stale Kalshi quotes
- Skip reasons logged to `data/trading_decisions.json`

### Paper trading

Paper mode runs when `KALSHI_DRY_RUN=true` (default):

```bash
curl -X POST http://127.0.0.1:5000/api/trading/paper/run
```

Results in `data/paper_trades.json`. Paper can use aggressive thresholds (`PAPER_AGGRESSIVE`, lower min edge) independent of live rules.

---

## Tests

```bash
# Fast unit tests (mocked — no API calls)
pytest tests/ -m "not integration"

# Live tests against API-Football (requires APIFOOTBALL_KEY in .env)
pytest tests/test_api_integration.py -m integration -v

# Everything
python test_all_apis.py -v
```

Coverage includes:

- **API-Football client** — fixtures, live stats, budget, errors
- **Flask routes** — predictions, live, today, scheduler, trading
- **Scheduler & live trainer** — polling, lambda updates
- **Kalshi trading** — V2 order mapping, auth, orderbook_fp parsing, edge engine, risk manager, entry guards, settlement, paper/live position managers, market discovery, goal markets, score matrix

Run trading tests only:

```bash
pytest tests/test_trading.py -v
```

---

## Project layout (trading-related)

```
├── server.py                 # Flask app + API routes
├── scheduler.py              # Live polling + trading cycle hook
├── trading_service.py        # Opportunity scan + cache
├── kalshi_client.py          # Kalshi REST + V2 orders
├── live_trader.py            # Live position ledger
├── live_position_manager.py  # Live enter / exit / settle
├── entry_guards.py           # Live entry safety checks
├── position_outcomes.py      # Score-based outcome detection
├── dashboard.js              # Predictions + Trading UI
├── data/
│   ├── kalshi_market_mapping.json
│   ├── kalshi_discovered_markets.json
│   ├── live_positions.json
│   ├── paper_trades.json
│   ├── trading_opportunities.json
│   └── trading_decisions.json
└── tests/
    ├── test_trading.py
    ├── test_kalshi_discovery.py
    └── test_kalshi_account.py
```

---

## Security notes

- **`.env` is gitignored** — contains API keys and paths to private keys
- **`*.pem` is gitignored** — never commit Kalshi RSA keys
- Commit **`.env.example`** only (placeholders, no secrets)
- Use `KALSHI_DRY_RUN=true` until you have verified mappings and paper results
