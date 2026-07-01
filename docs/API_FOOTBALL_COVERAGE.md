# API-Football Coverage

## Why coverage discovery exists

API-Football coverage varies by:

- league and season
- competition type (friendly vs World Cup vs qualifier)
- fixture age
- endpoint (events, lineups, player stats, injuries)
- API plan limits

The backfill system discovers what is available **before** assuming complete data.

## Discovery outputs

| File | Purpose |
|------|---------|
| `data/raw/providers/api_football/coverage/endpoint_coverage.json` | Per-team endpoint availability |
| `data/raw/providers/api_football/coverage/competition_coverage.json` | Per-league/season summary |
| `data/raw/providers/api_football/coverage/missing_endpoint_log.json` | Append-only missing endpoint log |
| `data/manifests/providers/api_football/coverage_manifest.json` | Discovery run summary |

## Coverage flags

```text
has_fixtures
has_events
has_team_statistics
has_player_statistics
has_lineups
has_injuries
has_standings
has_team_season_statistics
has_coach_data
has_squad_data
```

## Coverage score (foundation only)

`compute_endpoint_coverage_score()` in `src/data/providers/api_football/models.py`:

| Component | Weight |
|-----------|--------|
| Fixture core | 0.40 |
| Events | +0.10 |
| Team statistics | +0.20 |
| Player statistics | +0.15 |
| Lineups | +0.10 |
| Injuries | +0.05 |

This is **not** the final training sample weight — it is a reusable data-quality signal.

## Missing data does not invalidate a match

A fixture with only score data is kept with:

- `success: false` wrappers for missing endpoints
- explicit `missing_*` indicators
- lower coverage score

## Command

```bash
python scripts/api_football_discover_coverage.py --from-year 2000 --to-year 2026 --sample-fixtures 5
```

Use `--dry-run` to preview without writing files.

## Limitations

- National-team search can return ambiguous results; unresolved teams go to `unresolved_teams.json`.
- Very old fixtures may lack lineups/player stats even when events exist.
- Injury endpoint coverage is sparse for historical friendlies.
- `/predictions` and `/odds` are intentionally excluded from historical backfill (leakage / market comparison only).
