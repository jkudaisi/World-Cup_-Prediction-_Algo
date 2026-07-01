#!/usr/bin/env python3
"""Generate config/world_cup_teams.yaml from TEAM_STATS (does not overwrite resolved_teams.json)."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from wc2026_ml_pipeline import TEAM_STATS

FIFA = {
    "Argentina": "ARG", "Brazil": "BRA", "France": "FRA", "Germany": "GER",
    "England": "ENG", "Spain": "ESP", "Portugal": "POR", "Netherlands": "NED",
    "Belgium": "BEL", "Croatia": "CRO", "USA": "USA", "Mexico": "MEX",
    "Japan": "JPN", "South Korea": "KOR", "Morocco": "MAR", "Senegal": "SEN",
    "Switzerland": "SUI", "Uruguay": "URU", "Colombia": "COL", "Ecuador": "ECU",
    "Australia": "AUS", "Canada": "CAN", "Qatar": "QAT", "Saudi Arabia": "KSA",
    "IR Iran": "IRN", "Tunisia": "TUN", "Egypt": "EGY", "Ghana": "GHA",
    "Algeria": "ALG", "Sweden": "SWE", "Scotland": "SCO", "Norway": "NOR",
    "Austria": "AUT", "Turkiye": "TUR", "Czechia": "CZE", "Paraguay": "PAR",
    "Panama": "PAN", "New Zealand": "NZL", "Ivory Coast": "CIV",
}

lines = ["teams:"]
for name in sorted(TEAM_STATS.keys()):
    code = FIFA.get(name)
    lines.append(f"  - name: {name}")
    lines.append("    aliases:")
    lines.append(f"      - {name}")
    lines.append(f"      - {name} National Team")
    lines.append("    api_football_team_id: null")
    lines.append(f"    fifa_code: {code if code else 'null'}")

out = ROOT / "config" / "world_cup_teams.yaml"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {len(TEAM_STATS)} teams to {out}")
