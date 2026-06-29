"""
WC 2026 Full Machine Learning Pipeline
Models: Poisson Regression, Random Forest, Gradient Boosting (XGBoost),
        LightGBM, Ridge Regression, Neural Network (MLPRegressor), Ensemble
Each model predicts home_goals and away_goals independently → score.
Training data: synthetic historical features built from ELO, xG, form, H2H.
"""

import math, warnings, json
import logging
import numpy as np
import pandas as pd
from collections import defaultdict

warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)

# ── 1. TEAM FEATURE DATABASE ──────────────────────────────────────────────────
# Source: eloratings.net, FBref xG averages, Transfermarkt squad values,
#         FIFA rankings (June 2026), StatsBomb discipline averages
TEAM_STATS = {
    "Mexico":                 dict(elo=1795,rank=13,xg=1.52,xga=1.05,yc=1.8,rc=0.10,sq_val=520,wc_apps=17,titles=0,form=1.8,press=12.4,dribble=5.2,aerial=52),
    "South Africa":           dict(elo=1480,rank=61,xg=1.05,xga=1.42,yc=2.1,rc=0.12,sq_val=95, wc_apps=3, titles=0,form=1.1,press=9.1, dribble=3.8,aerial=48),
    "South Korea":            dict(elo=1685,rank=22,xg=1.32,xga=1.15,yc=1.9,rc=0.08,sq_val=390,wc_apps=11,titles=0,form=1.5,press=11.2,dribble=4.9,aerial=45),
    "Czechia":                dict(elo=1620,rank=44,xg=1.25,xga=1.20,yc=1.7,rc=0.06,sq_val=280,wc_apps=9, titles=0,form=1.4,press=10.5,dribble=4.4,aerial=50),
    "Canada":                 dict(elo=1650,rank=32,xg=1.38,xga=1.22,yc=1.6,rc=0.07,sq_val=310,wc_apps=2, titles=0,form=1.6,press=11.8,dribble=5.1,aerial=46),
    "Qatar":                  dict(elo=1420,rank=49,xg=0.88,xga=1.72,yc=2.0,rc=0.11,sq_val=60, wc_apps=2, titles=0,form=0.8,press=8.4, dribble=3.2,aerial=44),
    "Switzerland":            dict(elo=1730,rank=19,xg=1.48,xga=1.00,yc=1.5,rc=0.05,sq_val=430,wc_apps=12,titles=0,form=1.7,press=12.0,dribble=4.8,aerial=51),
    "Bosnia and Herzegovina": dict(elo=1590,rank=63,xg=1.30,xga=1.28,yc=1.9,rc=0.08,sq_val=190,wc_apps=1, titles=0,form=1.3,press=10.0,dribble=4.2,aerial=49),
    "Brazil":                 dict(elo=2020,rank=5, xg=2.05,xga=0.80,yc=1.6,rc=0.07,sq_val=1100,wc_apps=22,titles=5,form=2.1,press=14.2,dribble=7.2,aerial=54),
    "Morocco":                dict(elo=1770,rank=6, xg=1.42,xga=0.88,yc=1.8,rc=0.09,sq_val=460,wc_apps=6, titles=0,form=1.7,press=12.5,dribble=5.4,aerial=57),
    "Haiti":                  dict(elo=1340,rank=85,xg=0.80,xga=1.78,yc=2.2,rc=0.14,sq_val=35, wc_apps=1, titles=0,form=0.7,press=7.8, dribble=3.0,aerial=43),
    "Scotland":               dict(elo=1640,rank=38,xg=1.28,xga=1.15,yc=1.7,rc=0.06,sq_val=250,wc_apps=8, titles=0,form=1.4,press=11.0,dribble=4.3,aerial=53),
    "USA":                    dict(elo=1785,rank=15,xg=1.65,xga=1.10,yc=1.5,rc=0.06,sq_val=650,wc_apps=11,titles=0,form=1.8,press=12.8,dribble=5.5,aerial=50),
    "Paraguay":               dict(elo=1570,rank=42,xg=1.15,xga=1.25,yc=2.0,rc=0.10,sq_val=160,wc_apps=9, titles=0,form=1.2,press=9.5, dribble=4.0,aerial=52),
    "Australia":              dict(elo=1640,rank=23,xg=1.35,xga=1.20,yc=1.8,rc=0.08,sq_val=270,wc_apps=6, titles=0,form=1.5,press=11.1,dribble=4.6,aerial=48),
    "Turkiye":                dict(elo=1710,rank=26,xg=1.55,xga=1.15,yc=2.1,rc=0.10,sq_val=480,wc_apps=2, titles=0,form=1.7,press=12.2,dribble=5.3,aerial=55),
    "Germany":                dict(elo=1960,rank=9, xg=2.00,xga=0.90,yc=1.6,rc=0.05,sq_val=1050,wc_apps=20,titles=4,form=2.0,press=14.0,dribble=6.8,aerial=56),
    "Curacao":                dict(elo=1380,rank=83,xg=0.90,xga=1.65,yc=2.0,rc=0.11,sq_val=55, wc_apps=1, titles=0,form=0.9,press=8.0, dribble=3.4,aerial=42),
    "Ivory Coast":            dict(elo=1650,rank=30,xg=1.40,xga=1.25,yc=1.8,rc=0.09,sq_val=320,wc_apps=4, titles=0,form=1.5,press=11.5,dribble=5.0,aerial=51),
    "Ecuador":                dict(elo=1620,rank=28,xg=1.30,xga=1.20,yc=1.9,rc=0.09,sq_val=240,wc_apps=4, titles=0,form=1.4,press=10.8,dribble=4.5,aerial=48),
    "Netherlands":            dict(elo=1900,rank=8, xg=1.90,xga=0.90,yc=1.5,rc=0.05,sq_val=890,wc_apps=11,titles=0,form=1.9,press=13.5,dribble=6.2,aerial=52),
    "Japan":                  dict(elo=1760,rank=17,xg=1.58,xga=1.05,yc=1.4,rc=0.04,sq_val=440,wc_apps=8, titles=0,form=1.7,press=13.0,dribble=5.6,aerial=46),
    "Sweden":                 dict(elo=1720,rank=35,xg=1.50,xga=1.00,yc=1.6,rc=0.06,sq_val=380,wc_apps=12,titles=0,form=1.6,press=11.8,dribble=4.9,aerial=54),
    "Tunisia":                dict(elo=1540,rank=56,xg=1.15,xga=1.25,yc=1.9,rc=0.09,sq_val=130,wc_apps=6, titles=0,form=1.2,press=9.8, dribble=4.1,aerial=50),
    "Belgium":                dict(elo=1870,rank=10,xg=1.85,xga=0.95,yc=1.6,rc=0.06,sq_val=800,wc_apps=14,titles=0,form=1.9,press=13.2,dribble=6.0,aerial=53),
    "Egypt":                  dict(elo=1590,rank=29,xg=1.25,xga=1.15,yc=1.8,rc=0.08,sq_val=175,wc_apps=4, titles=0,form=1.3,press=10.2,dribble=4.3,aerial=50),
    "IR Iran":                dict(elo=1600,rank=24,xg=1.20,xga=1.15,yc=2.0,rc=0.10,sq_val=180,wc_apps=6, titles=0,form=1.3,press=9.9, dribble=4.1,aerial=51),
    "New Zealand":            dict(elo=1390,rank=82,xg=0.85,xga=1.60,yc=1.7,rc=0.07,sq_val=50, wc_apps=3, titles=0,form=0.8,press=8.2, dribble=3.1,aerial=44),
    "Spain":                  dict(elo=2050,rank=3, xg=2.15,xga=0.72,yc=1.4,rc=0.04,sq_val=1200,wc_apps=16,titles=1,form=2.2,press=15.0,dribble=7.5,aerial=51),
    "Cabo Verde":             dict(elo=1480,rank=64,xg=1.05,xga=1.35,yc=1.8,rc=0.09,sq_val=70, wc_apps=1, titles=0,form=1.0,press=8.8, dribble=3.6,aerial=46),
    "Saudi Arabia":           dict(elo=1560,rank=59,xg=1.10,xga=1.30,yc=1.9,rc=0.09,sq_val=155,wc_apps=7, titles=0,form=1.2,press=9.4, dribble=3.9,aerial=48),
    "Uruguay":                dict(elo=1850,rank=18,xg=1.75,xga=0.95,yc=1.8,rc=0.08,sq_val=680,wc_apps=14,titles=2,form=1.8,press=12.9,dribble=5.8,aerial=57),
    "France":                 dict(elo=2030,rank=2, xg=2.10,xga=0.78,yc=1.5,rc=0.05,sq_val=1250,wc_apps=16,titles=2,form=2.1,press=14.5,dribble=7.0,aerial=53),
    "Senegal":                dict(elo=1740,rank=16,xg=1.55,xga=1.00,yc=1.7,rc=0.08,sq_val=420,wc_apps=4, titles=0,form=1.6,press=12.1,dribble=5.5,aerial=55),
    "Iraq":                   dict(elo=1430,rank=60,xg=0.95,xga=1.50,yc=2.1,rc=0.12,sq_val=80, wc_apps=1, titles=0,form=0.9,press=8.5, dribble=3.5,aerial=47),
    "Norway":                 dict(elo=1750,rank=27,xg=1.60,xga=1.00,yc=1.5,rc=0.05,sq_val=510,wc_apps=4, titles=0,form=1.7,press=12.0,dribble=5.2,aerial=56),
    "Argentina":              dict(elo=2010,rank=1, xg=2.05,xga=0.78,yc=1.6,rc=0.06,sq_val=1100,wc_apps=18,titles=3,form=2.1,press=14.2,dribble=6.9,aerial=54),
    "Algeria":                dict(elo=1600,rank=31,xg=1.25,xga=1.15,yc=1.9,rc=0.09,sq_val=170,wc_apps=4, titles=0,form=1.3,press=10.1,dribble=4.2,aerial=49),
    "Austria":                dict(elo=1720,rank=21,xg=1.55,xga=1.10,yc=1.7,rc=0.07,sq_val=400,wc_apps=7, titles=0,form=1.6,press=12.0,dribble=5.0,aerial=52),
    "Jordan":                 dict(elo=1400,rank=67,xg=0.90,xga=1.55,yc=1.8,rc=0.09,sq_val=65, wc_apps=1, titles=0,form=0.9,press=8.3, dribble=3.3,aerial=46),
    "Portugal":               dict(elo=1980,rank=7, xg=2.00,xga=0.85,yc=1.5,rc=0.05,sq_val=1050,wc_apps=9, titles=0,form=2.0,press=13.8,dribble=6.5,aerial=52),
    "DRC":                    dict(elo=1490,rank=43,xg=1.05,xga=1.35,yc=2.0,rc=0.11,sq_val=90, wc_apps=1, titles=0,form=1.0,press=9.0, dribble=3.7,aerial=50),
    "Uzbekistan":             dict(elo=1480,rank=50,xg=1.00,xga=1.40,yc=1.9,rc=0.09,sq_val=75, wc_apps=1, titles=0,form=1.0,press=8.7, dribble=3.5,aerial=47),
    "Colombia":               dict(elo=1790,rank=14,xg=1.70,xga=1.00,yc=1.8,rc=0.08,sq_val=620,wc_apps=7, titles=0,form=1.7,press=12.5,dribble=5.7,aerial=50),
    "England":                dict(elo=1940,rank=4, xg=1.95,xga=0.85,yc=1.5,rc=0.05,sq_val=1100,wc_apps=17,titles=1,form=2.0,press=13.6,dribble=6.4,aerial=55),
    "Croatia":                dict(elo=1810,rank=11,xg=1.65,xga=0.90,yc=1.6,rc=0.06,sq_val=460,wc_apps=6, titles=0,form=1.7,press=12.4,dribble=5.4,aerial=53),
    "Ghana":                  dict(elo=1570,rank=73,xg=1.20,xga=1.25,yc=1.8,rc=0.09,sq_val=200,wc_apps=4, titles=0,form=1.3,press=10.2,dribble=4.4,aerial=50),
    "Panama":                 dict(elo=1540,rank=34,xg=1.05,xga=1.30,yc=1.9,rc=0.09,sq_val=110,wc_apps=2, titles=0,form=1.1,press=9.3, dribble=3.8,aerial=47),
}

# EMA weight for rolling form / xG updates after each completed WC match
_TEAM_STATS_EMA_ALPHA = 0.25


def _observed_match_xg(match_row: dict, side: str) -> float:
    """Attack xG for home/away from match row, falling back to goals scored."""
    if side == "home":
        for key in ("home_expected_goals", "home_xg_proxy"):
            val = match_row.get(key)
            if val is not None:
                return float(val)
        return float(match_row["goals_h"])
    for key in ("away_expected_goals", "away_xg_proxy"):
        val = match_row.get(key)
        if val is not None:
            return float(val)
    return float(match_row["goals_a"])


def _match_form_points(goals_for: int, goals_against: int) -> float:
    if goals_for > goals_against:
        return 3.0
    if goals_for == goals_against:
        return 1.0
    return 0.0


def _ema_update(current: float, observed: float, alpha: float = _TEAM_STATS_EMA_ALPHA) -> float:
    return round((1.0 - alpha) * current + alpha * observed, 3)


def update_team_stats_from_match(match_row: dict) -> None:
    """
    Mutate TEAM_STATS in place after a completed WC match.

    Updates form (W/D/L points), attacking xg, and defensive xga for both teams
    using an exponential moving average of the actual result.
    """
    home = match_row.get("home_team")
    away = match_row.get("away_team")
    if not home or not away:
        return
    if home not in TEAM_STATS or away not in TEAM_STATS:
        return

    gh = match_row.get("goals_h")
    ga = match_row.get("goals_a")
    if gh is None or ga is None:
        return

    gh, ga = int(gh), int(ga)
    home_xg = _observed_match_xg(match_row, "home")
    away_xg = _observed_match_xg(match_row, "away")

    home_stats = TEAM_STATS[home]
    home_stats["form"] = _ema_update(home_stats["form"], _match_form_points(gh, ga))
    home_stats["xg"] = _ema_update(home_stats["xg"], home_xg)
    home_stats["xga"] = _ema_update(home_stats["xga"], float(ga))

    away_stats = TEAM_STATS[away]
    away_stats["form"] = _ema_update(away_stats["form"], _match_form_points(ga, gh))
    away_stats["xg"] = _ema_update(away_stats["xg"], away_xg)
    away_stats["xga"] = _ema_update(away_stats["xga"], float(gh))


# ── 2. GROUP STAGE FIXTURES (from matches.csv) ────────────────────────────────
ID_MAP = {
    1:"Mexico",2:"South Africa",3:"South Korea",4:"Czechia",
    5:"Canada",6:"Bosnia and Herzegovina",7:"Qatar",8:"Switzerland",
    9:"Brazil",10:"Morocco",11:"Haiti",12:"Scotland",
    13:"USA",14:"Paraguay",15:"Australia",16:"Turkiye",
    17:"Germany",18:"Curacao",19:"Ivory Coast",20:"Ecuador",
    21:"Netherlands",22:"Japan",23:"Sweden",24:"Tunisia",
    25:"Belgium",26:"Egypt",27:"IR Iran",28:"New Zealand",
    29:"Spain",30:"Cabo Verde",31:"Saudi Arabia",32:"Uruguay",
    33:"France",34:"Senegal",35:"Iraq",36:"Norway",
    37:"Argentina",38:"Algeria",39:"Austria",40:"Jordan",
    41:"Portugal",42:"DRC",43:"Uzbekistan",44:"Colombia",
    45:"England",46:"Croatia",47:"Ghana",48:"Panama",
}

RAW = [(1,1,2,"A"),(2,3,4,"A"),(3,5,6,"B"),(4,13,14,"D"),(5,7,8,"B"),
       (6,9,10,"C"),(7,11,12,"C"),(8,15,16,"D"),(9,17,18,"E"),(10,21,22,"F"),
       (11,19,20,"E"),(12,23,24,"F"),(13,29,30,"H"),(14,25,26,"G"),(15,31,32,"H"),
       (16,27,28,"G"),(17,33,34,"I"),(18,35,36,"I"),(19,37,38,"J"),(20,39,40,"J"),
       (21,41,42,"K"),(22,45,46,"L"),(23,47,48,"L"),(24,43,44,"K"),
       (25,4,2,"A"),(26,8,6,"B"),(27,5,7,"B"),(28,1,3,"A"),
       (29,13,15,"D"),(30,12,10,"C"),(31,9,11,"C"),(32,16,14,"D"),
       (33,21,23,"F"),(34,17,19,"E"),(35,20,18,"E"),(36,24,22,"F"),
       (37,29,31,"H"),(38,25,27,"G"),(39,32,30,"H"),(40,28,26,"G"),
       (41,37,39,"J"),(42,33,35,"I"),(43,36,34,"I"),(44,40,38,"J"),
       (45,41,43,"K"),(46,45,47,"L"),(47,48,46,"L"),(48,44,42,"K"),
       (49,8,5,"B"),(50,6,7,"B"),(51,12,9,"C"),(52,10,11,"C"),
       (53,4,1,"A"),(54,2,3,"A"),(55,18,19,"E"),(56,20,17,"E"),
       (57,22,23,"F"),(58,24,21,"F"),(59,16,13,"D"),(60,14,15,"D"),
       (61,36,33,"I"),(62,34,35,"I"),(63,30,31,"H"),(64,32,29,"H"),
       (65,26,27,"G"),(66,28,25,"G"),(67,48,45,"L"),(68,46,47,"L"),
       (69,44,41,"K"),(70,42,43,"K"),(71,38,39,"J"),(72,40,37,"J")]

FIXTURES = [(mn, ID_MAP[h], ID_MAP[a], g) for mn,h,a,g in RAW]

# ── 3. FEATURE ENGINEERING ────────────────────────────────────────────────────
from feature_builder import build_features, elo_prob, get_feature_cols as _fb_feature_cols

# ── 4. FLAGS & MODEL NAME MAP (for HTML export) ───────────────────────────────
FLAGS = {
    "Mexico": "🇲🇽", "South Africa": "🇿🇦", "South Korea": "🇰🇷", "Czechia": "🇨🇿",
    "Canada": "🇨🇦", "Qatar": "🇶🇦", "Switzerland": "🇨🇭", "Bosnia and Herzegovina": "🇧🇦",
    "Brazil": "🇧🇷", "Morocco": "🇲🇦", "Haiti": "🇭🇹", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "USA": "🇺🇸", "Paraguay": "🇵🇾", "Australia": "🇦🇺", "Turkiye": "🇹🇷",
    "Germany": "🇩🇪", "Curacao": "🇨🇼", "Ivory Coast": "🇨🇮", "Ecuador": "🇪🇨",
    "Netherlands": "🇳🇱", "Japan": "🇯🇵", "Sweden": "🇸🇪", "Tunisia": "🇹🇳",
    "Belgium": "🇧🇪", "Egypt": "🇪🇬", "IR Iran": "🇮🇷", "New Zealand": "🇳🇿",
    "Spain": "🇪🇸", "Cabo Verde": "🇨🇻", "Saudi Arabia": "🇸🇦", "Uruguay": "🇺🇾",
    "France": "🇫🇷", "Senegal": "🇸🇳", "Iraq": "🇮🇶", "Norway": "🇳🇴",
    "Argentina": "🇦🇷", "Algeria": "🇩🇿", "Austria": "🇦🇹", "Jordan": "🇯🇴",
    "Portugal": "🇵🇹", "DRC": "🇨🇩", "Uzbekistan": "🇺🇿", "Colombia": "🇨🇴",
    "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Croatia": "🇭🇷", "Ghana": "🇬🇭", "Panama": "🇵🇦",
}

MODEL_HTML_NAMES = {
    "Poisson Regression": "Poisson",
    "Ridge Regression": "Ridge",
    "Random Forest": "Random Forest",
    "Gradient Boosting": "Gradient Boost",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
    "Neural Network": "Neural Net",
}

SCALED_MODELS = {"Poisson Regression", "Ridge Regression", "Neural Network"}


def get_feature_cols() -> list[str]:
    return _fb_feature_cols()


def generate_synthetic_dataset(n_synthetic: int = 5000, seed: int = 42) -> "pd.DataFrame":
    np.random.seed(seed)
    team_names = list(TEAM_STATS.keys())
    rows = []
    for _ in range(n_synthetic):
        h, a = np.random.choice(team_names, size=2, replace=False)
        feats = build_features(h, a)
        lh_noise = feats["lambda_h"] * np.random.uniform(0.75, 1.35)
        la_noise = feats["lambda_a"] * np.random.uniform(0.75, 1.35)
        row = feats.copy()
        row["goals_h"] = np.random.poisson(lh_noise)
        row["goals_a"] = np.random.poisson(la_noise)
        rows.append(row)
    return pd.DataFrame(rows)


def _build_model_defs(X: np.ndarray, X_sc: np.ndarray):
    from sklearn.linear_model import PoissonRegressor, Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor

    return {
        "Poisson Regression": (
            PoissonRegressor(alpha=1.0, max_iter=300),
            PoissonRegressor(alpha=1.0, max_iter=300),
            X_sc,
        ),
        "Ridge Regression": (Ridge(alpha=1.0), Ridge(alpha=1.0), X_sc),
        "Random Forest": (
            RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
            RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
            X,
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.08, random_state=42),
            GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.08, random_state=42),
            X,
        ),
        "XGBoost": (
            XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.07, subsample=0.8,
                         colsample_bytree=0.8, random_state=42, verbosity=0),
            XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.07, subsample=0.8,
                         colsample_bytree=0.8, random_state=42, verbosity=0),
            X,
        ),
        "LightGBM": (
            LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.07, subsample=0.8,
                          colsample_bytree=0.8, random_state=42, verbose=-1),
            LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.07, subsample=0.8,
                          colsample_bytree=0.8, random_state=42, verbose=-1),
            X,
        ),
        "Neural Network": (
            MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation="relu", max_iter=500,
                         random_state=42, learning_rate_init=0.001),
            MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation="relu", max_iter=500,
                         random_state=42, learning_rate_init=0.001),
            X_sc,
        ),
    }


def train_models_from_frame(df: "pd.DataFrame", feature_cols: list[str] | None = None, verbose: bool = False,
                            sample_weight: np.ndarray | None = None):
    from sklearn.preprocessing import StandardScaler
    from feature_builder import sanitize_training_frame

    if feature_cols is None:
        feature_cols = get_feature_cols()
    sw = None
    if "sample_weight" in df.columns:
        sw = df["sample_weight"].to_numpy().astype(float)
        if sample_weight is not None and len(sample_weight) == len(sw):
            missing = np.isnan(sw)
            sw = sw.copy()
            sw[missing] = sample_weight[missing]
        else:
            sw = np.nan_to_num(sw, nan=1.0)
    elif sample_weight is not None:
        sw = sample_weight
    df = sanitize_training_frame(df, feature_cols)
    X = df[feature_cols].values.astype(float)
    y_h = df["goals_h"].values.astype(float)
    y_a = df["goals_a"].values.astype(float)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    model_defs = _build_model_defs(X, X_sc)

    if verbose:
        print(f"Training data: {len(df)} rows | {len(feature_cols)} features")

    trained = {}
    for name, (mh, ma, Xtr) in model_defs.items():
        if sw is not None and name == "Neural Network":
            log.debug("MLP does not support sample_weight, training with uniform weights")
            mh.fit(Xtr, y_h)
            ma.fit(Xtr, y_a)
        elif sw is not None:
            mh.fit(Xtr, y_h, sample_weight=sw)
            ma.fit(Xtr, y_a, sample_weight=sw)
        else:
            mh.fit(Xtr, y_h)
            ma.fit(Xtr, y_a)
        trained[name] = (mh, ma)
        if verbose:
            print(f"  OK {name}")
    return trained, scaler


def predict_single_match(
    home: str,
    away: str,
    *,
    group: str = "KO",
    match_number: int | None = None,
    trained=None,
    scaler=None,
    feature_cols: list[str] | None = None,
    knockout: bool = True,
) -> dict:
    """Predict one fixture using saved or provided model artifacts."""
    from ensemble import build_prediction_envelope, weighted_ensemble_goals
    from model_store import load_artifacts

    if home not in TEAM_STATS or away not in TEAM_STATS:
        raise ValueError(f"Unknown teams for prediction: {home} vs {away}")

    if trained is None or scaler is None or feature_cols is None:
        artifacts = load_artifacts()
        if not artifacts:
            raise RuntimeError("No trained model artifacts found — run training first")
        trained = artifacts["trained"]
        scaler = artifacts["scaler"]
        feature_cols = artifacts["feature_cols"]

    ctx = {"knockout_stage": 1.0 if knockout else 0.0}
    feats = build_features(home, away, context=ctx)
    feat_vec = np.array([feats[c] for c in feature_cols]).reshape(1, -1)
    feat_vec_sc = scaler.transform(feat_vec)

    model_preds = {}
    for name, (mh, ma) in trained.items():
        Xin = feat_vec_sc if name in SCALED_MODELS else feat_vec
        raw_h = float(mh.predict(Xin)[0])
        raw_a = float(ma.predict(Xin)[0])
        gh = max(0, round(raw_h))
        ga = max(0, round(raw_a))
        model_preds[name] = (gh, ga, raw_h, raw_a)

    rh, ra, model_agreement = weighted_ensemble_goals(model_preds)
    ens_h = max(0, round(rh))
    ens_a = max(0, round(ra))

    html_models = {
        MODEL_HTML_NAMES[name]: {"gh": gh, "ga": ga, "rh": rh, "ra": ra}
        for name, (gh, ga, rh, ra) in model_preds.items()
    }

    envelope = build_prediction_envelope(
        home, away, model_preds,
        data_quality=0.65,
        lineup_completeness=0.5,
    )
    envelope["ensemble"]["model_agreement"] = round(model_agreement, 3)

    entry = {
        "mn": match_number,
        "group": group,
        "home": home,
        "away": away,
        "home_flag": FLAGS.get(home, "🏳️"),
        "away_flag": FLAGS.get(away, "🏳️"),
        "models": html_models,
        "ens_h": ens_h,
        "ens_a": ens_a,
        "ens": f"{ens_h}-{ens_a}",
        "prediction": envelope["prediction"],
        "confidence": envelope["confidence"],
        "explanation": envelope["explanation"],
        "ensemble": envelope["ensemble"],
    }
    return entry


def predict_all_fixtures(trained, scaler, feature_cols: list[str], verbose: bool = False) -> list[dict]:
    from ensemble import build_prediction_envelope, weighted_ensemble_goals

    ml_data = []
    current_group = None

    for mn, home, away, grp in FIXTURES:
        feats = build_features(home, away)
        feat_vec = np.array([feats[c] for c in feature_cols]).reshape(1, -1)
        feat_vec_sc = scaler.transform(feat_vec)

        model_preds = {}
        for name, (mh, ma) in trained.items():
            Xin = feat_vec_sc if name in SCALED_MODELS else feat_vec
            raw_h = float(mh.predict(Xin)[0])
            raw_a = float(ma.predict(Xin)[0])
            gh = max(0, round(raw_h))
            ga = max(0, round(raw_a))
            model_preds[name] = (gh, ga, raw_h, raw_a)

        rh, ra, model_agreement = weighted_ensemble_goals(model_preds)
        ens_h = max(0, round(rh))
        ens_a = max(0, round(ra))

        if verbose and grp != current_group:
            current_group = grp
            print(f"\n{'-' * 80}")
            print(f"  GROUP {grp}")
            print(f"{'-' * 80}")

        hf = FLAGS.get(home, "🏳️")
        af = FLAGS.get(away, "🏳️")

        if verbose:
            print(f"\n  Match #{mn:2d}  {home:<25} vs  {away}")
            print(f"  {'ENSEMBLE (weighted)':<22}  {ens_h} - {ens_a}   * CONSENSUS")

        html_models = {
            MODEL_HTML_NAMES[name]: {"gh": gh, "ga": ga, "rh": rh, "ra": ra}
            for name, (gh, ga, rh, ra) in model_preds.items()
        }

        envelope = build_prediction_envelope(
            home, away, model_preds,
            data_quality=0.65,
            lineup_completeness=0.5,
        )
        envelope["ensemble"]["model_agreement"] = round(model_agreement, 3)

        ml_data.append({
            "mn": mn,
            "group": grp,
            "home": home,
            "away": away,
            "home_flag": hf,
            "away_flag": af,
            "models": html_models,
            "ens_h": ens_h,
            "ens_a": ens_a,
            "ens": f"{ens_h}-{ens_a}",
            "prediction": envelope["prediction"],
            "confidence": envelope["confidence"],
            "explanation": envelope["explanation"],
            "ensemble": envelope["ensemble"],
        })
    return ml_data


def run_pipeline(verbose=True, seed=42, n_synthetic=5000):
    """Train all models and predict 72 group-stage matches. Returns JSON-ready dict."""
    from datetime import datetime, timezone

    feature_cols = get_feature_cols()
    df = generate_synthetic_dataset(n_synthetic=n_synthetic, seed=seed)
    y_h = df["goals_h"].values
    y_a = df["goals_a"].values

    if verbose:
        print(f"Training data: {len(df)} synthetic matches | {len(feature_cols)} features")
        print(f"  Avg home goals: {y_h.mean():.2f} | Avg away goals: {y_a.mean():.2f}")
        print("\nTraining all 7 models (home goals + away goals each)...")

    trained, scaler = train_models_from_frame(df, feature_cols, verbose=verbose)

    if verbose:
        print("\n" + "=" * 80)
        print("WC 2026 GROUP STAGE — PREDICTED SCORES BY MODEL")
        print("=" * 80)

    ml_data = predict_all_fixtures(trained, scaler, feature_cols, verbose=verbose)

    agree_count = sum(
        1 for m in ml_data
        if len({f"{p['gh']}-{p['ga']}" for p in m["models"].values()}) == 1
    )
    home_wins = sum(1 for m in ml_data if m["ens_h"] > m["ens_a"])
    draws = sum(1 for m in ml_data if m["ens_h"] == m["ens_a"])
    away_wins = sum(1 for m in ml_data if m["ens_h"] < m["ens_a"])
    total_goals = sum(m["ens_h"] + m["ens_a"] for m in ml_data)

    if verbose:
        print("\n\n" + "=" * 80)
        print("SUMMARY — ALL 72 MATCHES ENSEMBLE PREDICTIONS")
        print("=" * 80)
        print(f"{'#':>3}  {'Grp':^4}  {'Home':<25} {'Score':^7} {'Away':<25}")
        print("-" * 80)
        for m in ml_data:
            print(f"  {m['mn']:>2}  [{m['group']}]  {m['home']:<25} {m['ens']:^7} {m['away']}")
        print("\n\n" + "=" * 80)
        print("MODEL AGREEMENT ANALYSIS")
        print("=" * 80)
        print(f"  Matches where ALL models agree exactly: {agree_count}/{len(ml_data)}")
        print(f"  Ensemble (mean of 7 models) used as final prediction")
        print(f"\n  Ensemble outcome distribution (72 group matches):")
        print(f"    Home wins : {home_wins} ({home_wins / 72 * 100:.1f}%)")
        print(f"    Draws     : {draws}  ({draws / 72 * 100:.1f}%)")
        print(f"    Away wins : {away_wins} ({away_wins / 72 * 100:.1f}%)")
        print(f"\n  Total predicted group-stage goals: {total_goals}")
        print(f"  Average goals per match: {total_goals / 72:.2f}")
        print("=" * 80)
        print("Done.")

    return {
        "ml_data": ml_data,
        "team_elo": {team: stats["elo"] for team, stats in TEAM_STATS.items()},
        "stats": {
            "total_goals": total_goals,
            "goals_per_match": round(total_goals / 72, 2),
            "full_agree": agree_count,
            "home_wins": home_wins,
            "draws": draws,
            "away_wins": away_wins,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "training": {
            "mode": "full",
            "n_matches": n_synthetic,
            "n_features": len(feature_cols),
            "avg_home_goals": round(float(y_h.mean()), 2),
            "avg_away_goals": round(float(y_a.mean()), 2),
            "seed": seed,
            "last_trained_at": datetime.now(timezone.utc).isoformat(),
            "new_matches_used": 0,
            "total_world_cup_matches_used": 0,
            "trained_fixture_ids": [],
        },
    }


def save_predictions(path, data=None, verbose=True, incremental=True, **kwargs):
    """Save predictions — uses incremental training by default."""
    from pathlib import Path
    from model_store import models_exist

    path = Path(path)
    if incremental:
        from incremental_trainer import run_incremental_training
        force = kwargs.pop("force", not models_exist())
        result = run_incremental_training(
            force=force,
            fetch_from_api=True,
            verbose=verbose,
            predictions_path=path,
            **{k: v for k, v in kwargs.items() if k in ("seed", "n_synthetic")},
        )
        if result.get("status") == "skipped" and path.exists():
            import json
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return result

    if data is None:
        data = run_pipeline(verbose=verbose, **kwargs)
    from training_store import atomic_write_json
    atomic_write_json(path, data)
    return data


if __name__ == "__main__":
    import os
    out = os.path.join(os.path.dirname(__file__), "predictions.json")
    save_predictions(out, verbose=False)
    print(f"Saved predictions to {out}")
