"""
NBA Playoff Predictor — Configuration
Central configuration for seasons, features, model hyperparameters, and constants.
"""

import numpy as np

# =============================================================================
# Season Configuration
# =============================================================================
TRAINING_SEASONS = [
    "2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"
]

PREDICTION_SEASON = "2025-26"

# The 2019-20 bubble season (no home-court advantage)
BUBBLE_SEASON = "2019-20"

# =============================================================================
# Exponential Decay Weighting
# =============================================================================
DECAY_FACTOR = 0.85

def get_season_weights(seasons: list) -> dict:
    """
    Compute exponential decay weights for each season.
    Most recent season gets weight 1.0, previous gets DECAY_FACTOR, etc.
    """
    n = len(seasons)
    weights = {}
    for i, season in enumerate(seasons):
        weights[season] = DECAY_FACTOR ** (n - 1 - i)
    return weights

SEASON_WEIGHTS = get_season_weights(TRAINING_SEASONS)

# =============================================================================
# Feature Configuration
# =============================================================================
CORE_FEATURES = [
    "off_rating", "def_rating", "net_rating", "pace", "win_pct"
]

MATCHUP_FEATURES = [
    "seed", "seed_diff", "sos", "h2h_win_pct", "is_home"
]

FORM_FEATURES = [
    "last_15_win_pct", "last_15_net_rating"
]

EXPERIENCE_FEATURES = [
    "playoff_appearances_5yr"
]

# Differential features are computed as home_X - away_X for core features
DIFFERENTIAL_FEATURES = [f"{feat}_diff" for feat in CORE_FEATURES]

# All features used for model training
ALL_FEATURES = (
    [f"home_{f}" for f in CORE_FEATURES] +
    [f"away_{f}" for f in CORE_FEATURES] +
    DIFFERENTIAL_FEATURES +
    MATCHUP_FEATURES +
    FORM_FEATURES +
    [f"home_{f}" for f in EXPERIENCE_FEATURES] +
    [f"away_{f}" for f in EXPERIENCE_FEATURES]
)

# =============================================================================
# Model Hyperparameters
# =============================================================================
LOGISTIC_REGRESSION_PARAMS = {
    "C": 1.0,
    "l1_ratio": 0,  # Equivalent to L2 penalty
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 42,
}

XGBOOST_PARAMS = {
    "max_depth": 4,
    "n_estimators": 200,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "eval_metric": "logloss",
}

# =============================================================================
# Playoff Structure
# =============================================================================
PLAYOFF_ROUNDS = ["First Round", "Conference Semifinals", "Conference Finals", "NBA Finals"]
CONFERENCES = ["East", "West"]

# Home-court format for best-of-7: True = higher seed is home
GAME_HOME_COURT = [True, True, False, False, True, False, True]  # 2-2-1-1-1

# =============================================================================
# Paths
# =============================================================================
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_DIR, "data", "cache")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Misc
# =============================================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# API rate limit delay (seconds)
API_DELAY = 0.6
