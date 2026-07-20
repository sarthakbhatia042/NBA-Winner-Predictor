"""
NBA Playoff Predictor — Feature Engineering
Builds game-level features for playoff game prediction from regular-season data.
All features for a playoff game are derived from regular-season data only (no leakage).
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# =============================================================================
# Helper Functions
# =============================================================================

def _compute_win_pct(game_logs: pd.DataFrame, team_id: int) -> float:
    """Compute regular-season win percentage for a team."""
    team_games = game_logs[game_logs["TEAM_ID"] == team_id]
    if len(team_games) == 0:
        return 0.5
    wins = (team_games["WL"] == "W").sum()
    return wins / len(team_games)


def _compute_last_n_stats(game_logs: pd.DataFrame, team_id: int, n: int = 15) -> dict:
    """
    Compute stats over the last N regular-season games for a team.
    Returns dict with 'win_pct' and 'net_rating' (PTS - opponent PTS avg).
    """
    team_games = game_logs[game_logs["TEAM_ID"] == team_id].copy()
    if len(team_games) == 0:
        return {"win_pct": 0.5, "net_rating": 0.0}
    
    # Sort by game date (most recent last)
    team_games = team_games.sort_values("GAME_DATE")
    last_n = team_games.tail(n)
    
    win_pct = (last_n["WL"] == "W").sum() / len(last_n)
    
    # Net rating approximation from box score: PLUS_MINUS average
    if "PLUS_MINUS" in last_n.columns:
        net_rating = last_n["PLUS_MINUS"].mean()
    else:
        net_rating = 0.0
    
    return {"win_pct": win_pct, "net_rating": net_rating}


def _compute_h2h_win_pct(game_logs: pd.DataFrame, team_id: int,
                          opponent_id: int) -> float:
    """Compute head-to-head win percentage of team vs opponent in regular season."""
    team_games = game_logs[game_logs["TEAM_ID"] == team_id]
    
    # Find games against the specific opponent via MATCHUP column
    # or via GAME_ID matching
    opponent_game_ids = set(
        game_logs[game_logs["TEAM_ID"] == opponent_id]["GAME_ID"].values
    )
    h2h_games = team_games[team_games["GAME_ID"].isin(opponent_game_ids)]
    
    if len(h2h_games) == 0:
        return 0.5  # No data, assume neutral
    
    wins = (h2h_games["WL"] == "W").sum()
    return wins / len(h2h_games)


def _compute_sos(game_logs: pd.DataFrame, team_id: int,
                  all_win_pcts: dict) -> float:
    """
    Compute Strength of Schedule: average win percentage of all opponents faced.
    """
    team_games = game_logs[game_logs["TEAM_ID"] == team_id]
    
    # Get all opponents' GAME_IDs
    team_game_ids = set(team_games["GAME_ID"].values)
    opponents = game_logs[
        (game_logs["GAME_ID"].isin(team_game_ids)) &
        (game_logs["TEAM_ID"] != team_id)
    ]["TEAM_ID"].values
    
    if len(opponents) == 0:
        return 0.5
    
    opp_win_pcts = [all_win_pcts.get(opp_id, 0.5) for opp_id in opponents]
    return np.mean(opp_win_pcts)


def _get_team_seed(standings: pd.DataFrame, team_id: int) -> int:
    """Get a team's playoff seed from standings. Returns 0 if not found."""
    if standings is None or standings.empty:
        return 0
    
    # Try different column names that standings might use
    team_id_col = None
    for col in ["TeamID", "TEAM_ID", "team_id"]:
        if col in standings.columns:
            team_id_col = col
            break
    
    if team_id_col is None:
        return 0
    
    team_row = standings[standings[team_id_col] == team_id]
    if len(team_row) == 0:
        return 0
    
    # Try different column names for seed/rank
    for col in ["PlayoffRank", "PLAYOFF_RANK", "SeedNumber", "SEED_NUMBER",
                "ConferenceRank", "CONFERENCE_RANK"]:
        if col in standings.columns:
            seed = team_row[col].values[0]
            try:
                return int(seed)
            except (ValueError, TypeError):
                continue
    
    return 0


def _count_playoff_appearances(all_season_data: dict, team_id: int,
                                current_season: str, lookback: int = 5) -> int:
    """Count how many times a team made the playoffs in the last `lookback` years."""
    season_index = config.TRAINING_SEASONS.index(current_season) if current_season in config.TRAINING_SEASONS else -1
    
    count = 0
    seasons_to_check = config.TRAINING_SEASONS[max(0, season_index - lookback):season_index]
    
    for season in seasons_to_check:
        if season in all_season_data:
            playoff_logs = all_season_data[season].get("playoff_logs", pd.DataFrame())
            if not playoff_logs.empty and team_id in playoff_logs["TEAM_ID"].values:
                count += 1
    
    return count


def _get_advanced_stat(advanced_stats: pd.DataFrame, team_id: int,
                        stat_name: str) -> float:
    """Extract a specific advanced stat for a team."""
    if advanced_stats is None or advanced_stats.empty:
        return 0.0
    
    team_id_col = None
    for col in ["TEAM_ID", "TeamID", "team_id"]:
        if col in advanced_stats.columns:
            team_id_col = col
            break
    
    if team_id_col is None:
        return 0.0
    
    team_row = advanced_stats[advanced_stats[team_id_col] == team_id]
    if len(team_row) == 0 or stat_name not in team_row.columns:
        return 0.0
    
    val = team_row[stat_name].values[0]
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


# =============================================================================
# Main Feature Engineering
# =============================================================================

def build_team_features(season_data: dict, team_id: int,
                         all_season_data: dict, season: str) -> dict:
    """
    Build all features for a single team based on their regular-season data.
    
    Returns a dict of feature_name -> value.
    """
    game_logs = season_data["regular_season_logs"]
    advanced_stats = season_data["advanced_stats"]
    standings = season_data["standings"]
    
    # All teams' win percentages (for SOS calculation)
    all_team_ids = game_logs["TEAM_ID"].unique()
    all_win_pcts = {tid: _compute_win_pct(game_logs, tid) for tid in all_team_ids}
    
    features = {}
    
    # Core performance
    features["off_rating"] = _get_advanced_stat(advanced_stats, team_id, "OFF_RATING")
    features["def_rating"] = _get_advanced_stat(advanced_stats, team_id, "DEF_RATING")
    features["net_rating"] = _get_advanced_stat(advanced_stats, team_id, "NET_RATING")
    features["pace"] = _get_advanced_stat(advanced_stats, team_id, "PACE")
    features["win_pct"] = _compute_win_pct(game_logs, team_id)
    
    # Seed
    features["seed"] = _get_team_seed(standings, team_id)
    
    # Strength of Schedule
    features["sos"] = _compute_sos(game_logs, team_id, all_win_pcts)
    
    # Recent form
    last_n_stats = _compute_last_n_stats(game_logs, team_id, n=15)
    features["last_15_win_pct"] = last_n_stats["win_pct"]
    features["last_15_net_rating"] = last_n_stats["net_rating"]
    
    # Playoff experience
    features["playoff_appearances_5yr"] = _count_playoff_appearances(
        all_season_data, team_id, season
    )
    
    return features


def build_game_features(season_data: dict, home_team_id: int, away_team_id: int,
                         all_season_data: dict, season: str) -> dict:
    """
    Build features for a single playoff game between home and away teams.
    All features are derived from regular-season data.
    """
    game_logs = season_data["regular_season_logs"]
    
    # Get team-level features
    home_feats = build_team_features(season_data, home_team_id, all_season_data, season)
    away_feats = build_team_features(season_data, away_team_id, all_season_data, season)
    
    features = {}
    
    # Home team features (prefixed)
    for key, val in home_feats.items():
        features[f"home_{key}"] = val
    
    # Away team features (prefixed)
    for key, val in away_feats.items():
        features[f"away_{key}"] = val
    
    # Differential features (home - away)
    for feat in config.CORE_FEATURES:
        features[f"{feat}_diff"] = home_feats.get(feat, 0) - away_feats.get(feat, 0)
    
    # Matchup-specific features
    features["seed_diff"] = away_feats.get("seed", 0) - home_feats.get("seed", 0)
    features["h2h_win_pct"] = _compute_h2h_win_pct(
        game_logs, home_team_id, away_team_id
    )
    features["is_home"] = 1  # Home team perspective
    
    # Seed (of home team for reference)
    features["seed"] = home_feats.get("seed", 0)
    
    # SOS (of home team)
    features["sos"] = home_feats.get("sos", 0.5)
    
    # Form features (home team perspective)
    features["last_15_win_pct"] = home_feats.get("last_15_win_pct", 0.5)
    features["last_15_net_rating"] = home_feats.get("last_15_net_rating", 0.0)
    
    return features


def build_training_dataset(all_season_data: dict) -> pd.DataFrame:
    """
    Build the full training dataset from all training seasons.
    Each row = one playoff game with features + HOME_WIN label + season weight.
    
    Returns a DataFrame ready for model training.
    """
    all_rows = []
    
    for season in config.TRAINING_SEASONS:
        if season not in all_season_data:
            print(f"  [SKIP] No data for {season}")
            continue
        
        season_data = all_season_data[season]
        playoff_games = season_data.get("playoff_games", pd.DataFrame())
        
        if playoff_games.empty:
            print(f"  [SKIP] No playoff games for {season}")
            continue
        
        print(f"  Engineering features for {season} ({len(playoff_games)} games)...")
        
        for _, game in playoff_games.iterrows():
            try:
                features = build_game_features(
                    season_data,
                    int(game["HOME_TEAM_ID"]),
                    int(game["AWAY_TEAM_ID"]),
                    all_season_data,
                    season,
                )
                
                features["HOME_WIN"] = int(game["HOME_WIN"])
                features["SEASON"] = season
                features["GAME_ID"] = game["GAME_ID"]
                features["HOME_TEAM"] = game.get("HOME_TEAM", "")
                features["AWAY_TEAM"] = game.get("AWAY_TEAM", "")
                features["SAMPLE_WEIGHT"] = config.SEASON_WEIGHTS.get(season, 1.0)
                
                # Handle bubble season: remove home-court advantage
                if season == config.BUBBLE_SEASON:
                    features["is_home"] = 0  # Neutralize home-court
                
                all_rows.append(features)
            except Exception as e:
                print(f"    [ERROR] Game {game.get('GAME_ID', '?')}: {e}")
                continue
    
    if not all_rows:
        print("[WARNING] No training data generated!")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_rows)
    print(f"\nTraining dataset: {len(df)} games across {df['SEASON'].nunique()} seasons")
    
    return df


if __name__ == "__main__":
    # Test feature engineering for a single season
    from data.fetch_data import load_season_data
    
    print("Testing feature engineering...")
    season = "2024-25"
    data = {season: load_season_data(season)}
    
    df = build_training_dataset(data)
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFeature columns:\n{[c for c in df.columns if c not in ['HOME_WIN', 'SEASON', 'GAME_ID', 'HOME_TEAM', 'AWAY_TEAM', 'SAMPLE_WEIGHT']]}")
    print(f"\nSample row:\n{df.iloc[0]}")
