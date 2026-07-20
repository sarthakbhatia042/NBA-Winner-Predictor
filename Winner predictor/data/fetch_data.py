"""
NBA Playoff Predictor — Data Fetching & Caching
Pulls regular-season and playoff data from nba_api for all teams across
the configured seasons. Caches results as CSVs to avoid repeated API calls.
"""

import os
import time
import pandas as pd
from nba_api.stats.static import teams as nba_teams
from nba_api.stats.endpoints import (
    leaguegamefinder,
    leaguedashteamstats,
    leaguestandings,
)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def get_all_team_ids() -> dict:
    """Return a dict mapping team abbreviation -> team_id for all 30 NBA teams."""
    all_teams = nba_teams.get_teams()
    return {t["abbreviation"]: t["id"] for t in all_teams}


def get_team_names() -> dict:
    """Return a dict mapping team_id -> full team name."""
    all_teams = nba_teams.get_teams()
    return {t["id"]: t["full_name"] for t in all_teams}


def _cache_path(filename: str) -> str:
    """Return the full path to a cache file."""
    return os.path.join(config.CACHE_DIR, filename)


def _load_or_fetch(cache_file: str, fetch_fn, force_refresh: bool = False) -> pd.DataFrame:
    """
    Load data from cache if it exists, otherwise fetch and cache it.
    """
    path = _cache_path(cache_file)
    if os.path.exists(path) and not force_refresh:
        print(f"  [CACHE] Loading {cache_file}")
        return pd.read_csv(path)
    
    print(f"  [API] Fetching {cache_file}...")
    df = fetch_fn()
    df.to_csv(path, index=False)
    time.sleep(config.API_DELAY)
    return df


# =============================================================================
# Fetching Functions
# =============================================================================

def fetch_game_logs(season: str, season_type: str = "Regular Season",
                    force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch all game logs for a given season and season type.
    Returns a DataFrame with one row per team per game.
    
    Parameters
    ----------
    season : str
        e.g. "2024-25"
    season_type : str
        "Regular Season" or "Playoffs"
    force_refresh : bool
        If True, ignore cache and re-fetch from API
    """
    type_tag = season_type.replace(" ", "_").lower()
    cache_file = f"game_logs_{season}_{type_tag}.csv"
    
    def _fetch():
        finder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            season_type_nullable=season_type,
            league_id_nullable="00",  # NBA
        )
        df = finder.get_data_frames()[0]
        return df
    
    return _load_or_fetch(cache_file, _fetch, force_refresh)


def fetch_advanced_team_stats(season: str, season_type: str = "Regular Season",
                               force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch advanced team stats (OFF_RATING, DEF_RATING, NET_RATING, PACE, etc.)
    for all teams in a given season.
    """
    type_tag = season_type.replace(" ", "_").lower()
    cache_file = f"advanced_stats_{season}_{type_tag}.csv"
    
    def _fetch():
        stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star=season_type,
            measure_type_detailed_defense="Advanced",
        )
        df = stats.get_data_frames()[0]
        return df
    
    return _load_or_fetch(cache_file, _fetch, force_refresh)


def fetch_standings(season: str, force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch league standings for a given season to get playoff seedings.
    """
    cache_file = f"standings_{season}.csv"
    
    def _fetch():
        standings = leaguestandings.LeagueStandings(
            season=season,
            league_id="00",
        )
        df = standings.get_data_frames()[0]
        return df
    
    return _load_or_fetch(cache_file, _fetch, force_refresh)


# =============================================================================
# Playoff Matchup Extraction
# =============================================================================

def extract_playoff_games(game_logs: pd.DataFrame) -> pd.DataFrame:
    """
    From playoff game logs, extract individual games with home/away team info.
    Each game appears twice in game_logs (once per team).
    We deduplicate to get one row per game with home and away team columns.
    """
    if game_logs.empty:
        return pd.DataFrame()
    
    # Split into home and away based on MATCHUP column
    # Home games have "vs." in MATCHUP, away games have "@"
    home_games = game_logs[game_logs["MATCHUP"].str.contains("vs.", na=False)].copy()
    away_games = game_logs[game_logs["MATCHUP"].str.contains("@", na=False)].copy()
    
    # Rename columns for merge
    home_cols = {
        "TEAM_ID": "HOME_TEAM_ID",
        "TEAM_ABBREVIATION": "HOME_TEAM",
        "GAME_ID": "GAME_ID",
        "GAME_DATE": "GAME_DATE",
        "WL": "HOME_WL",
        "PTS": "HOME_PTS",
    }
    away_cols = {
        "TEAM_ID": "AWAY_TEAM_ID",
        "TEAM_ABBREVIATION": "AWAY_TEAM",
        "WL": "AWAY_WL",
        "PTS": "AWAY_PTS",
    }
    
    home_df = home_games.rename(columns=home_cols)[list(home_cols.values())]
    away_df = away_games.rename(columns=away_cols)
    away_df = away_df[["GAME_ID"] + [c for c in away_cols.values()]]
    
    # Merge on GAME_ID
    games = home_df.merge(away_df, on="GAME_ID", how="inner")
    games["HOME_WIN"] = (games["HOME_WL"] == "W").astype(int)
    
    return games


# =============================================================================
# High-Level Data Loader
# =============================================================================

def load_season_data(season: str, force_refresh: bool = False) -> dict:
    """
    Load all data needed for a single season.
    
    Returns a dict with keys:
        - 'regular_season_logs': game-by-game regular season logs
        - 'playoff_logs': game-by-game playoff logs
        - 'playoff_games': deduplicated playoff games (home vs away)
        - 'advanced_stats': team-level advanced stats
        - 'standings': league standings with seeds
    """
    print(f"\n{'='*60}")
    print(f"Loading data for season {season}")
    print(f"{'='*60}")
    
    data = {}
    
    # Regular season game logs
    data["regular_season_logs"] = fetch_game_logs(
        season, "Regular Season", force_refresh
    )
    
    # Playoff game logs
    data["playoff_logs"] = fetch_game_logs(
        season, "Playoffs", force_refresh
    )
    
    # Extract playoff games (deduplicated)
    data["playoff_games"] = extract_playoff_games(data["playoff_logs"])
    
    # Advanced team stats (regular season)
    data["advanced_stats"] = fetch_advanced_team_stats(
        season, "Regular Season", force_refresh
    )
    
    # Standings
    data["standings"] = fetch_standings(season, force_refresh)
    
    return data


def load_all_training_data(force_refresh: bool = False) -> dict:
    """
    Load data for all training seasons.
    Returns a dict mapping season -> season_data dict.
    """
    all_data = {}
    for season in config.TRAINING_SEASONS:
        all_data[season] = load_season_data(season, force_refresh)
    return all_data


def load_prediction_season_data(force_refresh: bool = False) -> dict:
    """
    Load data for the prediction season (current season).
    """
    return load_season_data(config.PREDICTION_SEASON, force_refresh)


if __name__ == "__main__":
    # Test data fetching for the most recent training season
    print("Testing data fetch for one season...")
    data = load_season_data("2024-25")
    
    print(f"\nRegular season logs: {len(data['regular_season_logs'])} rows")
    print(f"Playoff logs: {len(data['playoff_logs'])} rows")
    print(f"Playoff games: {len(data['playoff_games'])} rows")
    print(f"Advanced stats: {len(data['advanced_stats'])} rows")
    print(f"Standings: {len(data['standings'])} rows")
