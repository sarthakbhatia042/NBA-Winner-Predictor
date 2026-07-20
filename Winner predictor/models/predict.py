"""
NBA Playoff Predictor — Game Prediction & Bracket Simulation
Predicts individual playoff games and simulates the full bracket.
Incorporates actual playoff results for completed/in-progress series.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from features.engineer import build_game_features
from data.fetch_data import get_team_names


# =============================================================================
# Game-Level Prediction
# =============================================================================

def predict_game(models: dict, season_data: dict, all_season_data: dict,
                 season: str, home_team_id: int, away_team_id: int,
                 model_type: str = "xgb") -> dict:
    """
    Predict the outcome of a single game.
    
    Returns dict with:
        - home_win_prob: probability of home team winning
        - predicted_winner: team_id of predicted winner
        - features: the computed feature dict
    """
    features = build_game_features(
        season_data, home_team_id, away_team_id,
        all_season_data, season
    )
    
    feature_cols = models["feature_cols"]
    X = np.array([[features.get(col, 0) for col in feature_cols]])
    
    if model_type == "lr":
        scaler = models["scaler"]
        X_scaled = scaler.transform(X)
        prob = models["lr"].predict_proba(X_scaled)[0][1]
    else:
        prob = models["xgb"].predict_proba(X)[0][1]
    
    winner = home_team_id if prob > 0.5 else away_team_id
    
    return {
        "home_win_prob": prob,
        "away_win_prob": 1 - prob,
        "predicted_winner": winner,
        "features": features,
    }


# =============================================================================
# Actual Playoff Results Parser
# =============================================================================

def parse_actual_playoff_results(season_data: dict) -> dict:
    """
    Parse actual playoff game logs to determine completed and in-progress series.
    
    Returns a dict keyed by frozenset({team_id_1, team_id_2}) with values:
        {
            "team1_id": int, "team1_abbr": str, "team1_wins": int,
            "team2_id": int, "team2_abbr": str, "team2_wins": int,
            "games_played": int,
            "is_complete": bool,
            "winner": team_id or None,
        }
    """
    playoff_logs = season_data.get("playoff_logs", pd.DataFrame())
    
    if playoff_logs.empty:
        return {}
    
    # Split into home and away
    home_games = playoff_logs[playoff_logs["MATCHUP"].str.contains("vs.", na=False)].copy()
    away_games = playoff_logs[playoff_logs["MATCHUP"].str.contains("@", na=False)].copy()
    
    # Build per-game view
    home_r = home_games[["GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_ABBREVIATION", "PTS", "WL"]].rename(
        columns={"TEAM_ID": "HOME_TEAM_ID", "TEAM_ABBREVIATION": "HOME_TEAM",
                 "PTS": "HOME_PTS", "WL": "HOME_WL"}
    )
    away_r = away_games[["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "PTS", "WL"]].rename(
        columns={"TEAM_ID": "AWAY_TEAM_ID", "TEAM_ABBREVIATION": "AWAY_TEAM",
                 "PTS": "AWAY_PTS", "WL": "AWAY_WL"}
    )
    games = home_r.merge(away_r, on="GAME_ID")
    games = games.sort_values("GAME_DATE")
    
    # Group by the pair of teams
    series_results = {}
    
    games["team_pair"] = games.apply(
        lambda r: frozenset([int(r["HOME_TEAM_ID"]), int(r["AWAY_TEAM_ID"])]), axis=1
    )
    
    for pair, grp in games.groupby("team_pair"):
        team_ids = list(pair)
        t1_id, t2_id = team_ids[0], team_ids[1]
        
        # Get abbreviations
        t1_abbr = grp.loc[grp["HOME_TEAM_ID"] == t1_id, "HOME_TEAM"].values
        if len(t1_abbr) == 0:
            t1_abbr = grp.loc[grp["AWAY_TEAM_ID"] == t1_id, "AWAY_TEAM"].values
        t1_abbr = t1_abbr[0] if len(t1_abbr) > 0 else str(t1_id)
        
        t2_abbr = grp.loc[grp["HOME_TEAM_ID"] == t2_id, "HOME_TEAM"].values
        if len(t2_abbr) == 0:
            t2_abbr = grp.loc[grp["AWAY_TEAM_ID"] == t2_id, "AWAY_TEAM"].values
        t2_abbr = t2_abbr[0] if len(t2_abbr) > 0 else str(t2_id)
        
        # Count wins
        t1_wins = (
            ((grp["HOME_TEAM_ID"] == t1_id) & (grp["HOME_WL"] == "W")).sum() +
            ((grp["AWAY_TEAM_ID"] == t1_id) & (grp["AWAY_WL"] == "W")).sum()
        )
        t2_wins = (
            ((grp["HOME_TEAM_ID"] == t2_id) & (grp["HOME_WL"] == "W")).sum() +
            ((grp["AWAY_TEAM_ID"] == t2_id) & (grp["AWAY_WL"] == "W")).sum()
        )
        
        is_complete = max(t1_wins, t2_wins) == 4
        winner = None
        if is_complete:
            winner = t1_id if t1_wins == 4 else t2_id
        
        series_results[pair] = {
            "team1_id": t1_id, "team1_abbr": t1_abbr, "team1_wins": int(t1_wins),
            "team2_id": t2_id, "team2_abbr": t2_abbr, "team2_wins": int(t2_wins),
            "games_played": len(grp),
            "is_complete": is_complete,
            "winner": winner,
        }
    
    return series_results


def find_series_result(actual_results: dict, team1_id: int, team2_id: int) -> dict:
    """Look up the actual result for a series between two teams."""
    key = frozenset([team1_id, team2_id])
    return actual_results.get(key, None)


# =============================================================================
# Series Simulation (with actual results)
# =============================================================================

def simulate_series(models: dict, season_data: dict, all_season_data: dict,
                    season: str, higher_seed_id: int, lower_seed_id: int,
                    model_type: str = "xgb",
                    actual_results: dict = None) -> dict:
    """
    Simulate a best-of-7 playoff series using game-by-game predictions.
    If actual results exist, uses them:
      - Complete series: returns actual result directly
      - In-progress series: uses actual wins so far, predicts remaining games
      - No data: predicts all games
    
    Home court follows 2-2-1-1-1 format (higher seed has home court).
    """
    # Check for actual results
    actual = None
    if actual_results:
        actual = find_series_result(actual_results, higher_seed_id, lower_seed_id)
    
    # ── CASE 1: Completed series — use actual result ──
    if actual and actual["is_complete"]:
        winner_id = actual["winner"]
        
        # Determine wins for higher/lower seed
        if actual["team1_id"] == higher_seed_id:
            h_wins = actual["team1_wins"]
            l_wins = actual["team2_wins"]
        else:
            h_wins = actual["team2_wins"]
            l_wins = actual["team1_wins"]
        
        return {
            "winner": winner_id,
            "loser": lower_seed_id if winner_id == higher_seed_id else higher_seed_id,
            "games_played": actual["games_played"],
            "series_score": (h_wins, l_wins),
            "game_results": [],
            "win_probability": 1.0 if winner_id == higher_seed_id else 0.0,
            "higher_seed_id": higher_seed_id,
            "lower_seed_id": lower_seed_id,
            "is_actual": True,
            "is_in_progress": False,
        }
    
    # ── CASE 2: In-progress series — use actual wins, predict remaining ──
    if actual and not actual["is_complete"]:
        if actual["team1_id"] == higher_seed_id:
            higher_wins = actual["team1_wins"]
            lower_wins = actual["team2_wins"]
        else:
            higher_wins = actual["team2_wins"]
            lower_wins = actual["team1_wins"]
        
        games_played = actual["games_played"]
        game_results = []
        all_probs = []
        
        # Predict remaining games starting from game_num = games_played
        for game_num in range(games_played, 7):
            if higher_wins == 4 or lower_wins == 4:
                break
            
            higher_is_home = config.GAME_HOME_COURT[game_num]
            
            if higher_is_home:
                home_id = higher_seed_id
                away_id = lower_seed_id
            else:
                home_id = lower_seed_id
                away_id = higher_seed_id
            
            result = predict_game(
                models, season_data, all_season_data, season,
                home_id, away_id, model_type
            )
            
            if higher_is_home:
                higher_seed_win_prob = result["home_win_prob"]
            else:
                higher_seed_win_prob = result["away_win_prob"]
            
            all_probs.append(higher_seed_win_prob)
            
            if higher_seed_win_prob > 0.5:
                higher_wins += 1
                game_winner = higher_seed_id
            else:
                lower_wins += 1
                game_winner = lower_seed_id
            
            game_results.append({
                "game_number": game_num + 1,
                "home_team_id": home_id,
                "away_team_id": away_id,
                "home_win_prob": result["home_win_prob"],
                "winner": game_winner,
                "higher_seed_win_prob": higher_seed_win_prob,
                "predicted": True,
            })
        
        winner = higher_seed_id if higher_wins >= 4 else lower_seed_id
        series_prob = np.mean(all_probs) if all_probs else 0.5
        
        return {
            "winner": winner,
            "loser": lower_seed_id if winner == higher_seed_id else higher_seed_id,
            "games_played": higher_wins + lower_wins,
            "series_score": (higher_wins, lower_wins),
            "game_results": game_results,
            "win_probability": series_prob,
            "higher_seed_id": higher_seed_id,
            "lower_seed_id": lower_seed_id,
            "is_actual": False,
            "is_in_progress": True,
            "actual_score_before_prediction": (
                actual["team1_wins"] if actual["team1_id"] == higher_seed_id else actual["team2_wins"],
                actual["team2_wins"] if actual["team1_id"] == higher_seed_id else actual["team1_wins"],
            ),
        }
    
    # ── CASE 3: No actual data — predict all games ──
    higher_wins = 0
    lower_wins = 0
    game_results = []
    all_probs = []
    
    for game_num in range(7):
        if higher_wins == 4 or lower_wins == 4:
            break
        
        higher_is_home = config.GAME_HOME_COURT[game_num]
        
        if higher_is_home:
            home_id = higher_seed_id
            away_id = lower_seed_id
        else:
            home_id = lower_seed_id
            away_id = higher_seed_id
        
        if season == config.BUBBLE_SEASON:
            home_id = higher_seed_id
            away_id = lower_seed_id
        
        result = predict_game(
            models, season_data, all_season_data, season,
            home_id, away_id, model_type
        )
        
        if higher_is_home or season == config.BUBBLE_SEASON:
            higher_seed_win_prob = result["home_win_prob"]
        else:
            higher_seed_win_prob = result["away_win_prob"]
        
        all_probs.append(higher_seed_win_prob)
        
        if higher_seed_win_prob > 0.5:
            higher_wins += 1
            game_winner = higher_seed_id
        else:
            lower_wins += 1
            game_winner = lower_seed_id
        
        game_results.append({
            "game_number": game_num + 1,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_win_prob": result["home_win_prob"],
            "winner": game_winner,
            "higher_seed_win_prob": higher_seed_win_prob,
            "predicted": True,
        })
    
    winner = higher_seed_id if higher_wins >= 4 else lower_seed_id
    series_prob = np.mean(all_probs) if all_probs else 0.5
    
    return {
        "winner": winner,
        "loser": lower_seed_id if winner == higher_seed_id else higher_seed_id,
        "games_played": higher_wins + lower_wins,
        "series_score": (higher_wins, lower_wins),
        "game_results": game_results,
        "win_probability": series_prob,
        "higher_seed_id": higher_seed_id,
        "lower_seed_id": lower_seed_id,
        "is_actual": False,
        "is_in_progress": False,
    }


# =============================================================================
# Full Bracket Simulation
# =============================================================================

def get_playoff_seedings(season_data: dict) -> dict:
    """
    Extract playoff seedings from standings.
    Returns dict with 'East' and 'West' keys, each containing
    a list of (seed, team_id, team_name) tuples sorted by seed.
    """
    standings = season_data["standings"]
    team_names = get_team_names()
    
    seedings = {"East": [], "West": []}
    
    if standings is None or standings.empty:
        return seedings
    
    # Find relevant columns
    team_id_col = None
    for col in ["TeamID", "TEAM_ID", "team_id"]:
        if col in standings.columns:
            team_id_col = col
            break
    
    conf_col = None
    for col in ["Conference", "CONFERENCE", "conference"]:
        if col in standings.columns:
            conf_col = col
            break
    
    seed_col = None
    for col in ["PlayoffRank", "PLAYOFF_RANK", "SeedNumber", "SEED_NUMBER",
                "ConferenceRank", "CONFERENCE_RANK"]:
        if col in standings.columns:
            seed_col = col
            break
    
    if not all([team_id_col, conf_col, seed_col]):
        print(f"  [WARNING] Could not find required columns in standings.")
        print(f"  Available columns: {list(standings.columns)}")
        return seedings
    
    for _, row in standings.iterrows():
        try:
            team_id = int(row[team_id_col])
            conf = row[conf_col]
            seed = int(row[seed_col])
        except (ValueError, TypeError):
            continue
        
        if seed < 1 or seed > 8:
            continue
        
        team_name = team_names.get(team_id, f"Team {team_id}")
        
        conf_key = "East" if "East" in conf else "West" if "West" in conf else None
        if conf_key:
            seedings[conf_key].append((seed, team_id, team_name))
    
    # Sort by seed
    for conf in seedings:
        seedings[conf].sort(key=lambda x: x[0])
    
    return seedings


def _format_series_line(h_seed, h_name, l_seed, l_name, result) -> str:
    """Format a series result line for console output."""
    winner_name = result["winner_name"]
    score = result["series_score"]
    is_actual = result.get("is_actual", False)
    is_in_progress = result.get("is_in_progress", False)
    
    if is_actual:
        tag = "  [ACTUAL]"
    elif is_in_progress:
        actual_before = result.get("actual_score_before_prediction", (0, 0))
        tag = f"  [WAS {actual_before[0]}-{actual_before[1]}, PREDICTED REST]"
    else:
        tag = f"  [PREDICTED] (prob: {result['win_probability']:.1%})"
    
    return (f"    ({h_seed}) {h_name} vs ({l_seed}) {l_name}: "
            f"{winner_name} wins {score[0]}-{score[1]}{tag}")


def simulate_bracket(models: dict, season_data: dict, all_season_data: dict,
                     season: str, model_type: str = "xgb") -> dict:
    """
    Simulate the entire NBA playoff bracket.
    Uses actual results for completed/in-progress series and predicts the rest.
    """
    team_names = get_team_names()
    seedings = get_playoff_seedings(season_data)
    
    # Parse actual playoff results
    actual_results = parse_actual_playoff_results(season_data)
    
    # Print actual results summary
    print("\n" + "=" * 70)
    print(f"PLAYOFF STATE — {season}")
    print("=" * 70)
    
    complete_count = sum(1 for v in actual_results.values() if v["is_complete"])
    in_progress_count = sum(1 for v in actual_results.values() if not v["is_complete"])
    
    print(f"  Actual series found: {len(actual_results)} "
          f"({complete_count} complete, {in_progress_count} in progress)")
    
    for key, sr in actual_results.items():
        status = "COMPLETE" if sr["is_complete"] else "IN PROGRESS"
        print(f"    {sr['team1_abbr']} {sr['team1_wins']}-{sr['team2_wins']} "
              f"{sr['team2_abbr']}  [{status}]")
    
    print("\n" + "=" * 70)
    print(f"SIMULATING {season} NBA PLAYOFF BRACKET")
    print("=" * 70)
    
    bracket = {"East": {}, "West": {}, "Finals": None, "champion": None}
    conf_champions = {}
    
    for conf in ["East", "West"]:
        seeds = seedings[conf]
        if len(seeds) < 8:
            print(f"  [WARNING] {conf} has only {len(seeds)} teams seeded. "
                  f"Need 8. Skipping.")
            continue
        
        print(f"\n{'─'*35} {conf}ern Conference {'─'*35}")
        
        # ── First Round ──
        # 1 vs 8, 2 vs 7, 3 vs 6, 4 vs 5
        first_round_matchups = [
            (seeds[0], seeds[7]),  # 1 vs 8
            (seeds[3], seeds[4]),  # 4 vs 5
            (seeds[2], seeds[5]),  # 3 vs 6
            (seeds[1], seeds[6]),  # 2 vs 7
        ]
        
        first_round_results = []
        print(f"\n  FIRST ROUND:")
        for (h_seed, h_id, h_name), (l_seed, l_id, l_name) in first_round_matchups:
            result = simulate_series(
                models, season_data, all_season_data, season,
                h_id, l_id, model_type,
                actual_results=actual_results,
            )
            result["higher_seed_name"] = h_name
            result["lower_seed_name"] = l_name
            result["higher_seed_num"] = h_seed
            result["lower_seed_num"] = l_seed
            winner_name = h_name if result["winner"] == h_id else l_name
            result["winner_name"] = winner_name
            
            first_round_results.append(result)
            print(_format_series_line(h_seed, h_name, l_seed, l_name, result))
        
        bracket[conf]["First Round"] = first_round_results
        
        # ── Conference Semifinals ──
        semi_matchups = [
            (first_round_results[0], first_round_results[1]),
            (first_round_results[2], first_round_results[3]),
        ]
        
        semi_results = []
        print(f"\n  CONFERENCE SEMIFINALS:")
        for r1, r2 in semi_matchups:
            w1_seed = r1["higher_seed_num"] if r1["winner"] == r1["higher_seed_id"] else r1["lower_seed_num"]
            w2_seed = r2["higher_seed_num"] if r2["winner"] == r2["higher_seed_id"] else r2["lower_seed_num"]
            w1_name = r1["winner_name"]
            w2_name = r2["winner_name"]
            
            if w1_seed < w2_seed:
                higher_id, lower_id = r1["winner"], r2["winner"]
                higher_name, lower_name = w1_name, w2_name
                higher_seed_num, lower_seed_num = w1_seed, w2_seed
            else:
                higher_id, lower_id = r2["winner"], r1["winner"]
                higher_name, lower_name = w2_name, w1_name
                higher_seed_num, lower_seed_num = w2_seed, w1_seed
            
            result = simulate_series(
                models, season_data, all_season_data, season,
                higher_id, lower_id, model_type,
                actual_results=actual_results,
            )
            result["higher_seed_name"] = higher_name
            result["lower_seed_name"] = lower_name
            result["higher_seed_num"] = higher_seed_num
            result["lower_seed_num"] = lower_seed_num
            winner_name = higher_name if result["winner"] == higher_id else lower_name
            result["winner_name"] = winner_name
            
            semi_results.append(result)
            print(_format_series_line(higher_seed_num, higher_name, lower_seed_num, lower_name, result))
        
        bracket[conf]["Conference Semifinals"] = semi_results
        
        # ── Conference Finals ──
        print(f"\n  CONFERENCE FINALS:")
        r1, r2 = semi_results[0], semi_results[1]
        w1_seed = r1["higher_seed_num"] if r1["winner"] == r1["higher_seed_id"] else r1["lower_seed_num"]
        w2_seed = r2["higher_seed_num"] if r2["winner"] == r2["higher_seed_id"] else r2["lower_seed_num"]
        w1_name = r1["winner_name"]
        w2_name = r2["winner_name"]
        
        if w1_seed < w2_seed:
            higher_id, lower_id = r1["winner"], r2["winner"]
            higher_name, lower_name = w1_name, w2_name
            higher_seed_num, lower_seed_num = w1_seed, w2_seed
        else:
            higher_id, lower_id = r2["winner"], r1["winner"]
            higher_name, lower_name = w2_name, w1_name
            higher_seed_num, lower_seed_num = w2_seed, w1_seed
        
        result = simulate_series(
            models, season_data, all_season_data, season,
            higher_id, lower_id, model_type,
            actual_results=actual_results,
        )
        result["higher_seed_name"] = higher_name
        result["lower_seed_name"] = lower_name
        result["higher_seed_num"] = higher_seed_num
        result["lower_seed_num"] = lower_seed_num
        winner_name = higher_name if result["winner"] == higher_id else lower_name
        result["winner_name"] = winner_name
        
        bracket[conf]["Conference Finals"] = [result]
        conf_champions[conf] = result
        
        print(_format_series_line(higher_seed_num, higher_name, lower_seed_num, lower_name, result))
        print(f"\n  >> {conf}ern Conference Champion: {winner_name}")
    
    # ── NBA Finals ──
    if "East" in conf_champions and "West" in conf_champions:
        print(f"\n{'='*35} NBA FINALS {'='*35}")
        
        east_champ = conf_champions["East"]
        west_champ = conf_champions["West"]
        
        east_id = east_champ["winner"]
        west_id = west_champ["winner"]
        east_name = east_champ["winner_name"]
        west_name = west_champ["winner_name"]
        
        east_seed = east_champ["higher_seed_num"] if east_champ["winner"] == east_champ["higher_seed_id"] else east_champ["lower_seed_num"]
        west_seed = west_champ["higher_seed_num"] if west_champ["winner"] == west_champ["higher_seed_id"] else west_champ["lower_seed_num"]
        
        if east_seed <= west_seed:
            higher_id, lower_id = east_id, west_id
            higher_name, lower_name = east_name, west_name
        else:
            higher_id, lower_id = west_id, east_id
            higher_name, lower_name = west_name, east_name
        
        result = simulate_series(
            models, season_data, all_season_data, season,
            higher_id, lower_id, model_type,
            actual_results=actual_results,
        )
        result["higher_seed_name"] = higher_name
        result["lower_seed_name"] = lower_name
        result["east_team"] = east_name
        result["west_team"] = west_name
        winner_name = higher_name if result["winner"] == higher_id else lower_name
        result["winner_name"] = winner_name
        
        bracket["Finals"] = result
        bracket["champion"] = winner_name
        bracket["champion_id"] = result["winner"]
        
        score = result["series_score"]
        is_actual = result.get("is_actual", False)
        tag = " [ACTUAL]" if is_actual else " [PREDICTED]"
        
        print(f"\n    {east_name} (East) vs {west_name} (West)")
        print(f"    {winner_name} wins {score[0]}-{score[1]}{tag}")
        print(f"\n    >>> NBA CHAMPION: {winner_name} <<<")
    
    return bracket
