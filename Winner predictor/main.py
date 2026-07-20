"""
NBA Playoff Predictor — Main Entry Point
Orchestrates the full pipeline: data fetching, feature engineering,
model training, cross-validation, prediction, and bracket visualization.

Usage:
    python main.py                    # Full pipeline
    python main.py --skip-fetch       # Skip data fetching (use cache)
    python main.py --model xgboost    # Use specific model for predictions
    python main.py --model lr         # Use logistic regression for predictions
"""

import argparse
import sys
import os
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data.fetch_data import load_all_training_data, load_prediction_season_data
from features.engineer import build_training_dataset
from models.train import (
    leave_one_season_out_cv,
    train_final_models,
    _get_feature_columns,
)
from models.predict import simulate_bracket
from visualization.bracket import render_bracket


def main():
    parser = argparse.ArgumentParser(description="NBA Playoff Predictor")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip data fetching and use cached data")
    parser.add_argument("--model", type=str, default="xgb", choices=["xgb", "lr"],
                        help="Model to use for predictions (xgb or lr)")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Force re-download of all data from API")
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("=" * 70)
    print("  🏀  NBA PLAYOFF PREDICTOR  🏀")
    print(f"  Training seasons: {config.TRAINING_SEASONS[0]} to {config.TRAINING_SEASONS[-1]}")
    print(f"  Prediction season: {config.PREDICTION_SEASON}")
    print(f"  Model: {'XGBoost' if args.model == 'xgb' else 'Logistic Regression'}")
    print("=" * 70)
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 1: Fetch Data
    # ═══════════════════════════════════════════════════════════════════
    print("\n\n📊 STEP 1: FETCHING DATA")
    print("-" * 70)
    
    all_training_data = load_all_training_data(force_refresh=args.force_refresh)
    
    print(f"\nLoaded data for {len(all_training_data)} training seasons")
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 2: Feature Engineering
    # ═══════════════════════════════════════════════════════════════════
    print("\n\n⚙️  STEP 2: FEATURE ENGINEERING")
    print("-" * 70)
    
    training_df = build_training_dataset(all_training_data)
    
    if training_df.empty:
        print("\n[ERROR] No training data generated. Exiting.")
        sys.exit(1)
    
    feature_cols = _get_feature_columns(training_df)
    print(f"\nUsing {len(feature_cols)} features: {feature_cols}")
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 3: Cross-Validation
    # ═══════════════════════════════════════════════════════════════════
    print("\n\n📈 STEP 3: CROSS-VALIDATION")
    print("-" * 70)
    
    cv_results = leave_one_season_out_cv(training_df, feature_cols)
    
    # Save CV results
    cv_path = os.path.join(config.OUTPUT_DIR, "cv_results.csv")
    cv_results.to_csv(cv_path, index=False)
    print(f"\nCV results saved to {cv_path}")
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 4: Train Final Models
    # ═══════════════════════════════════════════════════════════════════
    print("\n\n🧠 STEP 4: TRAINING FINAL MODELS")
    print("-" * 70)
    
    models = train_final_models(training_df, feature_cols)
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 5: Load Prediction Season Data
    # ═══════════════════════════════════════════════════════════════════
    print("\n\n🔮 STEP 5: LOADING PREDICTION SEASON DATA")
    print("-" * 70)
    
    prediction_data = load_prediction_season_data(force_refresh=args.force_refresh)
    
    # Add to all_training_data for feature engineering (needs historical context)
    all_data_with_current = {**all_training_data}
    all_data_with_current[config.PREDICTION_SEASON] = prediction_data
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 6: Simulate Bracket
    # ═══════════════════════════════════════════════════════════════════
    print("\n\n🏆 STEP 6: SIMULATING PLAYOFF BRACKET")
    print("-" * 70)
    
    bracket = simulate_bracket(
        models, prediction_data, all_data_with_current,
        config.PREDICTION_SEASON, model_type=args.model
    )
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 7: Render Bracket Visualization
    # ═══════════════════════════════════════════════════════════════════
    print("\n\n🎨 STEP 7: RENDERING BRACKET VISUALIZATION")
    print("-" * 70)
    
    model_display = "XGBoost" if args.model == "xgb" else "Logistic Regression"
    bracket_path = render_bracket(
        bracket, config.PREDICTION_SEASON, model_type=model_display
    )
    
    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    elapsed = time.time() - start_time
    
    print("\n\n" + "=" * 70)
    print("  ✅  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Time elapsed: {elapsed:.1f}s")
    print(f"  Training games: {len(training_df)}")
    print(f"  Features used: {len(feature_cols)}")
    print(f"  Bracket saved: {bracket_path}")
    
    if bracket.get("champion"):
        print(f"\n  🏆 PREDICTED CHAMPION: {bracket['champion']}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
