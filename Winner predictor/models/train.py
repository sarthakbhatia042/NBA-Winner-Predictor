"""
NBA Playoff Predictor — Model Training & Cross-Validation
Trains Logistic Regression and XGBoost models with leave-one-season-out CV.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# =============================================================================
# Feature Preparation
# =============================================================================

def _get_feature_columns(df: pd.DataFrame) -> list:
    """Get the list of feature columns from the training DataFrame."""
    exclude = {"HOME_WIN", "SEASON", "GAME_ID", "HOME_TEAM", "AWAY_TEAM",
               "SAMPLE_WEIGHT"}
    feature_cols = [c for c in config.ALL_FEATURES if c in df.columns]
    
    if not feature_cols:
        # Fallback: use all numeric columns except excluded
        feature_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude
        ]
    
    return feature_cols


def prepare_features(df: pd.DataFrame, feature_cols: list):
    """Extract feature matrix X, labels y, and sample weights."""
    X = df[feature_cols].fillna(0).values
    y = df["HOME_WIN"].values
    weights = df["SAMPLE_WEIGHT"].values if "SAMPLE_WEIGHT" in df.columns else None
    return X, y, weights


# =============================================================================
# Model Builders
# =============================================================================

def build_logistic_regression() -> LogisticRegression:
    """Create a Logistic Regression model with configured hyperparameters."""
    return LogisticRegression(**config.LOGISTIC_REGRESSION_PARAMS)


def build_xgboost() -> XGBClassifier:
    """Create an XGBoost classifier with configured hyperparameters."""
    return XGBClassifier(**config.XGBOOST_PARAMS)


# =============================================================================
# Leave-One-Season-Out Cross-Validation
# =============================================================================

def leave_one_season_out_cv(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Perform leave-one-season-out cross-validation for both models.
    
    For each season S:
        - Train on all seasons except S
        - Test on season S
        - Record accuracy, log-loss, and AUC
    
    Returns a DataFrame with CV results per season per model.
    """
    seasons = sorted(df["SEASON"].unique())
    results = []
    
    print("\n" + "=" * 70)
    print("LEAVE-ONE-SEASON-OUT CROSS-VALIDATION")
    print("=" * 70)
    
    for test_season in seasons:
        train_df = df[df["SEASON"] != test_season]
        test_df = df[df["SEASON"] == test_season]
        
        if len(test_df) == 0 or len(train_df) == 0:
            print(f"  [SKIP] {test_season}: insufficient data")
            continue
        
        X_train, y_train, w_train = prepare_features(train_df, feature_cols)
        X_test, y_test, _ = prepare_features(test_df, feature_cols)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # --- Logistic Regression ---
        lr_model = build_logistic_regression()
        lr_model.fit(X_train_scaled, y_train, sample_weight=w_train)
        lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]
        lr_preds = lr_model.predict(X_test_scaled)
        
        lr_acc = accuracy_score(y_test, lr_preds)
        lr_logloss = log_loss(y_test, lr_probs)
        lr_auc = roc_auc_score(y_test, lr_probs) if len(np.unique(y_test)) > 1 else 0.5
        
        # --- XGBoost ---
        xgb_model = build_xgboost()
        xgb_model.fit(X_train, y_train, sample_weight=w_train)
        xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
        xgb_preds = xgb_model.predict(X_test)
        
        xgb_acc = accuracy_score(y_test, xgb_preds)
        xgb_logloss = log_loss(y_test, xgb_probs)
        xgb_auc = roc_auc_score(y_test, xgb_probs) if len(np.unique(y_test)) > 1 else 0.5
        
        results.append({
            "Season": test_season,
            "N_Games": len(test_df),
            "LR_Accuracy": lr_acc,
            "LR_LogLoss": lr_logloss,
            "LR_AUC": lr_auc,
            "XGB_Accuracy": xgb_acc,
            "XGB_LogLoss": xgb_logloss,
            "XGB_AUC": xgb_auc,
        })
        
        print(f"  {test_season} ({len(test_df):>3} games) | "
              f"LR: {lr_acc:.3f} acc, {lr_logloss:.3f} ll | "
              f"XGB: {xgb_acc:.3f} acc, {xgb_logloss:.3f} ll")
    
    results_df = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "-" * 70)
    print("CROSS-VALIDATION SUMMARY")
    print("-" * 70)
    
    summary = {
        "Metric": ["Accuracy (mean)", "Accuracy (std)", "Log-Loss (mean)",
                    "Log-Loss (std)", "AUC (mean)", "AUC (std)"],
        "Logistic Regression": [
            f"{results_df['LR_Accuracy'].mean():.3f}",
            f"{results_df['LR_Accuracy'].std():.3f}",
            f"{results_df['LR_LogLoss'].mean():.3f}",
            f"{results_df['LR_LogLoss'].std():.3f}",
            f"{results_df['LR_AUC'].mean():.3f}",
            f"{results_df['LR_AUC'].std():.3f}",
        ],
        "XGBoost": [
            f"{results_df['XGB_Accuracy'].mean():.3f}",
            f"{results_df['XGB_Accuracy'].std():.3f}",
            f"{results_df['XGB_LogLoss'].mean():.3f}",
            f"{results_df['XGB_LogLoss'].std():.3f}",
            f"{results_df['XGB_AUC'].mean():.3f}",
            f"{results_df['XGB_AUC'].std():.3f}",
        ],
    }
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    
    return results_df


# =============================================================================
# Final Model Training
# =============================================================================

def train_final_models(df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Train both models on the FULL training dataset.
    Returns dict with 'lr', 'xgb', 'scaler', 'feature_cols', and 'best_model'.
    """
    print("\n" + "=" * 70)
    print("TRAINING FINAL MODELS ON ALL DATA")
    print("=" * 70)
    
    X, y, weights = prepare_features(df, feature_cols)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Logistic Regression
    lr_model = build_logistic_regression()
    lr_model.fit(X_scaled, y, sample_weight=weights)
    lr_train_acc = accuracy_score(y, lr_model.predict(X_scaled))
    print(f"  Logistic Regression train accuracy: {lr_train_acc:.3f}")
    
    # XGBoost
    xgb_model = build_xgboost()
    xgb_model.fit(X, y, sample_weight=weights)
    xgb_train_acc = accuracy_score(y, xgb_model.predict(X))
    print(f"  XGBoost train accuracy: {xgb_train_acc:.3f}")
    
    # Feature importance (XGBoost)
    importances = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": xgb_model.feature_importances_,
        "LR_Coefficient": lr_model.coef_[0],
    }).sort_values("Importance", ascending=False)
    
    print(f"\n  Top 10 Features (XGBoost importance):")
    for _, row in importances.head(10).iterrows():
        print(f"    {row['Feature']:<30} {row['Importance']:.4f}")
    
    models = {
        "lr": lr_model,
        "xgb": xgb_model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "feature_importance": importances,
    }
    
    # Save models
    model_path = os.path.join(config.OUTPUT_DIR, "trained_models.joblib")
    joblib.dump(models, model_path)
    print(f"\n  Models saved to {model_path}")
    
    return models


def load_trained_models() -> dict:
    """Load previously trained models from disk."""
    model_path = os.path.join(config.OUTPUT_DIR, "trained_models.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained models found at {model_path}")
    return joblib.load(model_path)
