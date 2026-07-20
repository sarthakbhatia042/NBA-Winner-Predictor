# 🏀 NBA Playoff Predictor

Predict the NBA champion using machine learning. This project pulls real NBA data from the `nba_api`, engineers advanced features from 10 seasons of regular-season stats, and trains **Logistic Regression** + **XGBoost** models to predict game-by-game playoff outcomes and simulate the full bracket.

## Features

- **Data Pipeline**: Automated fetching + caching from `nba_api` (10 seasons)
- **Advanced Features**: Net Rating, Strength of Schedule, pace-adjusted stats, seed differential, H2H records, recent form, playoff experience
- **Dual Models**: Logistic Regression (baseline) + XGBoost (boosted)
- **Season Weighting**: Exponential decay — recent seasons matter more
- **Proper Validation**: Leave-One-Season-Out cross-validation (no data leakage)
- **Bracket Simulation**: Game-by-game predictions → best-of-7 series → full bracket
- **Visualization**: Dark-themed bracket PNG with team names, series scores, and win probabilities

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python main.py

# 4. Use XGBoost (default) or Logistic Regression
python main.py --model xgb
python main.py --model lr

# 5. Force re-download data
python main.py --force-refresh
```

## Project Structure

```
Winner predictor/
├── config.py               # Seasons, features, hyperparameters, paths
├── data/
│   ├── fetch_data.py       # nba_api data fetching + CSV caching
│   └── cache/              # Cached API responses (auto-generated)
├── features/
│   └── engineer.py         # Feature engineering pipeline
├── models/
│   ├── train.py            # Model training + cross-validation
│   └── predict.py          # Game prediction + bracket simulation
├── visualization/
│   └── bracket.py          # Bracket rendering (matplotlib)
├── output/                 # Generated outputs (auto-created)
│   ├── bracket_2025_26.png # Bracket visualization
│   ├── cv_results.csv      # Cross-validation results
│   └── trained_models.joblib
├── main.py                 # Entry point
├── requirements.txt
└── README.md
```

## How It Works

1. **Fetch**: Pull regular-season game logs, advanced stats, and standings for 10 seasons via `nba_api`
2. **Engineer**: For each historical playoff game, compute features using *only* regular-season data (no leakage)
3. **Train**: Fit Logistic Regression + XGBoost with exponential decay sample weighting
4. **Validate**: Leave-One-Season-Out CV — train on 9 seasons, test on 1, rotate
5. **Predict**: Use current-season regular-season data to predict each playoff game
6. **Simulate**: Best-of-7 series with 2-2-1-1-1 home-court format → advance winners through bracket
7. **Visualize**: Render a publication-quality bracket with results

## Training Seasons

| Season | Weight |
|--------|--------|
| 2024-25 | 1.000 |
| 2023-24 | 0.850 |
| 2022-23 | 0.723 |
| 2021-22 | 0.614 |
| 2020-21 | 0.522 |
| 2019-20 | 0.444 (Bubble — no home-court) |
| 2018-19 | 0.377 |
| 2017-18 | 0.321 |
| 2016-17 | 0.272 |
| 2015-16 | 0.232 |

## Dependencies

- `nba_api` — NBA stats API wrapper
- `pandas` / `numpy` — Data processing
- `scikit-learn` — Logistic Regression + preprocessing
- `xgboost` — Gradient boosted trees
- `matplotlib` — Bracket visualization
- `joblib` — Model persistence
