#!/usr/bin/env python3
"""
knn_series_predictor_extra.py (previously knn_series_predictor.py)

Advanced forecaster + Profit-Aware Band Trading.

Models supported:
  - RandomForest (rf)
  - GradientBoosting (gbm, via MultiOutputRegressor)
  - RF+GBM ensemble (ensemble)
  - K-Nearest Neighbors (knn)
  - ExtraTreesRegressor (extra)
  - Ridge Regression (ridge)  <-- NEW, linear, non-tree, non-KNN

Forecasting:
  - Direct multi-output horizon (96 steps per block).
  - Block-recursive: 8 blocks * 96 = 768 steps (8 days).

Trading:
  - Band-based: buy in low-quantile band, sell in high-quantile band.
  - Daily SoC reset + end-of-day flush.
"""

from __future__ import annotations

import argparse
import os
import math
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from handling_data import read_csv, to_ts
from helper_function import validate_battery_actions


# ---------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------

def add_cyclical_features(df: pd.DataFrame, col_name: str, period: int) -> pd.DataFrame:
    """Adds sin/cos transformation for a cyclical feature (e.g., hour, day-of-week)."""
    df[f"{col_name}_sin"] = np.sin(2 * np.pi * df[col_name] / period)
    df[f"{col_name}_cos"] = np.cos(2 * np.pi * df[col_name] / period)
    return df


def build_feature_frame(
    ts: pd.Series,
    lags: List[int],
    horizon: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build advanced features including cyclical time and volatility stats.

    If horizon > 1, prepare multi-output targets: y has columns
    [Price(t+1), ..., Price(t+horizon)].
    """
    df = pd.DataFrame({"Price": ts.astype(float).values}, index=ts.index)

    # 1. Time features
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek
    df = add_cyclical_features(df, "hour", 24)
    df = add_cyclical_features(df, "dow", 7)

    # 2. Lag features
    for lag in lags:
        df[f"lag_{lag}"] = df["Price"].shift(lag)

    # 3. Rolling stats (short-term and daily)
    for w in [4, 96]:
        roll = df["Price"].rolling(w)
        df[f"roll_mean_{w}"] = roll.mean()
        df[f"roll_std_{w}"] = roll.std()
        df[f"roll_spread_{w}"] = roll.max() - roll.min()

    # Drop NaNs from lags/rolling
    df = df.dropna()

    # Features: everything except raw Price and raw hour/dow
    feature_cols = [
        c for c in df.columns
        if c not in ("Price", "hour", "dow")
    ]
    X = df[feature_cols]

    # Targets
    if horizon > 1:
        y_list = []
        for h in range(1, horizon + 1):
            y_list.append(df["Price"].shift(-h))
        y = pd.concat(y_list, axis=1)
        # Remove rows where future targets are missing
        valid_idx = y.dropna().index
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        y.columns = [f"t+{k}" for k in range(1, horizon + 1)]
    else:
        y = df[["Price"]]

    print(f"[Features] X: {X.shape}, y: {y.shape}, #features: {len(feature_cols)}")
    return X, y


# ---------------------------------------------------------------------
# Profit-Aware Band Trading
# ---------------------------------------------------------------------

def generate_battery_actions(
    forecast_prices: np.ndarray,
    intervals_per_day: int = 96,
    capacity_mwh: float = 10.0,
    daily_reset_mwh: float = 5.0,
    min_transaction_mwh: float = 0.1,
    n_days: int = 8,
    # Band strategy params (exposed via CLI)
    buy_quantile: float = 0.25,
    sell_quantile: float = 0.75,
    step_mwh: float = 1.0,
    flush_last_intervals: int = 2,
) -> np.ndarray:
    """
    Hyper-aggressive band trading strategy.

    - No explicit cap on trades/day.
    - Every time the price is in the lowest `buy_quantile` of the day,
      we BUY `step_mwh` (if there is free capacity).
    - Every time the price is in the highest `sell_quantile` of the day,
      we SELL `step_mwh` (if SoC > step_mwh).
    - We can buy/sell multiple times per day as long as SoC stays in [0, capacity].
    - In the last `flush_last_intervals` slots of the day, we discharge everything.

    This uses only forecast RANK, not absolute level.
    """
    prices = np.asarray(forecast_prices, dtype=float).flatten()
    actions = np.zeros_like(prices, dtype=float)

    total_intervals = len(prices)
    n_days_eff = min(n_days, total_intervals // intervals_per_day)

    for day in range(n_days_eff):
        start = day * intervals_per_day
        end = start + intervals_per_day
        day_prices = prices[start:end]
        if day_prices.size == 0:
            continue

        # Quantile thresholds for this day
        q_low, q_high = np.quantile(day_prices, [buy_quantile, sell_quantile])

        soc = float(daily_reset_mwh)

        for k in range(intervals_per_day):
            idx = start + k
            price = day_prices[k]
            intervals_left = intervals_per_day - k
            proposed_action = 0.0

            # End-of-day flush: always empty the battery
            if intervals_left <= flush_last_intervals:
                if soc > 0.0:
                    proposed_action = -soc
            else:
                # Buy band: low prices -> buy step_mwh
                if (price <= q_low) and (soc < capacity_mwh - 1e-9):
                    proposed_action = min(step_mwh, capacity_mwh - soc)

                # Sell band: high prices -> sell step_mwh
                elif (price >= q_high) and (soc > 0.0 + 1e-9):
                    proposed_action = -min(step_mwh, soc)

            # Clamp to SoC limits
            max_charge = capacity_mwh - soc
            max_discharge = -soc
            action = float(np.clip(proposed_action, max_discharge, max_charge))

            # Quantize to min_transaction_mwh
            if abs(action) < min_transaction_mwh:
                action = 0.0
            else:
                steps = round(action / min_transaction_mwh)
                action = steps * min_transaction_mwh
                if action > 0:
                    action = min(action, capacity_mwh - soc)
                elif action < 0:
                    action = max(action, -soc)

            actions[idx] = action
            soc = float(np.clip(soc + action, 0.0, capacity_mwh))

    n_trades = int(np.count_nonzero(actions))
    print(f"[Strategy] Trades: {n_trades}/{len(actions)}  (~{n_trades / max(n_days_eff,1):.1f} per day)")
    return actions


# ---------------------------------------------------------------------
# Multi-output models + ensemble
# ---------------------------------------------------------------------

class MultiOutputEnsemble:
    """Averaging ensemble for multi-output regressors."""

    def __init__(self, models: List):
        self.models = models

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        preds = [m.predict(X) for m in self.models]
        return np.mean(preds, axis=0)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="energy-trading-hackathon-2025")
    parser.add_argument("--filename", type=str, default="Dataset_clean.csv")
    parser.add_argument("--out_csv", type=str, default="energy-trading-hackathon-2025/submission_extra_v1.csv")
    parser.add_argument("--sample_submission", type=str, default="energy-trading-hackathon-2025/sample_submission.csv")

    # Forecasting / model
    parser.add_argument("--train_fraction", type=float, default=0.6,
                        help="Fraction of samples used for training (time-ordered).")
    parser.add_argument("--horizon", type=int, default=96,
                        help="Direct forecasting horizon in 15-min steps (96 = 1 day).")
    parser.add_argument(
        "--model_type",
        type=str,
        default="ridge",
        choices=["rf", "gbm", "ensemble", "knn", "extra", "ridge"],
        help="Multi-output model type.",
    )

    # RF params
    parser.add_argument("--rf_n_estimators", type=int, default=300)
    parser.add_argument("--rf_max_depth", type=int, default=12)

    # GBM params
    parser.add_argument("--gbm_n_estimators", type=int, default=300)
    parser.add_argument("--gbm_max_depth", type=int, default=3)
    parser.add_argument("--gbm_learning_rate", type=float, default=0.05)

    # KNN params
    parser.add_argument("--knn_neighbors", type=int, default=80,
                        help="Number of neighbors for KNN.")
    parser.add_argument("--knn_weights", type=str, default="distance",
                        choices=["uniform", "distance"],
                        help="KNN weighting scheme.")
    parser.add_argument("--knn_metric", type=str, default="euclidean",
                        help="KNN distance metric.")

    # ExtraTrees params
    parser.add_argument("--extra_n_estimators", type=int, default=400,
                        help="Number of trees for ExtraTrees.")
    parser.add_argument("--extra_max_depth", type=int, default=20,
                        help="Max depth for ExtraTrees.")
    parser.add_argument("--extra_max_features", type=float, default=0.7,
                        help="Max features fraction for ExtraTrees (0..1).")

    # Ridge params (NEW)
    parser.add_argument("--ridge_alpha", type=float, default=1.0,
                        help="L2 regularization strength for Ridge regression.")

    # Forecast smoothing
    parser.add_argument("--smooth_window", type=int, default=5,
                        help="Centered rolling mean window for smoothing 8-day forecast. <=1 disables smoothing.")

    # Band trading hyperparams
    parser.add_argument("--buy_quantile", type=float, default=0.25,
                        help="Lower quantile for buy band (0..0.5).")
    parser.add_argument("--sell_quantile", type=float, default=0.75,
                        help="Upper quantile for sell band (0.5..1).")
    parser.add_argument("--step_mwh", type=float, default=1.0,
                        help="MWh per buy/sell action in bands.")
    parser.add_argument("--flush_last_intervals", type=int, default=2,
                        help="Last N intervals each day used to fully discharge.")
    return parser.parse_args()


def main():
    args = parse_args()

    # -----------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------
    path = os.path.join(args.data_path, args.filename)
    raw = read_csv(path)
    ts_df = to_ts(raw)
    price_series = ts_df["Price"].astype(float)

    HORIZON = int(args.horizon)
    # 15min, 1h, 1d, 2d, 1w, ~1m
    lags = [1, 2, 3, 4, 96, 96 * 2, 96 * 7, 96 * 28]

    # -----------------------------------------------------------------
    # 2. Build dataset (direct multi-output)
    # -----------------------------------------------------------------
    X, y = build_feature_frame(price_series, lags=lags, horizon=HORIZON)

    N = len(X)
    train_size = int(N * args.train_fraction)
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_val = X.iloc[train_size:]
    y_val = y.iloc[train_size:]

    print(f"[Data] Train: {len(X_train)}, Val: {len(X_val)}, Horizon: {HORIZON}")

    # -----------------------------------------------------------------
    # 3. Build model(s)
    # -----------------------------------------------------------------
    models: List = []

    if args.model_type in ("rf", "ensemble"):
        rf = RandomForestRegressor(
            n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth,
            n_jobs=-1,
            random_state=42,
        )
        models.append(rf)

    if args.model_type in ("gbm", "ensemble"):
        base_gbm = GradientBoostingRegressor(
            n_estimators=args.gbm_n_estimators,
            max_depth=args.gbm_max_depth,
            learning_rate=args.gbm_learning_rate,
            random_state=42,
        )
        gbm = MultiOutputRegressor(base_gbm)
        models.append(gbm)

    if args.model_type == "knn":
        knn = KNeighborsRegressor(
            n_neighbors=args.knn_neighbors,
            weights=args.knn_weights,
            metric=args.knn_metric,
            n_jobs=-1,
        )
        models.append(knn)

    if args.model_type == "extra":
        extra = ExtraTreesRegressor(
            n_estimators=args.extra_n_estimators,
            max_depth=args.extra_max_depth,
            max_features=args.extra_max_features,
            n_jobs=-1,
            random_state=42,
        )
        models.append(extra)

    if args.model_type == "ridge":
        # Linear multi-output model with feature scaling
        ridge = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=args.ridge_alpha, fit_intercept=True)),
        ])
        models.append(ridge)

    # Pick main model
    if args.model_type == "rf":
        main_model = models[0]
        print("[Model] Using RandomForestRegressor (multi-output).")
    elif args.model_type == "gbm":
        main_model = models[0]
        print("[Model] Using GradientBoosting (MultiOutputRegressor).")
    elif args.model_type == "knn":
        main_model = models[0]
        print("[Model] Using KNeighborsRegressor (multi-output).")
    elif args.model_type == "extra":
        main_model = models[0]
        print("[Model] Using ExtraTreesRegressor (multi-output).")
    elif args.model_type == "ridge":
        main_model = models[0]
        print("[Model] Using Ridge regression (multi-output, scaled features).")
    else:
        main_model = MultiOutputEnsemble(models)
        print("[Model] Using RF + GBM ensemble (multi-output).")

    # -----------------------------------------------------------------
    # 4. Fit models
    # -----------------------------------------------------------------
    if isinstance(main_model, MultiOutputEnsemble):
        print("[Fit] Training RF and GBM for ensemble...")
        for m in models:
            m.fit(X_train, y_train)
    else:
        print("[Fit] Training main model...")
        main_model.fit(X_train, y_train)

    # -----------------------------------------------------------------
    # 5. Basic validation on first-step horizon
    # -----------------------------------------------------------------
    if len(X_val) > 0:
        y_val_pred = main_model.predict(X_val)  # DataFrame is fine for Pipeline / trees
        y_val_np = y_val.values
        mse_1 = np.mean((y_val_np[:, 0] - y_val_pred[:, 0]) ** 2)
        rmse_1 = float(np.sqrt(mse_1))
        print(f"[Metrics] 1-step-ahead Val RMSE: {rmse_1:.4f}")
    else:
        print("[Metrics] No validation set (train_fraction too high).")

    # -----------------------------------------------------------------
    # 6. Forecast 8 days (block-recursive, 96-step blocks)
    # -----------------------------------------------------------------
    future_preds: List[float] = []

    history_prices = price_series.values.tolist()
    last_timestamp = price_series.index[-1]

    for block in range(8):
        curr_time = last_timestamp

        feats = {}

        # Time features (same as in training)
        hour = curr_time.hour
        dow = curr_time.dayofweek
        feats["hour_sin"] = math.sin(2 * math.pi * hour / 24)
        feats["hour_cos"] = math.cos(2 * math.pi * hour / 24)
        feats["dow_sin"] = math.sin(2 * math.pi * dow / 7)
        feats["dow_cos"] = math.cos(2 * math.pi * dow / 7)

        # Lag features
        for lag in lags:
            feats[f"lag_{lag}"] = float(history_prices[-lag])

        # Rolling stats
        for w in [4, 96]:
            win = history_prices[-w:]
            win_arr = np.asarray(win, dtype=float)
            feats[f"roll_mean_{w}"] = float(np.mean(win_arr))
            feats[f"roll_std_{w}"] = float(np.std(win_arr))
            feats[f"roll_spread_{w}"] = float(np.max(win_arr) - np.min(win_arr))

        # Align with training feature columns
        feat_vector = pd.DataFrame([[feats.get(col, 0.0) for col in X_train.columns]],
                                   columns=X_train.columns)

        block_pred = main_model.predict(feat_vector)[0]  # shape (HORIZON,)
        future_preds.extend(block_pred.tolist())

        history_prices.extend(block_pred.tolist())
        last_timestamp = last_timestamp + pd.Timedelta(minutes=15 * HORIZON)

        print(f"[Forecast] Generated block {block + 1}/8")

    final_forecast = np.array(future_preds, dtype=float)

    # Smooth over all 8 days if requested
    if args.smooth_window > 1:
        s = pd.Series(final_forecast)
        final_forecast = s.rolling(
            window=args.smooth_window,
            min_periods=1,
            center=True
        ).mean().values
        print(f"[Forecast] Applied smoothing with window={args.smooth_window}")

    # -----------------------------------------------------------------
    # 7. Generate trading actions
    # -----------------------------------------------------------------
    actions = generate_battery_actions(
        final_forecast,
        intervals_per_day=96,
        capacity_mwh=10.0,
        daily_reset_mwh=5.0,
        min_transaction_mwh=0.1,
        n_days=8,
        buy_quantile=args.buy_quantile,
        sell_quantile=args.sell_quantile,
        step_mwh=args.step_mwh,
        flush_last_intervals=args.flush_last_intervals,
    )

    validate_battery_actions(actions, return_trace=True)

    # -----------------------------------------------------------------
    # 8. Save submission
    # -----------------------------------------------------------------
    sub = pd.read_csv(args.sample_submission)
    sub["Position"] = np.round(actions, 3)
    sub.to_csv(args.out_csv, index=False)
    print(f"[Main] Done. Saved to {args.out_csv}")


if __name__ == "__main__":
    main()
