#!/usr/bin/env python3
"""
gbm_series_predictor_v4.py

Advanced Tree-based price forecaster + Profit-Aware Trading.

Improvements vs v3:
1. Correct time alignment for block direct multi-output forecasting.
2. Multi-output ensemble (RandomForest + GradientBoosting) for better predictions.
3. Consistent feature construction at forecast time (same as training).
4. Optional smoothing of the full 8-day forecast before trading.

The trading logic is still profit-aware: trade only if the day has enough spread.
"""

from __future__ import annotations

import argparse
import os
import math
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

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
# Profit-Aware Trading
# ---------------------------------------------------------------------

def generate_battery_actions(
    forecast_prices: np.ndarray,
    intervals_per_day: int = 96,
    capacity_mwh: float = 10.0,
    daily_reset_mwh: float = 5.0,
    min_transaction_mwh: float = 0.1,
    n_days: int = 8,
    target_trades_per_day: int = 4,
    min_profit_spread: float = 10.0,
) -> np.ndarray:
    """
    Profit-aware trading strategy.

    For each day:
      1. Select ~target_trades_per_day/2 lowest-price slots as buy candidates,
         and ~target_trades_per_day/2 highest-price slots as sell candidates.
      2. Compute avg buy price vs avg sell price. If spread is too small
         (< min_profit_spread), shrink to just the single min and max slot.
      3. Execute full-charge / full-discharge trades at candidate slots,
         respecting capacity and SoC, with quantization to min_transaction_mwh.
      4. In the last 2 intervals of the day, flush remaining SoC.

    This uses only ranking & spread, not absolute price levels.
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

        sorted_indices = np.argsort(day_prices)  # 0 = lowest price
        n_actions = max(1, target_trades_per_day // 2)
        n_actions = min(n_actions, len(sorted_indices) // 2)

        buy_candidate_indices = sorted_indices[:n_actions]
        sell_candidate_indices = sorted_indices[-n_actions:]

        avg_buy = float(np.mean(day_prices[buy_candidate_indices]))
        avg_sell = float(np.mean(day_prices[sell_candidate_indices]))

        # If day is almost flat, only trade absolute extremes
        if (avg_sell - avg_buy) < min_profit_spread:
            buy_candidate_indices = [int(sorted_indices[0])]
            sell_candidate_indices = [int(sorted_indices[-1])]

        soc = float(daily_reset_mwh)

        for k in range(intervals_per_day):
            idx = start + k
            local_idx = k
            intervals_left = intervals_per_day - k

            proposed_action = 0.0

            # 1. End-of-day flush
            if intervals_left <= 2:
                if soc > 0.0:
                    proposed_action = -soc
            else:
                # 2. Main buy/sell logic
                if local_idx in buy_candidate_indices and soc < capacity_mwh:
                    proposed_action = capacity_mwh - soc  # charge to full
                elif local_idx in sell_candidate_indices and soc > 0.0:
                    proposed_action = -soc  # discharge to zero

            # 3. Clamp to physical constraints
            max_charge = capacity_mwh - soc
            max_discharge = -soc
            action = float(np.clip(proposed_action, max_discharge, max_charge))

            # 4. Quantization
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = [m.predict(X) for m in self.models]
        return np.mean(preds, axis=0)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="energy-trading-hackathon-2025")
    parser.add_argument("--filename", type=str, default="Dataset_clean.csv")
    parser.add_argument("--out_csv", type=str, default="energy-trading-hackathon-2025/submission_gbm_v4.csv")
    parser.add_argument("--sample_submission", type=str, default="energy-trading-hackathon-2025/sample_submission.csv")

    # Forecasting / model
    parser.add_argument("--train_fraction", type=float, default=0.6,
                        help="Fraction of samples used for training (time-ordered).")
    parser.add_argument("--horizon", type=int, default=96,
                        help="Direct forecasting horizon in 15-min steps (96 = 1 day).")
    parser.add_argument("--model_type", type=str, default="ensemble",
                        choices=["rf", "gbm", "ensemble"],
                        help="Multi-output model type.")

    # RF params
    parser.add_argument("--rf_n_estimators", type=int, default=300)
    parser.add_argument("--rf_max_depth", type=int, default=12)

    # GBM params
    parser.add_argument("--gbm_n_estimators", type=int, default=300)
    parser.add_argument("--gbm_max_depth", type=int, default=3)
    parser.add_argument("--gbm_learning_rate", type=float, default=0.05)

    # Forecast smoothing
    parser.add_argument("--smooth_window", type=int, default=5,
                        help="Centered rolling mean window for smoothing 8-day forecast. <=1 disables smoothing.")

    # Strategy
    parser.add_argument("--target_trades", type=int, default=12,
                        help="Target non-zero actions per day.")
    parser.add_argument("--min_spread", type=float, default=5.0,
                        help="Minimum EUR/MWh price spread between buys and sells.")
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
    lags = [1, 2, 3, 4, 96, 96 * 2, 96 * 7]  # 15min, 1h, 1d, 2d, 1w style

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
    models = []

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

    if args.model_type == "rf":
        main_model = models[0]
        print("[Model] Using RandomForestRegressor (multi-output).")
    elif args.model_type == "gbm":
        main_model = models[0]
        print("[Model] Using GradientBoosting (MultiOutputRegressor).")
    else:
        main_model = MultiOutputEnsemble(models)
        print("[Model] Using RF + GBM ensemble (multi-output).")

    # -----------------------------------------------------------------
    # 4. Fit models
    # -----------------------------------------------------------------
    if isinstance(main_model, MultiOutputEnsemble):
        # Fit both underlying models
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
        y_val_pred = main_model.predict(X_val.values)
        # y_val: DataFrame with columns t+1..t+HORIZON
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
        # Build feature vector at current last_timestamp (time t0)
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

        # Align with training feature columns (order + missing protection)
        feat_vector = pd.DataFrame([[feats.get(col, 0.0) for col in X_train.columns]],
                                   columns=X_train.columns)

        block_pred = main_model.predict(feat_vector.values)[0]  # shape (HORIZON,)
        future_preds.extend(block_pred.tolist())

        # Update history & timestamp for next block
        history_prices.extend(block_pred.tolist())
        last_timestamp = last_timestamp + pd.Timedelta(minutes=15 * HORIZON)

        print(f"[Forecast] Generated block {block + 1}/8 (day {block + 1})")

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
        target_trades_per_day=args.target_trades,
        min_profit_spread=args.min_spread,
    )

    # Validate with helper (uses its internal defaults)
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
