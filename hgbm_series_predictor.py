#!/usr/bin/env python3
"""
hgbm_series_predictor.py

1-step-ahead HistGradientBoosting forecaster + Band Trading Strategy.

Differences vs previous attempts:
- Uses HistGradientBoostingRegressor (faster & often stronger than classic GBM).
- Single-output 1-step-ahead model -> recursive 768-step forecast.
- Rich feature engineering (lags, rolling stats, cyclical time).
- Aggressive band-trading (buy low-quantile, sell high-quantile) with many trades.

No KNN, no MLP, no multi-output monster that eats RAM.
"""

from __future__ import annotations

import argparse
import os
import math
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingRegressor
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
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build features for 1-step-ahead regression:

    X(t) = features at time t (constructed from history up to t-1)
    y(t) = Price(t)

    We drop rows where lags/rolling are not available.
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
    y = df["Price"]

    print(f"[Features] X: {X.shape}, y: {y.shape}, #features: {len(feature_cols)}")
    return X, y


# ---------------------------------------------------------------------
# Band Trading Strategy (aggressive)
# ---------------------------------------------------------------------

def generate_battery_actions(
    forecast_prices: np.ndarray,
    intervals_per_day: int = 96,
    capacity_mwh: float = 10.0,
    daily_reset_mwh: float = 5.0,
    min_transaction_mwh: float = 0.1,
    n_days: int = 8,
    buy_quantile: float = 0.30,
    sell_quantile: float = 0.70,
    step_mwh: float = 0.5,
    flush_last_intervals: int = 3,
) -> np.ndarray:
    """
    Band trading strategy:

    - În fiecare zi:
      * Când prețul e în quantila de jos (buy_quantile) -> încărcăm cu step_mwh.
      * Când prețul e în quantila de sus (sell_quantile) -> descărcăm step_mwh.
    - Putem avea multe tranzacții pe zi, atâta timp cât SoC rămâne în [0, capacity].
    - În ultimele `flush_last_intervals` intervale din zi -> golim complet bateria.

    Folosește doar ordonarea relativă a prețurilor din ziua respectivă.
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

        # Quantile thresholds pentru ziua curentă
        q_low, q_high = np.quantile(day_prices, [buy_quantile, sell_quantile])

        soc = float(daily_reset_mwh)

        for k in range(intervals_per_day):
            idx = start + k
            price = day_prices[k]
            intervals_left = intervals_per_day - k
            proposed_action = 0.0

            # End-of-day flush
            if intervals_left <= flush_last_intervals:
                if soc > 0.0:
                    proposed_action = -soc
            else:
                # Band BUY
                if (price <= q_low) and (soc < capacity_mwh - 1e-9):
                    proposed_action = min(step_mwh, capacity_mwh - soc)
                # Band SELL
                elif (price >= q_high) and (soc > 0.0 + 1e-9):
                    proposed_action = -min(step_mwh, soc)

            # Clamp la limite SoC
            max_charge = capacity_mwh - soc
            max_discharge = -soc
            action = float(np.clip(proposed_action, max_discharge, max_charge))

            # Quantizare la min_transaction_mwh
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
# Args
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="energy-trading-hackathon-2025")
    parser.add_argument("--filename", type=str, default="Dataset_clean.csv")
    parser.add_argument("--out_csv", type=str, default="energy-trading-hackathon-2025/submission_hgbm_recursive.csv")
    parser.add_argument("--sample_submission", type=str, default="energy-trading-hackathon-2025/sample_submission.csv")

    # Data / model
    parser.add_argument("--train_fraction", type=float, default=0.6,
                        help="Fraction of samples used for training (time-ordered).")

    # HistGradientBoosting params (single-output)
    parser.add_argument("--hgbm_max_iter", type=int, default=400,
                        help="Number of boosting iterations.")
    parser.add_argument("--hgbm_learning_rate", type=float, default=0.05,
                        help="Learning rate.")
    parser.add_argument("--hgbm_max_depth", type=int, default=7,
                        help="Tree depth (None = no limit).")
    parser.add_argument("--hgbm_max_leaf_nodes", type=int, default=63,
                        help="Max leaf nodes per tree.")

    # Forecasting horizon
    parser.add_argument("--n_days_forecast", type=int, default=8,
                        help="Days to forecast (8 days = 768 steps).")

    # Forecast smoothing
    parser.add_argument("--smooth_window", type=int, default=3,
                        help="Centered rolling mean window for smoothing forecast. <=1 disables smoothing.")

    # Band trading hyperparams
    parser.add_argument("--buy_quantile", type=float, default=0.30,
                        help="Lower quantile for buy band (0..0.5).")
    parser.add_argument("--sell_quantile", type=float, default=0.70,
                        help="Upper quantile for sell band (0.5..1).")
    parser.add_argument("--step_mwh", type=float, default=0.5,
                        help="MWh per buy/sell action in bands.")
    parser.add_argument("--flush_last_intervals", type=int, default=3,
                        help="Last N intervals each day used to fully discharge.")
    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()

    # 1. Load data
    path = os.path.join(args.data_path, args.filename)
    raw = read_csv(path)
    ts_df = to_ts(raw)
    price_series = ts_df["Price"].astype(float)

    # Lags: 15min, 1h, 1d, 2d, 1w, ~1m
    lags = [1, 2, 3, 4, 96, 96 * 2, 96 * 7, 96 * 28]

    # 2. Build dataset (1-step-ahead regression)
    X, y = build_feature_frame(price_series, lags=lags)

    N = len(X)
    train_size = int(N * args.train_fraction)
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_val = X.iloc[train_size:]
    y_val = y.iloc[train_size:]

    print(f"[Data] Train: {len(X_train)}, Val: {len(X_val)}")

    # 3. Build HistGradientBoosting model (wrapped in pipeline with scaling)
    hgbm = HistGradientBoostingRegressor(
        max_iter=args.hgbm_max_iter,
        learning_rate=args.hgbm_learning_rate,
        max_depth=args.hgbm_max_depth,
        max_leaf_nodes=args.hgbm_max_leaf_nodes,
        random_state=42,
        validation_fraction=0.1,
        early_stopping=True,
        n_iter_no_change=20,
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("hgbm", hgbm),
    ])

    print("[Model] Using HistGradientBoostingRegressor (1-step-ahead, single-output).")

    # 4. Fit model
    print("[Fit] Training HGBM model...")
    model.fit(X_train, y_train)
    print("[Fit] Done.")

    # 5. Validation RMSE on hold-out
    if len(X_val) > 0:
        y_val_pred = model.predict(X_val)
        mse = np.mean((y_val.values - y_val_pred) ** 2)
        rmse = float(np.sqrt(mse))
        print(f"[Metrics] 1-step-ahead Val RMSE: {rmse:.4f}")
    else:
        print("[Metrics] No validation set (train_fraction too high).")

    # 6. Recursive forecast for n_days_forecast
    intervals_per_day = 96
    n_steps = args.n_days_forecast * intervals_per_day

    history_prices = price_series.values.tolist()
    last_timestamp = price_series.index[-1]

    future_preds: List[float] = []

    for step in range(1, n_steps + 1):
        ts_k = last_timestamp + pd.Timedelta(minutes=15 * step)

        feats = {}

        # Time features at ts_k
        hour = ts_k.hour
        dow = ts_k.dayofweek
        feats["hour_sin"] = math.sin(2 * math.pi * hour / 24)
        feats["hour_cos"] = math.cos(2 * math.pi * hour / 24)
        feats["dow_sin"] = math.sin(2 * math.pi * dow / 7)
        feats["dow_cos"] = math.cos(2 * math.pi * dow / 7)

        # Lag features from history (includes previous forecasts)
        for lag in lags:
            if lag <= len(history_prices):
                feats[f"lag_{lag}"] = float(history_prices[-lag])
            else:
                feats[f"lag_{lag}"] = float(history_prices[0])

        # Rolling stats from latest window
        for w in [4, 96]:
            if w <= len(history_prices):
                win = history_prices[-w:]
            else:
                win = history_prices[:]
            win_arr = np.asarray(win, dtype=float)
            feats[f"roll_mean_{w}"] = float(np.mean(win_arr))
            feats[f"roll_std_{w}"] = float(np.std(win_arr))
            feats[f"roll_spread_{w}"] = float(np.max(win_arr) - np.min(win_arr))

        # Align with training feature columns
        feat_vector = pd.DataFrame([[feats.get(col, 0.0) for col in X_train.columns]],
                                   columns=X_train.columns)

        y_hat = float(model.predict(feat_vector)[0])
        y_hat = max(0.0, y_hat)  # keep non-negative prices

        future_preds.append(y_hat)
        history_prices.append(y_hat)

        if step % intervals_per_day == 0:
            day_idx = step // intervals_per_day
            print(f"[Forecast] Generated up to day {day_idx}/{args.n_days_forecast}")

    final_forecast = np.array(future_preds, dtype=float)

    # 7. Smooth over forecast if requested
    if args.smooth_window > 1:
        s = pd.Series(final_forecast)
        final_forecast = s.rolling(
            window=args.smooth_window,
            min_periods=1,
            center=True
        ).mean().values
        print(f"[Forecast] Applied smoothing with window={args.smooth_window}")

    # 8. Generate trading actions
    actions = generate_battery_actions(
        final_forecast,
        intervals_per_day=intervals_per_day,
        capacity_mwh=10.0,
        daily_reset_mwh=5.0,
        min_transaction_mwh=0.1,
        n_days=args.n_days_forecast,
        buy_quantile=args.buy_quantile,
        sell_quantile=args.sell_quantile,
        step_mwh=args.step_mwh,
        flush_last_intervals=args.flush_last_intervals,
    )

    validate_battery_actions(actions, return_trace=True)

    # 9. Save submission
    sub = pd.read_csv(args.sample_submission)
    sub["Position"] = np.round(actions, 3)
    sub.to_csv(args.out_csv, index=False)
    print(f"[Main] Done. Saved to {args.out_csv}")


if __name__ == "__main__":
    main()
