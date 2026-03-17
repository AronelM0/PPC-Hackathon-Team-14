#!/usr/bin/env python3
"""
hmm_series_predictor.py

Gaussian HMM-based price forecasting + battery trading policy.

Improvements:
1. Uses Log-Returns (ln(p_t/p_{t-1})) for stationarity.
2. Adds damping to forecast loop to prevent exponential explosion.
3. Implements Z-score based trading strategy.
4. Implements aggressive End-of-Day discharge.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from pydantic import BaseModel, Field

from handling_data import read_csv, to_ts, normalize_data, adf_test  # type: ignore
from helper_function import validate_battery_actions  # official validator

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

class HMMConfig(BaseModel):
    data_path: str = Field("energy-trading-hackathon-2025", description="Root path for the dataset folder.")
    filename: str = Field("Dataset.csv", description="Main CSV filename inside data_path.")
    sample_submission: str = Field(
        "energy-trading-hackathon-2025/sample_submission.csv",
        description="Template submission CSV from the repo.",
    )
    out_csv: str = Field(
        "energy-trading-hackathon-2025/submission_hmm.csv",
        description="Output submission CSV filename.",
    )

    # HMM / split
    val_ratio: float = Field(0.2, ge=0.01, le=0.9, description="Fraction of deltas used for validation.")
    n_states: int = Field(6, ge=2, description="Default number of states if no grid search is given.")
    state_grid: Optional[str] = Field(
        "",
        description="Optional comma-separated list of n_states, e.g. '4,6,8'. ",
    )
    n_iter: int = Field(160, ge=10, description="Maximum EM iterations for HMM.")
    covariance_type: str = Field(
        "diag",
        description="Covariance type for GaussianHMM (full|diag|tied|spherical).",
    )
    tol: float = Field(1e-4, gt=0.0, description="Convergence tolerance for EM; smaller = more precise.")

    # Battery
    capacity_mwh: float = Field(10.0, gt=0.0)
    daily_reset_mwh: float = Field(5.0, ge=0.0)
    min_transaction_mwh: float = Field(0.1, ge=0.0)
    intervals_per_day: int = Field(96, description="Number of 15-min intervals per day (24*4).")
    n_days: int = Field(8, description="Forecast horizon in days (8 days -> 768 intervals).")


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    denom = np.clip(np.abs(y_true), 1e-8, None)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE_%": mape, "R2": r2}


# -----------------------------------------------------------------------------
# Battery trading policy
# -----------------------------------------------------------------------------

def generate_battery_actions(
    forecast_prices: np.ndarray,
    intervals_per_day: int = 96,
    capacity_mwh: float = 10.0,
    daily_reset_mwh: float = 5.0,
    min_transaction_mwh: float = 0.1,
    n_days: int = 8,
    low_quantile: float = 0.20,
    high_quantile: float = 0.80,
    flush_last_intervals: int = 2,
) -> np.ndarray:
    """
    High-contrast ranking-based trading policy.

    Idea:
    - For each day, compute low/high quantiles of the *forecast* prices.
    - If price is in the bottom quantile -> charge to FULL (10 MWh).
    - If price is in the top quantile   -> discharge to ZERO (0 MWh).
    - Otherwise do nothing.
    - In the very last intervals of the day, flush remaining SoC to 0
      so we don't get stuck with energy that will be sold at the day's
      lowest price by the rules.

    This uses ranking of forecast prices, not absolute levels, so it
    works even if the model is biased as long as it roughly orders
    cheap vs expensive intervals correctly.
    """
    prices = np.asarray(forecast_prices, dtype=float).flatten()
    actions = np.zeros_like(prices, dtype=float)

    total_intervals = len(prices)
    # Trust the actual length more than the n_days argument
    n_days_eff = min(n_days, total_intervals // intervals_per_day)

    for day in range(n_days_eff):
        start = day * intervals_per_day
        end = start + intervals_per_day
        day_prices = prices[start:end]

        if day_prices.size == 0:
            continue

        # Daily quantiles: define cheap / expensive regions
        q_low, q_high = np.quantile(day_prices, [low_quantile, high_quantile])

        soc = float(daily_reset_mwh)

        for k in range(intervals_per_day):
            idx = start + k
            price = day_prices[k]
            intervals_left = intervals_per_day - k

            action = 0.0

            # --- End-of-day flush to 0 in the last few intervals ---
            if intervals_left <= flush_last_intervals:
                if soc > 0.0:
                    # Dump everything now; remaining intervals do nothing
                    proposed_action = -soc
                    # Clip (just to be safe)
                    max_charge = capacity_mwh - soc
                    max_discharge = -soc
                    action = float(np.clip(proposed_action, max_discharge, max_charge))
                else:
                    action = 0.0

            else:
                # --- Main logic: full charge / full discharge at extremes ---
                if price <= q_low and soc < capacity_mwh:
                    # Very cheap -> move SoC to FULL
                    proposed_action = capacity_mwh - soc  # positive
                elif price >= q_high and soc > 0.0:
                    # Very expensive -> move SoC to ZERO
                    proposed_action = -soc  # negative
                else:
                    proposed_action = 0.0

                if proposed_action != 0.0:
                    max_charge = capacity_mwh - soc
                    max_discharge = -soc
                    action = float(np.clip(proposed_action, max_discharge, max_charge))
                else:
                    action = 0.0

            # Quantize to min_transaction_mwh
            if abs(action) < min_transaction_mwh:
                action = 0.0
            else:
                steps = round(action / min_transaction_mwh)
                action = steps * min_transaction_mwh
                # Re-clip after quantization to be safe
                if action > 0:
                    action = min(action, capacity_mwh - soc)
                elif action < 0:
                    action = max(action, -soc)

            actions[idx] = action
            soc = float(np.clip(soc + action, 0.0, capacity_mwh))

    n_trades = int(np.count_nonzero(actions))
    print(f"[Strategy] Generated actions. Non-zero trades: {n_trades}/{len(actions)}")
    return actions


def check_battery_constraints(
    actions: np.ndarray,
    timestamps: pd.DatetimeIndex,
    capacity_mwh: float = 10.0,
    daily_reset_mwh: float = 5.0,
) -> None:
    soc = daily_reset_mwh
    last_day = timestamps[0].date()
    violations = 0

    for t, (a, ts) in enumerate(zip(actions, timestamps)):
        day = ts.date()
        if day != last_day:
            soc = daily_reset_mwh
            last_day = day

        soc += a
        if soc < -1e-6 or soc > capacity_mwh + 1e-6:
            print(f"[check_battery_constraints] violation at t={t}, time={ts}, SoC={soc:.4f}")
            violations += 1

    if violations == 0:
        print("[check_battery_constraints] All constraints satisfied (internal check).")
    else:
        print(f"[check_battery_constraints] Total violations (internal check): {violations}")


# -----------------------------------------------------------------------------
# HMM utilities
# -----------------------------------------------------------------------------

def build_deltas(series_scaled: np.ndarray) -> np.ndarray:
    """
    Use LOG RETURNS instead of simple difference.
    log_ret_t = ln(p_t / p_{t-1})
    """
    values = series_scaled.astype(np.float64).reshape(-1)
    # Add small epsilon to avoid log(0)
    values = np.clip(values, 1e-6, None) 
    
    log_rets = np.log(values[1:] / values[:-1])
    return log_rets.reshape(-1, 1)


def fit_hmm(
    train_deltas: np.ndarray,
    n_states: int,
    n_iter: int,
    covariance_type: str,
    random_state: int = 42,
    tol: float = 1e-4,
) -> GaussianHMM:
    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
        verbose=False,
        tol=tol,
    )
    model.fit(train_deltas)
    return model


def one_step_ahead_price_predictions(
    scaled_series: np.ndarray,
    deltas: np.ndarray,
    model: GaussianHMM,
) -> np.ndarray:
    """
    Reconstruct prices from predicted log-returns.
    """
    T = scaled_series.shape[0]
    N = deltas.shape[0]

    state_probs = model.predict_proba(deltas)
    A = model.transmat_
    means = model.means_.reshape(-1)

    prices = scaled_series.reshape(-1)
    pred_scaled = np.full(T, np.nan, dtype=np.float64)

    for t in range(N - 1):
        alpha_t = state_probs[t]
        alpha_next = alpha_t @ A
        expected_log_ret = alpha_next @ means
        
        # p_{t+1} = p_t * exp(log_ret)
        current_price = max(prices[t+1], 1e-6)
        pred_price = current_price * np.exp(expected_log_ret)
        
        pred_scaled[t + 2] = float(np.clip(pred_price, 0.0, 1.0))

    return pred_scaled


def forecast_hmm_states_and_prices(
    last_price_scaled: float,
    last_state_probs: np.ndarray,
    model: GaussianHMM,
    n_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forecast using log-returns with DAMPING.
    """
    A = model.transmat_
    means = model.means_.reshape(-1)

    alpha = last_state_probs.astype(np.float64).reshape(-1)
    price = max(float(last_price_scaled), 1e-6)

    prices, states = [], []
    
    # Damping factor to prevent explosion
    damping = 0.98 

    for i in range(n_steps):
        alpha = alpha @ A
        state_idx = int(np.argmax(alpha))
        
        # Apply damping to the log-return
        expected_log_ret = float(means[state_idx]) * (damping ** i)
        
        price = price * np.exp(expected_log_ret)
        price = float(np.clip(price, 0.0, 1.0))
        
        prices.append(price)
        states.append(state_idx)

    return np.array(prices, dtype=np.float32).reshape(-1, 1), np.array(states, dtype=int)


# -----------------------------------------------------------------------------
# Simple hyperparameter search over n_states
# -----------------------------------------------------------------------------

def parse_state_grid(grid_str: str | None) -> Optional[List[int]]:
    if not grid_str:
        return None
    parts = [p.strip() for p in grid_str.split(",") if p.strip()]
    return [int(p) for p in parts]


def train_and_select_best_hmm(
    scaled_series: np.ndarray,
    deltas: np.ndarray,
    train_deltas: np.ndarray,
    val_size: int,
    cfg: HMMConfig,
) -> Tuple[GaussianHMM, dict]:
    N = deltas.shape[0]
    T = scaled_series.shape[0]
    prices_scaled = scaled_series.reshape(-1)
    start_val_idx = max(2, T - val_size)

    def eval_model(model: GaussianHMM) -> Tuple[float, dict]:
        pred_scaled_all = one_step_ahead_price_predictions(scaled_series, deltas, model)
        y_true_val_scaled = prices_scaled[start_val_idx:]
        y_pred_val_scaled = pred_scaled_all[start_val_idx:]

        mask = ~np.isnan(y_pred_val_scaled)
        y_true_val_scaled = y_true_val_scaled[mask]
        y_pred_val_scaled = y_pred_val_scaled[mask]

        m_scaled = regression_metrics(y_true_val_scaled, y_pred_val_scaled)
        return m_scaled["RMSE"], m_scaled

    state_grid = parse_state_grid(cfg.state_grid)
    if not state_grid:
        print(f"[HMM] Fitting single model with n_states={cfg.n_states}...")
        model = fit_hmm(train_deltas, cfg.n_states, cfg.n_iter, cfg.covariance_type, tol=cfg.tol)
        rmse, metrics_scaled = eval_model(model)
        print(f"[HMM] Single-model RMSE (scaled price val): {rmse:.6f}")
        return model, metrics_scaled

    best_model: Optional[GaussianHMM] = None
    best_rmse, best_metrics = float("inf"), {}

    for ns in state_grid:
        print(f"[HMM] Trying n_states={ns}...")
        model = fit_hmm(train_deltas, ns, cfg.n_iter, cfg.covariance_type, tol=cfg.tol)
        rmse, m_scaled = eval_model(model)
        print(f"[HMM]   -> val RMSE (scaled price): {rmse:.6f}")
        if rmse < best_rmse:
            best_rmse, best_model, best_metrics = rmse, model, m_scaled

    assert best_model is not None, "Failed to train any HMM in state_grid."
    print(f"[HMM] Best n_states={best_model.n_components}, RMSE={best_rmse:.6f}")
    return best_model, best_metrics


def parse_args_to_config() -> HMMConfig:
    parser = argparse.ArgumentParser(description="Gaussian HMM time-series predictor.")
    parser.add_argument("--data_path", type=str, default="energy-trading-hackathon-2025")
    parser.add_argument("--filename", type=str, default="Dataset_clean.csv")
    parser.add_argument("--sample_submission", type=str,
                        default="energy-trading-hackathon-2025/sample_submission.csv")
    parser.add_argument("--out_csv", type=str,
                        default="energy-trading-hackathon-2025/submission_hmm.csv")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--n_states", type=int, default=6)
    parser.add_argument("--state_grid", type=str, default="")
    parser.add_argument("--n_iter", type=int, default=160)
    parser.add_argument("--covariance_type", type=str, default="diag")
    parser.add_argument("--tol", type=float, default=1e-4)

    args = parser.parse_args()
    return HMMConfig(
        data_path=args.data_path,
        filename=args.filename,
        sample_submission=args.sample_submission,
        out_csv=args.out_csv,
        val_ratio=args.val_ratio,
        n_states=args.n_states,
        state_grid=args.state_grid,
        n_iter=args.n_iter,
        covariance_type=args.covariance_type,
        tol=args.tol,
    )


def main() -> None:
    cfg = parse_args_to_config()
    data_path = os.path.join(cfg.data_path, cfg.filename)

    print(f"[main] Loading data from: {data_path}")
    raw = read_csv(data_path)
    ts = to_ts(raw)

    ts_scaled, scaler = normalize_data(ts, col="Price")
    scaled_series = ts_scaled["Price"].values.reshape(-1, 1)

    adf_test(ts_scaled["Price"], label="Scaled Price (full history)")

    deltas = build_deltas(scaled_series)
    N = deltas.shape[0]
    
    val_size = int(N * cfg.val_ratio)
    train_size = N - val_size
    train_deltas = deltas[:train_size]

    best_model, m_scaled = train_and_select_best_hmm(
        scaled_series=scaled_series,
        deltas=deltas,
        train_deltas=train_deltas,
        val_size=val_size,
        cfg=cfg,
    )

    print("\n[metrics] Validation (scaled PRICES) from best HMM")
    for k, v in m_scaled.items():
        print(f"  {k}: {v:.6f}")

    # Forecast
    n_forecast = cfg.n_days * cfg.intervals_per_day
    prices_scaled_full = scaled_series.reshape(-1)
    last_price_scaled = float(prices_scaled_full[-1])

    state_probs_full = best_model.predict_proba(deltas)
    last_state_probs = state_probs_full[-1]

    preds_scaled, forecast_states = forecast_hmm_states_and_prices(
        last_price_scaled=last_price_scaled,
        last_state_probs=last_state_probs,
        model=best_model,
        n_steps=n_forecast,
    )
    preds_price = scaler.inverse_transform(preds_scaled).flatten()
    print(f"\n[main] Generated HMM forecast of length {len(preds_price)}")

    # Trading
    actions = generate_battery_actions(
        preds_price,
        intervals_per_day=cfg.intervals_per_day,
        capacity_mwh=cfg.capacity_mwh,
        daily_reset_mwh=cfg.daily_reset_mwh,
        min_transaction_mwh=cfg.min_transaction_mwh,
        n_days=cfg.n_days,
    )
    print(f"[main] Actions: min={actions.min():.3f}, max={actions.max():.3f}, mean={actions.mean():.3f}")

    # Validator
    is_valid, soc_trace, warnings = validate_battery_actions(
        actions,
        capacity=cfg.capacity_mwh,
        initial_soc=cfg.daily_reset_mwh,
        timestep_hours=1 / 4,
        return_trace=True,
        reset_daily=True,
    )
    print(f"[main] validate_battery_actions -> is_valid={is_valid}, warnings={len(warnings)}")
    if warnings:
        print("[main] First few validation warnings:")
        for w in warnings[:10]:
            print("  ", w)

    # Save
    submission = pd.read_csv(cfg.sample_submission)
    sub_ts = to_ts(submission.copy())
    check_battery_constraints(actions, sub_ts.index, capacity_mwh=cfg.capacity_mwh, daily_reset_mwh=cfg.daily_reset_mwh)

    submission["Position"] = np.round(actions, 3)
    submission.to_csv(cfg.out_csv, index=False)
    print(f"[main] Saved submission to: {cfg.out_csv}")


if __name__ == "__main__":
    main()
