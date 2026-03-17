import os
from typing import Tuple

import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import adfuller
import pmdarima as pm


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def read_csv_with_price(path: str,
                        time_col: str = "Time interval (CET/CEST)",
                        price_col: str = "Price") -> pd.DataFrame:
    print(f"[read_csv] Loading CSV from: {path}")
    df = pd.read_csv(path)
    print(
        f"[read_csv] Loaded {len(df)} rows. Numeric {price_col}: {df[price_col].notna().sum()}. "
        f"After dropping NaNs in {price_col}: {df[price_col].dropna().shape[0]} rows."
    )
    df = df.dropna(subset=[price_col]).copy()
    return df


def to_ts_index(df: pd.DataFrame,
                time_col: str = "Time interval (CET/CEST)") -> pd.DataFrame:
    """
    Convert '01.02.2021 00:00 - 01.02.2021 00:15' -> Timestamp('2021-02-01 00:00:00')
    and use it as the index.
    """
    print(f"[to_ts] Converting '{time_col}' to datetime index...")
    df = df.copy()

    raw = df[time_col].astype(str)
    start_str = raw.str.split(" - ").str[0].str.strip()

    df[time_col] = pd.to_datetime(
        start_str,
        format="%d.%m.%Y %H:%M",
        errors="coerce",
    )

    n_before = len(df)
    df = df.dropna(subset=[time_col])
    print(f"[to_ts] Dropped {n_before - len(df)} rows with invalid timestamps. Final rows: {len(df)}")

    df = df.set_index(time_col).sort_index()
    return df


def adf_test(series: pd.Series, name: str) -> None:
    clean = series.dropna()
    print(f"[adf_test] Running ADF for {name} (n={len(clean)})...")
    if len(clean) < 10:
        print("  Too few samples for ADF, skipping.\n")
        return

    result = adfuller(clean)
    stat, pvalue = result[0], result[1]
    print(f"  ADF - {name}: statistic={stat:.4f}, p-value={pvalue:.4f}")
    if pvalue < 0.05:
        print("  => Stationary (reject H0: unit root)\n")
    else:
        print("  => Non-stationary (fail to reject H0: unit root)\n")


# ---------------------------------------------------------------------------
# Strategy 1: High-contrast ranking-based trading policy (many trades)
# ---------------------------------------------------------------------------

def generate_battery_actions_rank(
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

        q_low, q_high = np.quantile(day_prices, [low_quantile, high_quantile])

        soc = float(daily_reset_mwh)

        for k in range(intervals_per_day):
            idx = start + k
            price = day_prices[k]
            intervals_left = intervals_per_day - k

            action = 0.0

            # --- End-of-day flush ---
            if intervals_left <= flush_last_intervals:
                if soc > 0.0:
                    proposed_action = -soc
                    max_charge = capacity_mwh - soc
                    max_discharge = -soc
                    action = float(np.clip(proposed_action, max_discharge, max_charge))
                else:
                    action = 0.0
            else:
                # --- Extremes of the day's forecast ---
                if price <= q_low and soc < capacity_mwh:
                    proposed_action = capacity_mwh - soc  # charge to full
                elif price >= q_high and soc > 0.0:
                    proposed_action = -soc                # discharge to zero
                else:
                    proposed_action = 0.0

                if proposed_action != 0.0:
                    max_charge = capacity_mwh - soc
                    max_discharge = -soc
                    action = float(np.clip(proposed_action, max_discharge, max_charge))
                else:
                    action = 0.0

            # Quantize
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
    print(f"[Strategy-RANK] Generated actions. Non-zero trades: {n_trades}/{len(actions)}")
    return actions


# ---------------------------------------------------------------------------
# Strategy 2: One-cycle-per-day (buy at daily min, sell at daily max)
# ---------------------------------------------------------------------------

def generate_battery_actions_one_cycle(
    forecast_prices: np.ndarray,
    intervals_per_day: int = 96,
    capacity_mwh: float = 10.0,
    daily_reset_mwh: float = 5.0,
    min_transaction_mwh: float = 0.1,
    n_days: int = 8,
    min_spread_factor: float = 0.3,
    flush_last_intervals: int = 1,
) -> np.ndarray:
    """
    One-cycle-per-day strategy.
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

        soc = float(daily_reset_mwh)

        # --- Find min & max (after min) within the day ---
        day_min_idx_rel = int(np.argmin(day_prices))

        if day_min_idx_rel == intervals_per_day - 1:
            # Min is at last interval, no time to sell afterwards
            for k in range(intervals_per_day):
                idx = start + k
                intervals_left = intervals_per_day - k
                action = 0.0

                if intervals_left <= flush_last_intervals and soc > 0.0:
                    proposed_action = -soc
                    max_charge = capacity_mwh - soc
                    max_discharge = -soc
                    action = float(np.clip(proposed_action, max_discharge, max_charge))

                # Quantize
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
            continue

        # Max only on the right side of the min (sell later than buy)
        right_segment = day_prices[day_min_idx_rel + 1:]
        if right_segment.size == 0:
            continue
        max_rel_after = int(np.argmax(right_segment))
        day_max_idx_rel = day_min_idx_rel + 1 + max_rel_after

        day_min_price = float(day_prices[day_min_idx_rel])
        day_max_price = float(day_prices[day_max_idx_rel])
        day_std = float(np.std(day_prices)) if np.std(day_prices) > 0 else 1.0
        spread = day_max_price - day_min_price

        # Require spread to be at least some fraction of daily std
        if spread < min_spread_factor * day_std:
            # Not worth trading this day; maybe just flush at the end
            for k in range(intervals_per_day):
                idx = start + k
                intervals_left = intervals_per_day - k
                action = 0.0

                if intervals_left <= flush_last_intervals and soc > 0.0:
                    proposed_action = -soc
                    max_charge = capacity_mwh - soc
                    max_discharge = -soc
                    action = float(np.clip(proposed_action, max_discharge, max_charge))

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
            continue

        # --- Execute daily plan: at min -> charge to full, at max -> discharge to zero ---
        for k in range(intervals_per_day):
            idx = start + k
            intervals_left = intervals_per_day - k
            action = 0.0

            if k == day_min_idx_rel and soc < capacity_mwh:
                # Charge to full at the minimum
                proposed_action = capacity_mwh - soc  # positive
                max_charge = capacity_mwh - soc
                max_discharge = -soc
                action = float(np.clip(proposed_action, max_discharge, max_charge))

            elif k == day_max_idx_rel and soc > 0.0:
                # Discharge to zero at the maximum
                proposed_action = -soc  # negative
                max_charge = capacity_mwh - soc
                max_discharge = -soc
                action = float(np.clip(proposed_action, max_discharge, max_charge))

            # Optional final flush if somehow SoC not zero
            elif intervals_left <= flush_last_intervals and soc > 0.0:
                proposed_action = -soc
                max_charge = capacity_mwh - soc
                max_discharge = -soc
                action = float(np.clip(proposed_action, max_discharge, max_charge))

            # Quantize
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
    print(f"[Strategy-ONECYCLE] Generated actions. Non-zero trades: {n_trades}/{len(actions)}")
    return actions


def check_battery_constraints(actions: np.ndarray,
                              intervals_per_day: int = 96,
                              capacity_mwh: float = 10.0,
                              daily_reset_mwh: float = 5.0) -> bool:
    soc = daily_reset_mwh
    n = len(actions)
    ok = True
    for t in range(n):
        if t % intervals_per_day == 0 and t != 0:
            soc = daily_reset_mwh
        soc += actions[t]
        if soc < -1e-6 or soc > capacity_mwh + 1e-6:
            print(f"[check_battery_constraints] Violation at t={t}, SoC={soc}")
            ok = False
            break
    if ok:
        print("[check_battery_constraints] All constraints satisfied (internal check).")
    return ok


# ---------------------------------------------------------------------------
# Non-seasonal ARIMA via pmdarima.auto_arima
# ---------------------------------------------------------------------------

def fit_arima(series: np.ndarray) -> pm.ARIMA:
    """
    Fit a *non-seasonal* ARIMA using pmdarima.auto_arima.
    """
    print("[ARIMA] Fitting non-seasonal auto_arima ...")
    model = pm.auto_arima(
        series,
        start_p=1,
        start_q=1,
        max_p=4,
        max_q=4,
        d=None,               # automatically choose differencing
        seasonal=False,       # no seasonal component
        max_order=8,
        trace=True,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )
    print(f"[ARIMA] Best model:\n{model.summary()}")
    return model


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    base_dir = "energy-trading-hackathon-2025"
    data_path = os.path.join(base_dir, "Dataset.csv")
    print(f"[main] Loading data from: {data_path}")
    df = read_csv_with_price(
        data_path,
        time_col="Time interval (CET/CEST)",
        price_col="Price",
    )
    df = to_ts_index(df, time_col="Time interval (CET/CEST)")

    prices = df["Price"].astype(float).values

    adf_test(df["Price"], "Raw Price")

    # Use only the last N days (e.g. 60 days) to keep it fast
    intervals_per_day = 96
    days_for_train = 60
    samples_for_train = days_for_train * intervals_per_day
    if samples_for_train < len(prices):
        train_series = prices[-samples_for_train:]
        print(
            f"[main] Using last {days_for_train} days "
            f"({samples_for_train} points) for ARIMA training."
        )
    else:
        train_series = prices
        print(f"[main] Using full history ({len(prices)} points) for ARIMA training.")

    # Fit ARIMA
    model = fit_arima(train_series)

    # Forecast next 8 days (768 intervals)
    horizon = 8 * intervals_per_day
    print(f"[main] Forecasting horizon={horizon}...")
    arima_forecast = model.predict(n_periods=horizon)
    arima_forecast = np.asarray(arima_forecast, dtype=float)
    print(f"[main] Generated ARIMA forecast of length {len(arima_forecast)}")

    # Strategy 1: rank-based (many trades)
    actions_rank = generate_battery_actions_rank(
        arima_forecast,
        intervals_per_day=intervals_per_day,
        capacity_mwh=10.0,
        daily_reset_mwh=5.0,
        min_transaction_mwh=0.1,
        n_days=8,
        low_quantile=0.20,
        high_quantile=0.80,
        flush_last_intervals=2,
    )

    # Strategy 2: one-cycle-per-day
    actions_one = generate_battery_actions_one_cycle(
        arima_forecast,
        intervals_per_day=intervals_per_day,
        capacity_mwh=10.0,
        daily_reset_mwh=5.0,
        min_transaction_mwh=0.1,
        n_days=8,
        min_spread_factor=0.3,
        flush_last_intervals=1,
    )

    # Check constraints
    check_battery_constraints(
        actions_rank,
        intervals_per_day=intervals_per_day,
        capacity_mwh=10.0,
        daily_reset_mwh=5.0,
    )
    check_battery_constraints(
        actions_one,
        intervals_per_day=intervals_per_day,
        capacity_mwh=10.0,
        daily_reset_mwh=5.0,
    )

    # Use helper_function if present
    try:
        from helper_function import validate_battery_actions
        is_valid_r, warnings_r = validate_battery_actions(actions_rank)
        print(f"[main] validate_battery_actions (rank) -> is_valid={is_valid_r}, warnings={warnings_r}")
        is_valid_o, warnings_o = validate_battery_actions(actions_one)
        print(f"[main] validate_battery_actions (onecycle) -> is_valid={is_valid_o}, warnings={warnings_o}")
    except Exception as e:
        print(f"[main] Could not import/use helper_function.validate_battery_actions: {e}")

    # Build submissions directly from last horizon timestamps
    if len(df) >= horizon:
        time_index = df.index[-horizon:]
    else:
        # Fallback: simple index if something is off
        print("[main] WARNING: not enough rows for horizon, using simple integer index instead.")
        time_index = np.arange(horizon)

    # Submission 1: rank-based
    sub_rank = pd.DataFrame({
        "Time interval (CET/CEST)": time_index,
        "Position": actions_rank[:horizon],
    })
    out_csv_rank = os.path.join(base_dir, "submission_arima_rank.csv")
    sub_rank.to_csv(out_csv_rank, index=False)
    print(f"[main] Saved submission to: {out_csv_rank}")

    # Submission 2: one-cycle-per-day
    sub_one = pd.DataFrame({
        "Time interval (CET/CEST)": time_index,
        "Position": actions_one[:horizon],
    })
    out_csv_one = os.path.join(base_dir, "submission_arima_onecycle.csv")
    sub_one.to_csv(out_csv_one, index=False)
    print(f"[main] Saved submission to: {out_csv_one}")


if __name__ == "__main__":
    main()
