#!/usr/bin/env python3
"""
lstm_series_predictor.py

PyTorch LSTM-based price forecasting + battery trading policy.

Improvements:
1. Uses Log-Returns (ln(p_t/p_{t-1})) for stationarity.
2. Implements Z-score based trading strategy.
3. Implements aggressive End-of-Day discharge to profit from daily reset.
"""

from typing import Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os

from handling_data import read_csv, to_ts, normalize_data, adf_test  # type: ignore
from helper_function import validate_battery_actions  # official validator


# -------------------- Reproducibility -------------------- #

np.random.seed(42)
torch.manual_seed(42)


# -------------------- Dataset utilities -------------------- #

def create_supervised_sequences(
    series_scaled: np.ndarray,
    lookback: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    X[i] = window of scaled prices
    y[i] = Log Return at time t: ln(p_t / p_{t-1})
    """
    values = np.asarray(series_scaled, dtype=np.float32)
    # Clip to avoid log(0)
    values = np.clip(values, 1e-6, 1.0)
    
    T = len(values)
    X, y = [], []
    for t in range(lookback, T):
        window = values[t - lookback:t]
        X.append(window)
        
        # Calculate Log Return
        curr = values[t][0]
        prev = values[t-1][0]
        log_ret = np.log(curr / prev)
        y.append(log_ret)

    X = np.stack(X, axis=0)       # (N, lookback, 1)
    y = np.array(y).reshape(-1, 1) # (N, 1)
    return X, y


# -------------------- Bi-encoder-decoder LSTM -------------------- #

class BiEncoderDecoderLSTM(nn.Module):
    """
    Bidirectional encoder + unidirectional decoder LSTM for 1-step regression.
    The network predicts a *Log Return*.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional encoder
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Bridge from 2 * hidden (bi) -> hidden (decoder)
        self.bridge_h = nn.Linear(2 * hidden_size, hidden_size)
        self.bridge_c = nn.Linear(2 * hidden_size, hidden_size)

        # Unidirectional decoder
        self.decoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, 1) scaled price window
        Returns: y_hat: (B, 1) one-step-ahead log return prediction
        """
        # ---- Encoder ----
        _, (h_n, c_n) = self.encoder(x)
        # h_n: (num_layers * 2, B, hidden_size)
        fwd_idx = (self.num_layers - 1) * 2
        bwd_idx = fwd_idx + 1

        h_fwd = h_n[fwd_idx, :, :]
        h_bwd = h_n[bwd_idx, :, :]
        c_fwd = c_n[fwd_idx, :, :]
        c_bwd = c_n[bwd_idx, :, :]

        h_enc = torch.cat([h_fwd, h_bwd], dim=-1)
        c_enc = torch.cat([c_fwd, c_bwd], dim=-1)

        # ---- Bridge to decoder hidden state ----
        h0_dec_single = torch.tanh(self.bridge_h(h_enc))
        c0_dec_single = torch.tanh(self.bridge_c(c_enc))

        # Repeat for num_layers
        h0_dec = h0_dec_single.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0_dec = c0_dec_single.unsqueeze(0).repeat(self.num_layers, 1, 1)

        # ---- Decoder ----
        dec_input = x[:, -1:, :]

        dec_out, _ = self.decoder(dec_input, (h0_dec, c0_dec))
        last_dec = dec_out[:, -1, :]
        y_hat = self.fc_out(last_dec)

        return y_hat


# -------------------- Training / evaluation -------------------- #

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for batch_idx, (xb, yb) in enumerate(loader):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)

        if max_batches is not None and max_batches > 0 and (batch_idx + 1) >= max_batches:
            break

    return total_loss / max(n, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            preds = model(xb)
            loss = criterion(preds, yb)
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
    return total_loss / max(n, 1)


# -------------------- Metrics -------------------- #

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

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE_%": mape,
        "R2": r2,
    }


# -------------------- Forecasting -------------------- #

def forecast_multi_step(
    model: nn.Module,
    last_window_scaled: np.ndarray,
    n_steps: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    # Ensure inputs are clipped for log safety
    window = np.clip(last_window_scaled.astype(np.float32).copy(), 1e-6, 1.0)
    preds_scaled_prices = []

    with torch.no_grad():
        for _ in range(n_steps):
            x = torch.from_numpy(window).unsqueeze(0).to(device)
            
            # Model predicts Log Return
            log_ret_hat = model(x).cpu().numpy()[0, 0]
            
            last_price = float(window[-1, 0])
            next_price = last_price * np.exp(float(log_ret_hat))
            next_price = float(np.clip(next_price, 0.0, 1.0))

            preds_scaled_prices.append(next_price)
            window = np.concatenate(
                [window[1:], np.array([[next_price]], dtype=np.float32)],
                axis=0
            )

    return np.array(preds_scaled_prices, dtype=np.float32).reshape(-1, 1)


# -------------------- Battery trading policy -------------------- #

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


# -------------------- Args -------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch LSTM time-series predictor for energy trading.")
    parser.add_argument("--data_path", type=str, default="energy-trading-hackathon-2025")
    parser.add_argument("--filename", type=str, default="Dataset_clean.csv")
    parser.add_argument("--sample_submission", type=str,
                        default="energy-trading-hackathon-2025/sample_submission.csv")
    parser.add_argument("--out_csv", type=str,
                        default="energy-trading-hackathon-2025/submission_lstm.csv")

    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--full_epochs", type=int, default=0)
    parser.add_argument("--train_fraction", type=float, default=0.3)
    parser.add_argument("--max_train_batches", type=int, default=200)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


# -------------------- Main -------------------- #

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = os.path.join(args.data_path, args.filename)
    out_csv = args.out_csv

    print(f"[main] Loading data from: {data_path}")
    raw = read_csv(data_path)
    ts = to_ts(raw)

    ts_scaled, scaler = normalize_data(ts, col="Price")
    scaled_series = ts_scaled["Price"].values.reshape(-1, 1)

    adf_test(ts_scaled["Price"], label="Scaled Price (full history)")

    # Create sequences
    lookback = args.lookback
    X_np, y_np = create_supervised_sequences(scaled_series, lookback)
    N_total = X_np.shape[0]

    # Use only recent data
    train_fraction = float(np.clip(args.train_fraction, 0.0 + 1e-6, 1.0))
    start_idx = int((1.0 - train_fraction) * N_total)
    X_np = X_np[start_idx:]
    y_np = y_np[start_idx:]
    
    prev_np_scaled = X_np[:, -1, :].copy()  # p_{t-1}

    X = torch.from_numpy(X_np)
    y = torch.from_numpy(y_np)

    N = X.shape[0]
    val_size = int(N * args.val_ratio)
    train_size = N - val_size
    print(f"[main] Train/val split: train={train_size}, val={val_size}")

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]
    prev_val_scaled = prev_np_scaled[train_size:]

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    full_dataset = TensorDataset(X, y)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    model = BiEncoderDecoderLSTM(
        input_size=1,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    max_train_batches = args.max_train_batches if args.max_train_batches > 0 else None

    # Training
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, max_batches=max_train_batches)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"[main] Epoch {epoch}/{args.epochs} - train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

    # Validation reconstruction
    model.eval()
    val_preds_list = []
    val_true_list = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            preds = model(xb).cpu().numpy()
            val_preds_list.append(preds)
            val_true_list.append(yb.numpy())

    y_val_pred_log_ret = np.concatenate(val_preds_list, axis=0)
    y_val_true_log_ret = np.concatenate(val_true_list, axis=0)

    # Reconstruct: p_t = p_{t-1} * exp(log_ret)
    # Clip prev_val to avoid weird multiplication
    prev_val_scaled = np.clip(prev_val_scaled, 1e-6, 1.0)
    
    y_val_true_price_scaled = prev_val_scaled * np.exp(y_val_true_log_ret)
    y_val_pred_price_scaled = prev_val_scaled * np.exp(y_val_pred_log_ret)

    y_val_true_price_scaled = np.clip(y_val_true_price_scaled, 0.0, 1.0)
    y_val_pred_price_scaled = np.clip(y_val_pred_price_scaled, 0.0, 1.0)

    m_scaled = regression_metrics(y_val_true_price_scaled, y_val_pred_price_scaled)
    print("\n[metrics] Validation (scaled PRICES in [0,1])")
    for k, v in m_scaled.items():
        print(f"  {k}: {v:.6f}")

    y_val_true_price = scaler.inverse_transform(y_val_true_price_scaled)
    y_val_pred_price = scaler.inverse_transform(y_val_pred_price_scaled)
    m_price = regression_metrics(y_val_true_price, y_val_pred_price)
    print("\n[metrics] Validation (real PRICES)")
    for k, v in m_price.items():
        print(f"  {k}: {v:.6f}")

    # Extra training
    if args.full_epochs > 0:
        print(f"\n[main] Extra training on FULL data for {args.full_epochs} epochs...")
        for epoch in range(1, args.full_epochs + 1):
            loss_all = train_one_epoch(model, full_loader, criterion, optimizer, device, max_batches=None)
            print(f"  [full-data] epoch {epoch}/{args.full_epochs} - loss={loss_all:.6f}")

    # Forecast
    n_forecast = 8 * 24 * 4
    last_window_scaled = scaled_series[-lookback:]
    preds_scaled = forecast_multi_step(model, last_window_scaled, n_steps=n_forecast, device=device)
    preds_price = scaler.inverse_transform(preds_scaled).flatten()
    print(f"\n[main] Generated forecast of length {len(preds_price)}")

    # Strategy
    actions = generate_battery_actions(preds_price, n_days=8)
    print(f"[main] Actions: min={actions.min():.3f}, max={actions.max():.3f}, mean={actions.mean():.3f}")

    # Validator
    is_valid, soc_trace, warnings = validate_battery_actions(
        actions,
        capacity=10,
        initial_soc=5,
        timestep_hours=1/4,
        return_trace=True,
        reset_daily=True,
    )
    print(f"[main] validate_battery_actions -> is_valid={is_valid}, warnings={len(warnings)}")
    if warnings:
        print("[main] First few validation warnings:")
        for w in warnings[:10]:
            print("  ", w)

    # Save
    submission = pd.read_csv(args.sample_submission)
    submission["Position"] = np.round(actions, 3)
    submission.to_csv(out_csv, index=False)
    print(f"[main] Saved submission to: {out_csv}")


if __name__ == "__main__":
    main()
