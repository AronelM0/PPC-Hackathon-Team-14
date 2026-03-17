import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas.plotting import autocorrelation_plot, lag_plot
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf

from handling_data import (
    read_csv,
    to_ts,
    normalize_data,
    adf_test,
)


# ---------- BASIC PLOTS ----------

def plot_ts(df: pd.DataFrame, title="Price time series", out_dir="output"):
    plt.figure(figsize=(12, 4))
    df["Price"].plot()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "price_time_series.png"))
    # plt.show()


def plot_hist(df: pd.DataFrame, bins: int = 30, out_dir="output"):
    plt.figure(figsize=(8, 3))
    plt.hist(df["Price"], bins=bins, density=True, alpha=0.7)
    plt.title("Price distribution")
    plt.xlabel("Price")
    plt.ylabel("Density")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "price_histogram.png"))
    # plt.show()


def plot_acf_lag(df: pd.DataFrame, out_dir="output"):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    autocorrelation_plot(df["Price"])
    plt.title("Autocorrelation")

    plt.subplot(1, 2, 2)
    lag_plot(df["Price"])
    plt.title("Lag plot")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "acf_lag_plot.png"))
    # plt.show()


# ---------- POWER SPECTRAL DENSITY (Welch) ----------

def plot_psd(
    df: pd.DataFrame,
    fs: float = 4.0,
    nfft: int = 256,
    pad_to: int | None = None,
    noverlap: int | None = None,
    sides: str = "default",
    scale_by_freq: bool = True,
    out_dir: str = "output",
):
    """
    Plot power spectral density using matplotlib.pyplot.psd.
    15-min data → fs = 4 samples/hour.
    """
    x = df["Price"].values
    plt.figure(figsize=(10, 3.5))
    plt.psd(
        x,
        NFFT=nfft,
        Fs=fs,
        noverlap=noverlap,
        pad_to=pad_to,
        sides=sides,
        scale_by_freq=scale_by_freq,
    )
    plt.title("Power Spectral Density of Price")
    plt.xlabel("Frequency")
    plt.ylabel("Power / Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "price_psd.png"))
    # plt.show()


# ---------- MOVING AVERAGES (SMA & EMA) ----------

def compute_mas(df: pd.DataFrame, sma_window: int = 7, ema_window: int = 30):
    """
    Simple/exp. moving averages – for visual trend smoothing.
    """
    s = df["Price"]
    sma = s.rolling(window=sma_window).mean()
    ema = s.ewm(span=ema_window, adjust=False).mean()
    return sma, ema


def plot_mas(df: pd.DataFrame, sma_window: int = 7, ema_window: int = 30, out_dir="output"):
    s = df["Price"]
    sma, ema = compute_mas(df, sma_window, ema_window)

    plt.figure(figsize=(12, 4))
    plt.plot(s, label="Original")
    plt.plot(sma, label=f"{sma_window}-period SMA")
    plt.plot(ema, label=f"{ema_window}-period EMA")
    plt.title("Original vs Moving Averages")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "moving_averages.png"))
    # plt.show()


# ---------- SEASONALITY, DECOMPOSITION & DIFFERENCING ----------

def detect_seasonality(df: pd.DataFrame, max_lag: int = 365) -> int:
    """
    Detect dominant seasonal lag using ACF (excluding lag 0).
    Returns lag in samples (0 if nothing clear).

    This matches the 'seasonal adjustment' discussion:
    we look for a strong repeating pattern at some lag.
    """
    x = df["Price"].dropna().values
    if len(x) < 3:
        return 0
    nlags = min(max_lag, len(x) - 1)
    vals = acf(x, nlags=nlags, fft=True)
    lag = int(np.argmax(vals[1:]) + 1)
    return lag if lag > 1 else 0


def decompose(
    df,
    model="additive",
    period=None,
    out_dir="output",
):
    """
    Decompose the series and *also* show a clearer intra-period seasonal profile.

    We keep the classical 4-panel decomposition (Observed / Trend / Seasonal / Residual)
    for completeness, but for a long and noisy series the seasonal panel tends to look
    like a solid bar. To actually *see* the pattern, we also aggregate the seasonal
    component over one full period.
    """
    print("\n[decompose] === Seasonal decomposition ===")
    series = df["Price"].astype(float).dropna()
    print(f"[decompose] Series length: {len(series)}")

    # 1) Detect / confirm period
    if period is None:
        period = detect_seasonality(
            df,
            max_lag=7 * 96,   # up to 1 week for 15-min data
        )
        print(f"[decompose] Detected seasonal period from ACF: {period}")
    else:
        print(f"[decompose] Using provided seasonal period: {period}")

    # 2) Classical decomposition (mainly for reference)
    dec = seasonal_decompose(
        series,
        model=model,
        period=period,
        extrapolate_trend="freq",
    )

    fig = dec.plot()
    fig.set_size_inches(12, 7)
    plt.suptitle(f"Seasonal decomposition (model={model}, period={period})", y=1.02)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "seasonal_decomposition.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[decompose] Saved standard decomposition to: {out_path}")

    # 3) Intra-period seasonal profile (this is the more informative part)
    seasonal = pd.Series(dec.seasonal).dropna()
    if period is not None and period > 1 and len(seasonal) >= period:
        print(f"[decompose] Building intra-period seasonal profile (period={period})...")
        idx_mod = np.arange(len(seasonal)) % period
        profile = pd.Series(seasonal.values, index=idx_mod).groupby(level=0).mean()

        fig2 = plt.figure(figsize=(10, 4))
        ax = fig2.add_subplot(111)
        ax.plot(np.arange(period), profile.values)
        ax.set_title(f"Intra-period seasonal pattern (period={period})")
        ax.set_xlabel(f"Position within period (0..{period-1})")
        ax.set_ylabel("Average seasonal component")
        ax.grid(True)

        # For 15-minute energy prices (period=96), label x-axis in clock time
        if period in (96, 48, 24):
            step_minutes = int(24 * 60 / period)  # minutes per step
            # tick every 6 hours
            steps_per_6h = max(1, int((6 * 60) // step_minutes))
            tick_positions = list(range(0, period, steps_per_6h))
            tick_labels = []
            for k in tick_positions:
                minutes = k * step_minutes
                hh = minutes // 60
                mm = minutes % 60
                tick_labels.append(f"{hh:02d}:{mm:02d}")
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45)

        plt.tight_layout()
        profile_path = os.path.join(out_dir, "seasonal_profile.png")
        plt.savefig(profile_path, bbox_inches="tight")
        plt.close(fig2)
        print(f"[decompose] Saved intra-period seasonal profile to: {profile_path}")
    else:
        print("[decompose] Not enough data to build seasonal profile or period<=1.")

    return dec, period


def remove_seasonality(df: pd.DataFrame, period: int, model: str = "additive") -> pd.Series:
    """
    Remove seasonal component using seasonal_decompose.

    This corresponds to the 'seasonal adjustment' step in your text.
    """
    s = df["Price"]
    dec = seasonal_decompose(s, model=model, period=period)

    if model == "multiplicative":
        deseasonal = s / dec.seasonal
    else:
        deseasonal = s - dec.seasonal

    return deseasonal


# ---------- MAIN: quick EDA run ----------

def parser_args():
    import argparse

    parser = argparse.ArgumentParser(description="Time Series Visualization")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="output",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        default="energy-trading-hackathon-2025",
        help="Path to the CSV data file",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="Dataset.csv",
        help="Filename of the dataset to visualize",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parser_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    base_dir = args.data_path
    filename = args.filename
    path = os.path.join(base_dir, filename)

    # --- Read & convert to proper time series ---------------------------
    raw = read_csv(path)
    ts_raw = to_ts(raw)          # index = timestamp, column "Price"

    # --- Scale to [0, 1] JUST for visualization / ML intuition ---------
    ts, _ = normalize_data(ts_raw, col="Price")
    print(
        "[main] MinMax-scaled Price in [0, 1]. "
        f"min={ts['Price'].min():.4f}, max={ts['Price'].max():.4f}"
    )

    # --- Basic exploratory plots (G4G: trend / seasonality / noise) -----
    plot_ts(ts, out_dir=out_dir)              # Price vs time (scaled)
    plot_hist(ts, out_dir=out_dir)            # distribution (scaled)
    plot_acf_lag(ts, out_dir=out_dir)         # ACF + lag plot

    # --- Spectrum / frequency domain (G4G: spectrum analysis) -----------
    # 15-min data -> 4 samples/hour. Scaling does not change frequencies.
    plot_psd(ts, fs=4.0, nfft=256, pad_to=512, noverlap=128, out_dir=out_dir)

    # --- Moving averages (G4G: trend & smoothing) ----------------------
    plot_mas(ts, sma_window=7, ema_window=30, out_dir=out_dir)

    # --- Decomposition & seasonal adjustment (G4G: decomposition) ------
    # Uses daily period=96 by default now.
    dec, period = decompose(ts, model="additive", period=96, out_dir=out_dir)
    print("[main] Used seasonal period for decomposition:", period)

    # Seasonal adjustment on the scaled series
    deseasonal = remove_seasonality(ts, period=period, model="additive")

    # --- Stationarity check (G4G: ADF, differencing) -------------------
    # Check ADF on the *scaled* original and deseasonalized series.
    adf_test(ts["Price"], label="Scaled Price (original)")
    adf_test(deseasonal, label="Scaled Price (deseasonalized)")

    # Optional: 1-lag differencing on deseasonalized series, still in [~−1,1]
    diff_series = deseasonal.diff().dropna()
    adf_test(diff_series, label="Scaled Price (deseasonalized & diff=1)")

    # --- Visual comparison (original vs deseasonalized vs differenced) --
    plt.figure(figsize=(12, 8))

    # Top: original vs deseasonalized
    plt.subplot(2, 1, 1)
    plt.plot(ts.index, ts["Price"], label="Original (scaled)", alpha=0.8)
    plt.plot(deseasonal.index, deseasonal, label="Deseasonalized (scaled)", alpha=0.8)
    plt.title("Original vs Deseasonalized (Scaled Price)")
    plt.xlabel("Time")
    plt.ylabel("Price (0-1)")
    plt.legend()
    plt.grid(True)

    # Bottom: deseasonalized & differenced (still bounded, not +/- 10000)
    plt.subplot(2, 1, 2)
    plt.plot(diff_series.index, diff_series,
            label="Deseasonalized & Differenced (scaled)", color="orange")
    plt.title("After Differencing (Closer to Stationary)")
    plt.xlabel("Time")
    plt.ylabel("Transformed Price")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "seasonality_differencing.png"))
    # plt.show()
