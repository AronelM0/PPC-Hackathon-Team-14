import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.stattools import (
    adfuller,
    acf as sm_acf,
    pacf as sm_pacf,
    grangercausalitytests,
)
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats, signal
from typing import Tuple, Optional, Dict, Any


# ============================================================
#  IO & PREP (used everywhere)
# ============================================================

def read_csv(path: str) -> pd.DataFrame:
    """
    Read CSV with the 2 required columns and ensure Price is numeric.
    """
    print(f"[read_csv] Loading CSV from: {path}")
    df = pd.read_csv(path, usecols=["Time interval (CET/CEST)", "Price"])
    total_raw = len(df)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    numeric_ok = df["Price"].notna().sum()
    # Drop rows where Price could not be converted
    df = df.dropna(subset=["Price"])
    after_drop = len(df)
    print(
        f"[read_csv] Loaded {total_raw} rows. "
        f"Numeric Price: {numeric_ok}. "
        f"After dropping NaNs in Price: {after_drop} rows."
    )
    return df


def to_ts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw DataFrame to a time-indexed series.

    'Time interval (CET/CEST)' looks like:
        '01.02.2021 00:00 - 01.02.2021 00:15'
    We keep only the *start* time as the timestamp.
    """
    print("[to_ts] Converting 'Time interval (CET/CEST)' to datetime index...")
    df = df.copy()
    start_str = df["Time interval (CET/CEST)"].astype(str).str.split(" - ", n=1).str[0]
    start_dt = pd.to_datetime(start_str, format="%d.%m.%Y %H:%M", errors="coerce")
    df.index = start_dt
    before = len(df)
    df = df[~df.index.isna()].sort_index()
    after = len(df)
    print(f"[to_ts] Dropped {before - after} rows with invalid timestamps. Final rows: {after}")
    return df


# ============================================================
#  OUTLIERS
# ============================================================

def outliers_removal(
    df: pd.DataFrame,
    col: str = "Price",
    z_thresh: float = 3.0,
) -> pd.DataFrame:
    """
    Remove outliers from the given column using a Z-score threshold.

    In practice:
      * z_thresh ≈ 3.0  → keep ~99.7% of Gaussian-like data.
      * Lower thresholds (2.0–2.5) are more aggressive.
    NOTE: in energy prices, big spikes may be real signals; use with care.
    """
    print(f"[outliers_removal] Using Z-score threshold z_thresh={z_thresh}")
    s = df[col].astype(float)
    z_scores = np.abs(stats.zscore(s, nan_policy="omit"))
    mask = z_scores < z_thresh
    total = len(df)
    kept = int(mask.sum())
    removed = total - kept
    print(f"[outliers_removal] Removing {removed} / {total} rows ({removed/total:.3%})")
    return df.loc[mask].copy()


def clip_outliers_iqr(
    df: pd.DataFrame,
    col: str = "Price",
    factor: float = 1.5
) -> pd.DataFrame:
    """
    Clip extreme values using IQR bounds instead of dropping rows.
    """
    print(f"[clip_outliers_iqr] Clipping with IQR factor={factor}")
    s = df[col].astype(float)
    q1, q3 = np.percentile(s, [25, 75])
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    print(f"[clip_outliers_iqr] Q1={q1:.4f}, Q3={q3:.4f}, IQR={iqr:.4f}")
    print(f"[clip_outliers_iqr] Clipping to [{lower:.4f}, {upper:.4f}]")

    df = df.copy()
    df[col] = s.clip(lower=lower, upper=upper)
    return df


# ============================================================
#  STATIONARITY HELPERS (for analysis / ARIMA-style models)
# ============================================================

def describe_stationarity_chunks(
    series: pd.Series,
    n_chunks: int = 3,
    label: str = "Series"
) -> Dict[str, Dict[str, float]]:
    """
    Rough stationarity check à la GeeksforGeeks:
    split the series into `n_chunks` time-ordered parts and
    compute mean/variance for each part.
    """
    s = series.dropna().astype(float).values
    n = len(s)
    if n < n_chunks:
        print(f"[describe_stationarity_chunks] Too few points ({n}) for {n_chunks} chunks.")
        return {}

    size = n // n_chunks
    stats_dict: Dict[str, Dict[str, float]] = {}

    print(f"\n[describe_stationarity_chunks] {label} – {n_chunks} chunks, total n={n}")
    for i in range(n_chunks):
        start = i * size
        end = (i + 1) * size if i < n_chunks - 1 else n
        chunk = s[start:end]
        m = float(chunk.mean())
        v = float(chunk.var())
        stats_dict[f"chunk_{i+1}"] = {"mean": m, "var": v}
        print(f"  chunk_{i+1}: mean={m:.6f}, var={v:.6f}, n={len(chunk)}")

    print("  -> If means/vars similar across chunks → closer to stationary.\n")
    return stats_dict


def adf_test(series: pd.Series, label: str = "Series") -> float:
    """
    Augmented Dickey-Fuller test for stationarity.
    Returns p-value.  p < 0.05 => stationary (reject unit-root H0).
    """
    series = series.dropna().astype(float)
    print(f"[adf_test] Running ADF for {label} (n={len(series)})...")
    stat, p, *_ = adfuller(series)
    print(f"  ADF - {label}: statistic={stat:.4f}, p-value={p:.4f}")
    if p < 0.05:
        print("  => Stationary (reject H0: unit root)\n")
    else:
        print("  => Non-stationary (fail to reject H0)\n")
    return p


def detrend_and_deseasonalize(
    df: pd.DataFrame,
    col: str = "Price",
    model: str = "additive",
    period: Optional[int] = None,
    remove_trend: bool = True,
    remove_seasonal: bool = True,
) -> pd.Series:
    """
    Remove trend and/or seasonal components via classical decomposition.
    Used for analysis / ARIMA-style modelling, not in the default pipeline.
    """
    s = df[col].astype(float).dropna()

    if period is None:
        print("[detrend_and_deseasonalize] period=None → returning original series.")
        return s

    print(
        f"[detrend_and_deseasonalize] Decomposing (model={model}, "
        f"period={period}, remove_trend={remove_trend}, remove_seasonal={remove_seasonal})"
    )

    if len(s) < 2 * period:
        print(
            f"[detrend_and_deseasonalize] WARNING: "
            f"series length {len(s)} < 2 * period ({2*period}). "
            "Decomposition may be unreliable."
        )

    dec = seasonal_decompose(
        s,
        model=model,
        period=period,
        extrapolate_trend="freq"
    )

    trend = dec.trend if remove_trend else 0
    seasonal = dec.seasonal if remove_seasonal else 0

    if model == "multiplicative":
        trend_adj = trend.replace(0, np.nan)
        seas_adj = seasonal.replace(0, np.nan)
        resid = s / (trend_adj * seas_adj)
    else:
        resid = s - trend - seasonal

    resid = resid.dropna()
    print(f"[detrend_and_deseasonalize] Residual length after dropna: {len(resid)}")
    return resid


def seasonal_difference(
    series: pd.Series,
    order: int = 1,
    period: int = 1
) -> pd.Series:
    """
    Seasonal differencing:
        order=1:  y(t) - y(t-period)
        order=2:  apply differencing twice, etc.
    """
    print(f"[seasonal_difference] order={order}, period={period}")
    s = series.astype(float)
    for k in range(order):
        s = s.diff(periods=period)
        print(f"  after order {k+1}, n={s.dropna().shape[0]}")
    return s.dropna()


def stationary_transform(
    df: pd.DataFrame,
    col: str = "Price",
    diff_order: int = 1,
    seasonal_period: Optional[int] = None,
    seasonal_diff_order: int = 0,
    seasonal_diff_period: Optional[int] = None,
    model: str = "additive",
    log_transform: bool = False,
) -> pd.Series:
    """
    Stationarity processing for ARIMA-like models.

    raw → (optional log) → (optional decomposition)
        → regular differencing → seasonal differencing.
    """
    print("\n[stationary_transform] === Stationarity pipeline start ===")
    s_raw = df[col].astype(float)
    print(f"[stationary_transform] Raw series length: {len(s_raw)}")

    describe_stationarity_chunks(s_raw, n_chunks=3, label=f"{col} (raw)")

    # ---- Log transform ----
    if log_transform:
        s_work_raw = s_raw.copy()
        min_val = s_work_raw.min()
        if min_val <= 0:
            shift = abs(min_val) + 1e-6
            print(f"[stationary_transform] Log-transform shift applied: +{shift:.6f}")
            s_work_raw = s_work_raw + shift
        s_work_raw = s_work_raw.dropna()
        s_work = np.log1p(s_work_raw)
        print("[stationary_transform] Applied safe log1p transform.")
        adf_test(s_work, label=f"{col} (log)")
    else:
        s_work = s_raw
        adf_test(s_work, label=f"{col} (raw)")

    # ---- Seasonal / trend adjustment ----
    if seasonal_period is not None:
        s_adj = detrend_and_deseasonalize(
            df.assign(**{col: s_work}),
            col=col,
            model=model,
            period=seasonal_period,
            remove_trend=True,
            remove_seasonal=True,
        )
        print(f"[stationary_transform] After decomposition: n={len(s_adj)}")
        adf_test(s_adj, label=f"{col} (deseasonalized)")
    else:
        print("[stationary_transform] No seasonal_period provided → skipping decomposition.")
        s_adj = s_work

    # ---- Regular differencing ----
    s_diff = s_adj.copy()
    for i in range(diff_order):
        s_diff = s_diff.diff().dropna()
        print(f"[stationary_transform] After {i+1}-order differencing: n={len(s_diff)}")
        adf_test(s_diff, label=f"{col} (diff={i+1})")

    # ---- Seasonal differencing ----
    if seasonal_diff_order > 0 and seasonal_diff_period is not None:
        s_seas_diff = seasonal_difference(
            s_diff,
            order=seasonal_diff_order,
            period=seasonal_diff_period
        )
        print(
            f"[stationary_transform] After seasonal differencing "
            f"(order={seasonal_diff_order}, period={seasonal_diff_period}): "
            f"n={len(s_seas_diff)}"
        )
        adf_test(s_seas_diff, label=f"{col} (seasonal diff)")
        print("[stationary_transform] === Stationarity pipeline end ===\n")
        return s_seas_diff

    print("[stationary_transform] === Stationarity pipeline end ===\n")
    return s_diff


# ============================================================
#  G4G-STYLE TIME SERIES ANALYSIS UTILITIES
# ============================================================

# --- Autocorrelation & PACF ---

def autocorrelation_analysis(
    series: pd.Series,
    nlags: int = 40,
    fft: bool = True,
) -> np.ndarray:
    """
    Compute autocorrelation coefficients up to `nlags`.

    This corresponds to the 'Autocorrelation Analysis' section:
    high values at lag k indicate that y(t) and y(t-k) move together.
    """
    x = series.dropna().astype(float).values
    acf_vals = sm_acf(x, nlags=nlags, fft=fft)
    print(f"[autocorrelation_analysis] First 5 ACF values: {acf_vals[:5]}")
    return acf_vals


def pacf_analysis(
    series: pd.Series,
    nlags: int = 40,
    method: str = "ywm",
) -> np.ndarray:
    """
    Compute partial autocorrelation coefficients up to `nlags`.
    PACF is used in Box-Jenkins identification: a sharp cut-off in PACF
    often suggests the AR order.
    """
    x = series.dropna().astype(float).values
    pacf_vals = sm_pacf(x, nlags=nlags, method=method)
    print(f"[pacf_analysis] First 5 PACF values: {pacf_vals[:5]}")
    return pacf_vals


# --- Trend & Seasonality ---

def moving_average_trend(
    series: pd.Series,
    window: int = 96,
) -> pd.Series:
    """
    Extract a simple trend using moving average.

    For 15-minute data, window=96 corresponds to a daily trend.
    """
    trend = series.rolling(window=window, center=True).mean()
    print(
        f"[moving_average_trend] Computed moving-average trend with "
        f"window={window}, non-NaN points={trend.dropna().shape[0]}"
    )
    return trend


def detect_seasonal_period_acf(series: pd.Series, max_lag: int = 365) -> int:
    """
    Detect dominant seasonal lag using ACF (excluding lag 0).

    Returns lag in samples (0 if nothing clear).
    """
    x = series.dropna().astype(float).values
    if len(x) < 3:
        print("[detect_seasonal_period_acf] Too short series.")
        return 0
    nlags = min(max_lag, len(x) - 1)
    vals = sm_acf(x, nlags=nlags, fft=True)
    lag = int(np.argmax(vals[1:]) + 1)
    print(
        f"[detect_seasonal_period_acf] Max ACF (excluding lag0) "
        f"at lag={lag}, value={vals[lag]:.4f}"
    )
    return lag if lag > 1 else 0


# --- STL Decomposition ---

def stl_decomposition(
    series: pd.Series,
    period: int,
    robust: bool = True,
) -> STL:
    """
    Seasonal and Trend decomposition using Loess (STL).

    More flexible than classical seasonal_decompose; used often in
    GeeksForGeeks' STL examples.
    """
    x = series.dropna().astype(float)
    print(
        f"[stl_decomposition] Running STL with period={period}, "
        f"robust={robust}, n={len(x)}"
    )
    stl = STL(x, period=period, robust=robust)
    res = stl.fit()
    return res


# --- Spectrum / Periodogram ---

def spectrum_analysis(
    series: pd.Series,
    fs: float = 1.0,
    nfft: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple spectrum analysis using the periodogram.

    Returns (frequencies, power).
    """
    x = series.dropna().astype(float).values
    freqs, power = signal.periodogram(x, fs=fs, nfft=nfft, scaling="density")
    print(
        f"[spectrum_analysis] Computed periodogram, "
        f"{len(freqs)} frequency bins, fs={fs}, nfft={nfft}"
    )
    return freqs, power


# --- Rolling correlation ---

def rolling_correlation(
    series1: pd.Series,
    series2: pd.Series,
    window: int = 96,
) -> pd.Series:
    """
    Rolling correlation between two series.

    Matches 'Rolling correlation' examples: useful to see how the
    relationship between two variables changes over time.
    """
    s1 = series1.astype(float)
    s2 = series2.astype(float)
    corr = s1.rolling(window=window).corr(s2)
    print(
        f"[rolling_correlation] window={window}, "
        f"non-NaN correlations={corr.dropna().shape[0]}"
    )
    return corr


# --- Cross-correlation ---

def cross_correlation(
    series1: pd.Series,
    series2: pd.Series,
    max_lag: int = 40,
) -> pd.DataFrame:
    """
    Cross-correlation between two series up to +/- max_lag.

    Returns a DataFrame with 'lag' and 'corr', where positive lags mean
    that series1 is correlated with past values of series2.
    """
    x = series1.dropna().astype(float)
    y = series2.dropna().astype(float)
    n = min(len(x), len(y))
    x = x.iloc[:n].values
    y = y.iloc[:n].values

    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()

    corr_full = np.correlate(x, y, mode="full") / n
    lags = np.arange(-n + 1, n)
    mask = (lags >= -max_lag) & (lags <= max_lag)
    corr = corr_full[mask]
    lags = lags[mask]

    df_cc = pd.DataFrame({"lag": lags, "corr": corr})
    best = df_cc.iloc[df_cc["corr"].abs().idxmax()]
    print(
        f"[cross_correlation] max_lag={max_lag}, "
        f"strongest at lag={best['lag']} with corr={best['corr']:.4f}"
    )
    return df_cc


# --- Box–Jenkins (ARIMA helper) ---

def fit_arima_box_jenkins(
    series: pd.Series,
    order: Tuple[int, int, int] = (1, 0, 0),
    seasonal_order: Tuple[int, int, int, int] | None = None,
) -> Tuple[ARIMA, Any]:
    """
    Convenience wrapper for Box-Jenkins ARIMA modelling.
    Steps (conceptually):
    1. Identification → use ACF/PACF to propose 'order' / 'seasonal_order'.
    2. Estimation     → this function fits the ARIMA.
    3. Diagnostics    → check residuals from the fitted model.
    """
    x = series.dropna().astype(float)
    print(
        f"[fit_arima_box_jenkins] Fitting ARIMA{order} "
        f"seasonal={seasonal_order} on n={len(x)}"
    )
    model = ARIMA(x, order=order, seasonal_order=seasonal_order)
    res = model.fit()
    print("[fit_arima_box_jenkins] Fit complete. AIC={:.2f}, BIC={:.2f}".format(res.aic, res.bic))
    return model, res


# --- Granger causality ---

def granger_causality_analysis(
    cause: pd.Series,
    effect: pd.Series,
    maxlag: int = 5,
) -> Dict[int, float]:
    """
    Granger causality test: does 'cause' help predict 'effect'?

    Returns a dict {lag: p_value} for the F-test on each lag.

    Interpretation (as in G4G examples):
        p < 0.05 → we reject H0 and say "cause Granger-causes effect"
        at that lag.
    """
    df = pd.DataFrame({"effect": effect, "cause": cause}).dropna()
    print(
        f"[granger_causality_analysis] Testing if 'cause' → 'effect', "
        f"maxlag={maxlag}, n={len(df)}"
    )
    result = grangercausalitytests(df[["effect", "cause"]], maxlag=maxlag, verbose=False)

    pvals = {}
    for lag, tests in result.items():
        p_val = tests[0]["ssr_ftest"][1]
        pvals[lag] = p_val
        print(f"  lag={lag}: p-value={p_val:.4f}")

    return pvals


# ============================================================
#  SCALING & SPLITS
# ============================================================

def normalize_data(
    df: pd.DataFrame,
    col: str = "Price"
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Min-Max scale the given column to [0, 1].
    """
    print("[normalize_data] Applying MinMaxScaler to column:", col)
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[col] = scaler.fit_transform(df[[col]].astype(float))
    print(
        f"[normalize_data] Done. "
        f"Price range after scaling: min={df_scaled[col].min():.4f}, "
        f"max={df_scaled[col].max():.4f}"
    )
    return df_scaled, scaler


def temporal_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-ordered train/test split (no shuffling).
    """
    n = len(df)
    split_idx = int(np.floor((1.0 - test_size) * n))
    print(
        f"[temporal_train_test_split] test_size={test_size} → "
        f"split_idx={split_idx}, n_train={split_idx}, n_test={n - split_idx}"
    )
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


# ============================================================
#  MAIN PIPELINE (light, ML-friendly)
# ============================================================

def preprocess_pipeline(
    df: pd.DataFrame,
    col: str = "Price",
    remove_outliers: bool = True,
    use_iqr_clipping: bool = False,  # if False, use Z-score method
    z_thresh: float = 3.0,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seasonal_period: Optional[int] = None,  # None → no decomposition in pipeline
    diff_order: int = 0,
    seasonal_diff_order: int = 0,
    seasonal_diff_period: Optional[int] = None,
    log_transform: bool = False,
    model: str = "additive",
) -> Dict[str, Any]:
    """
    Full preprocessing pipeline for time-series modelling.

    Default behaviour (recommended for ML models based on your EDA):
        - optional mild outlier handling
        - NO decomposition / differencing
        - MinMax scaling
        - temporal train / val / test split

    Heavier stationarity processing is opt-in via diff_order / seasonal_* / log_transform.
    """
    print("\n[preprocess_pipeline] === Preprocessing pipeline START ===")
    print(f"[preprocess_pipeline] Initial rows: {len(df)}")
    print(
        f"[preprocess_pipeline] Config → remove_outliers={remove_outliers}, "
        f"use_iqr_clipping={use_iqr_clipping}, z_thresh={z_thresh}, "
        f"diff_order={diff_order}, seasonal_period={seasonal_period}, "
        f"seasonal_diff_order={seasonal_diff_order}, "
        f"seasonal_diff_period={seasonal_diff_period}, "
        f"log_transform={log_transform}, model={model}"
    )

    df_clean = df.copy()

    # ---- 1) Outliers ----
    if remove_outliers:
        if use_iqr_clipping:
            print("[preprocess_pipeline] Step 1: IQR clipping...")
            df_clean = clip_outliers_iqr(df_clean, col=col)
        else:
            print("[preprocess_pipeline] Step 1: Z-score outlier removal...")
            df_clean = outliers_removal(df_clean, col=col, z_thresh=z_thresh)
    else:
        print("[preprocess_pipeline] Step 1: outlier removal DISABLED.")

    print(f"[preprocess_pipeline] After outlier step: {len(df_clean)} rows")

    # ---- 2) Stationarity transform (optional, for ARIMA-style models) ----
    series_stationary = None
    if (diff_order > 0 or seasonal_period is not None or
            seasonal_diff_order > 0 or log_transform):
        print("[preprocess_pipeline] Step 2: Stationarity transform ENABLED.")
        series_stationary = stationary_transform(
            df_clean,
            col=col,
            diff_order=diff_order,
            seasonal_period=seasonal_period,
            seasonal_diff_order=seasonal_diff_order,
            seasonal_diff_period=seasonal_diff_period,
            model=model,
            log_transform=log_transform,
        )
    else:
        print("[preprocess_pipeline] Step 2: Stationarity transform SKIPPED.")

    # ---- 3) Scaling ----
    print("[preprocess_pipeline] Step 3: Scaling with MinMaxScaler...")
    df_scaled, scaler = normalize_data(df_clean, col=col)

    # ---- 4) Train / val / test split (time ordered) ----
    print("[preprocess_pipeline] Step 4: Temporal train/val/test split...")
    train, test = temporal_train_test_split(df_scaled, test_size=test_size)
    train, val = temporal_train_test_split(train, test_size=val_size / (1.0 - test_size))

    print(
        "[preprocess_pipeline] Split sizes → "
        f"train={len(train)}, val={len(val)}, test={len(test)}"
    )
    print("[preprocess_pipeline] === Preprocessing pipeline END ===\n")

    return {
        "df_clean": df_clean,
        "series_stationary": series_stationary,
        "df_scaled": df_scaled,
        "scaler": scaler,
        "train": train,
        "val": val,
        "test": test,
    }


# ============================================================
#  MAIN: run preprocessing & save to CSV
# ============================================================

if __name__ == "__main__":
    path_in = "energy-trading-hackathon-2025/Dataset.csv"
    path_clean = "energy-trading-hackathon-2025/Dataset_clean.csv"
    path_scaled = "energy-trading-hackathon-2025/Dataset_scaled.csv"

    raw = read_csv(path_in)
    ts = to_ts(raw)

    # Light, ML-oriented preprocessing:
    #   - NO outlier removal (keep price spikes as real market signals)
    #   - NO decomposition / differencing
    #   - MinMax scaling + temporal split
    res = preprocess_pipeline(
        ts,
        remove_outliers=False,
        use_iqr_clipping=False,
        z_thresh=3.0,
        test_size=0.2,
        val_size=0.1,
        diff_order=0,
        seasonal_period=None,
        seasonal_diff_order=0,
        seasonal_diff_period=None,
        log_transform=False,
    )

    print("[__main__] Saving outputs...")
    res["df_clean"].to_csv(path_clean, index=False)
    res["df_scaled"].to_csv(path_scaled, index=False)

    print("[__main__] Preprocessing complete.")
    print(f"[__main__] Saved cleaned data to:  {path_clean}")
    print(f"[__main__] Saved scaled data to:   {path_scaled}")
