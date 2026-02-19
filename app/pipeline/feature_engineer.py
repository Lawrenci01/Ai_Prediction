"""
Feature Engineering Pipeline — FIXED
======================================
Key change: Reduced redundant CO2 lag/rolling features.

Previously 150 features were generated, with the top 20 all being
near-perfect copies of CO2 (corr > 0.997). This caused the LSTM to
collapse — it learned to output the mean (~0.665 scaled) rather than
learning actual patterns, because all inputs looked identical.

Fix: Keep only the most informative lags (1h, 24h, 168h) and one
rolling window per timeframe. Removed rollmax/rollmin/rollmom variants
that added noise without new information. Target feature count: ~60-70.
"""

import pandas as pd
import numpy as np
import logging
import pickle
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from app.pipeline.config import (
    TARGET_COLS, TIMESTAMP_COL,
    USE_CYCLICAL_ENCODING,
    WET_SEASON_MONTHS, get_model_path
)

logger = logging.getLogger(__name__)


# ============================================================================
# TIME FEATURES
# ============================================================================

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = df[TIMESTAMP_COL]

    df["month"]      = ts.dt.month
    df["dayofweek"]  = ts.dt.dayofweek
    df["dayofyear"]  = ts.dt.dayofyear
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    if USE_CYCLICAL_ENCODING:
        df["hour_sin"]      = np.sin(2 * np.pi * ts.dt.hour / 24)
        df["hour_cos"]      = np.cos(2 * np.pi * ts.dt.hour / 24)
        df["month_sin"]     = np.sin(2 * np.pi * ts.dt.month / 12)
        df["month_cos"]     = np.cos(2 * np.pi * ts.dt.month / 12)
        df["dayofyear_sin"] = np.sin(2 * np.pi * ts.dt.dayofyear / 365)
        df["dayofyear_cos"] = np.cos(2 * np.pi * ts.dt.dayofyear / 365)
        df["dayofweek_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
        df["dayofweek_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)

    df["hours_elapsed"] = (ts - ts.min()).dt.total_seconds() / 3600
    return df


# ============================================================================
# MANILA SEASON FEATURES
# ============================================================================

def add_manila_season_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = df[TIMESTAMP_COL]

    df["is_wet_season"]   = ts.dt.month.isin(WET_SEASON_MONTHS).astype(int)
    df["season_sin"]      = np.sin(2 * np.pi * ts.dt.dayofyear / 365)
    df["season_cos"]      = np.cos(2 * np.pi * ts.dt.dayofyear / 365)
    return df


# ============================================================================
# CO2 FEATURES — FIXED (reduced redundancy)
# ============================================================================

def add_co2_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "co2_ppm" not in df.columns:
        return df

    ts = df[TIMESTAMP_COL]

    # 1. Keeling Curve linear trend (~2.5 ppm/year globally)
    ref_year = ts.dt.year.min()
    df["co2_years_elapsed"] = (ts.dt.year - ref_year) + (ts.dt.dayofyear / 365.0)
    df["co2_trend_linear"]  = df["co2_years_elapsed"] * 2.5

    # 2. Detrended CO2 — this is the real signal the model needs to learn
    baseline_start      = df["co2_ppm"].iloc[:24].mean()
    df["co2_detrended"] = df["co2_ppm"] - (baseline_start + df["co2_trend_linear"])

    # 3. Diurnal plant respiration cycle
    df["co2_diurnal_sin"] = np.sin(2 * np.pi * (ts.dt.hour - 6) / 24)
    df["co2_diurnal_cos"] = np.cos(2 * np.pi * (ts.dt.hour - 6) / 24)

    # 4. Annual Keeling seasonal oscillation
    df["co2_seasonal_sin"] = np.sin(2 * np.pi * ts.dt.month / 12)
    df["co2_seasonal_cos"] = np.cos(2 * np.pi * ts.dt.month / 12)

    # 5. REDUCED rolling stats — one mean per timeframe only (no rollmax/rollmin/rollmom)
    #    These were creating 5x redundant features that all correlated > 0.997
    df["co2_trend_7d"]  = df["co2_ppm"].rolling(168, min_periods=1).mean()
    df["co2_trend_24h"] = df["co2_ppm"].rolling(24,  min_periods=1).mean()

    # 6. 30-day z-score anomaly — useful, kept
    co2_mean_30d = df["co2_ppm"].rolling(720, min_periods=1).mean()
    co2_std_30d  = df["co2_ppm"].rolling(720, min_periods=1).std().fillna(1.0)
    df["co2_zscore_30d"] = (df["co2_ppm"] - co2_mean_30d) / (co2_std_30d + 1e-9)

    # 7. REDUCED rate of change — keep 1h and 24h diff only
    #    168h diff was redundant with rolling mean features
    df["co2_diff_1h"]  = df["co2_ppm"].diff(1).fillna(0)
    df["co2_diff_24h"] = df["co2_ppm"].diff(24).fillna(0)

    # 8. Wet season interaction
    if "is_wet_season" in df.columns:
        df["co2_wetday_interact"] = df["is_wet_season"] * df["co2_diurnal_cos"]

    return df


# ============================================================================
# HUMIDITY FEATURES
# ============================================================================

def add_humidity_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "temperature_c" in df.columns and "humidity_percent" in df.columns:
        T  = df["temperature_c"]
        RH = df["humidity_percent"]

        a, b  = 17.27, 237.7
        gamma = (a * T / (b + T)) + np.log(RH / 100.0 + 1e-9)
        df["dew_point_c"] = (b * gamma) / (a - gamma)

        es = 6.112 * np.exp((17.67 * T) / (T + 243.5))
        df["vapor_pressure"]    = (RH / 100.0) * es
        df["absolute_humidity"] = (
            (6.112 * np.exp((17.67 * T) / (T + 243.5)) * RH * 2.1674)
            / (273.15 + T)
        )
    return df


# ============================================================================
# CROSS-TARGET FEATURES
# ============================================================================

def add_cross_target_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    has_co2  = "co2_ppm" in df.columns
    has_temp = "temperature_c" in df.columns
    has_hum  = "humidity_percent" in df.columns

    if has_temp and has_hum:
        df["temp_x_humidity"]   = df["temperature_c"] * df["humidity_percent"]
        df["heat_stress_index"] = (
            df["temperature_c"]
            + 0.33 * (df["humidity_percent"] / 100 * 6.105
            * np.exp(17.27 * df["temperature_c"] / (237.7 + df["temperature_c"])))
            - 4.0
        )

    if has_co2 and has_temp:
        df["co2_per_temp"] = df["co2_ppm"] / (df["temperature_c"] + 273.15)

    return df


# ============================================================================
# LAG FEATURES — FIXED (reduced from 9 lags to 3 key lags)
# ============================================================================

def add_lag_features(df: pd.DataFrame, target_cols: list = None) -> pd.DataFrame:
    if target_cols is None:
        target_cols = TARGET_COLS

    df = df.copy()

    # FIXED: was [1, 3, 6, 12, 24, 48, 72, 168, 336] — 9 lags per target = 27 lag cols
    # Top correlated features showed lags 1h, 3h, 6h, 12h all > 0.999 with CO2
    # meaning they're near-identical. Keep only 3 lags that capture distinct timescales:
    #   1h  = immediate past (most useful)
    #   24h = same hour yesterday (daily cycle)
    #   168h = same hour last week (weekly cycle)
    KEY_LAGS = [1, 24, 168]

    for col in target_cols:
        for lag in KEY_LAGS:
            df[f"{col}_lag_{lag}h"] = df[col].shift(lag)

    return df


# ============================================================================
# ROLLING FEATURES — FIXED (mean only, no max/min/mom variants)
# ============================================================================

def add_rolling_features(df: pd.DataFrame, target_cols: list = None) -> pd.DataFrame:
    if target_cols is None:
        target_cols = TARGET_COLS

    df = df.copy()

    # FIXED: was generating mean + std + min + max + momentum per window
    # That's 5 features x 5 windows x 3 targets = 75 rolling cols, mostly redundant
    # Now: mean + std only (std captures variance, mean captures level)
    # Windows: 24h (daily), 168h (weekly) — enough to capture CO2 patterns
    ROLLING_WINDOWS = [24, 168]

    for col in target_cols:
        for window in ROLLING_WINDOWS:
            roll = df[col].rolling(window=window, min_periods=1)
            df[f"{col}_rollmean_{window}h"] = roll.mean()
            df[f"{col}_rollstd_{window}h"]  = roll.std().fillna(0)

    return df


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def build_features(df: pd.DataFrame, drop_nan: bool = True) -> pd.DataFrame:
    logger.info(f"Starting feature engineering. Input shape: {df.shape}")

    df = df.copy()
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    df = add_time_features(df)
    df = add_manila_season_features(df)
    df = add_co2_features(df)
    df = add_humidity_features(df)
    df = add_cross_target_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    if drop_nan:
        before = len(df)
        df = df.dropna().reset_index(drop=True)
        logger.info(f"Dropped {before - len(df):,} NaN rows. Remaining: {len(df):,}")

    logger.info(f"Feature engineering complete. Output shape: {df.shape}")
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude = [TIMESTAMP_COL] + TARGET_COLS
    return [col for col in df.columns if col not in exclude]


def save_feature_cols(feature_cols: list):
    path = get_model_path("feature_cols")
    with open(path, "wb") as f:
        pickle.dump(feature_cols, f)
    logger.info(f"Saved {len(feature_cols)} feature cols to {path}")


def load_feature_cols() -> list:
    path = get_model_path("feature_cols")
    with open(path, "rb") as f:
        return pickle.load(f)