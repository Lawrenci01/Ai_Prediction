"""
Feature Engineering Pipeline
==============================
Transforms raw timestamp/co2/temperature/humidity data into
rich ML features for LSTM training and inference.
"""

import pandas as pd
import numpy as np
import logging
import pickle
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from app.pipeline.config import (
    LAG_HOURS, ROLLING_WINDOWS, TARGET_COLS, TIMESTAMP_COL,
    USE_LAG_FEATURES, USE_ROLLING_FEATURES, USE_CYCLICAL_ENCODING,
    WET_SEASON_MONTHS, get_model_path
)

logger = logging.getLogger(__name__)


# ============================================================================
# TIME FEATURES
# ============================================================================

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = df[TIMESTAMP_COL]

    df["year"]       = ts.dt.year
    df["month"]      = ts.dt.month
    df["day"]        = ts.dt.day
    df["hour"]       = ts.dt.hour
    df["dayofweek"]  = ts.dt.dayofweek
    df["dayofyear"]  = ts.dt.dayofyear
    df["quarter"]    = ts.dt.quarter
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    if USE_CYCLICAL_ENCODING:
        df["hour_sin"]      = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"]      = np.cos(2 * np.pi * df["hour"] / 24)
        df["month_sin"]     = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"]     = np.cos(2 * np.pi * df["month"] / 12)
        df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
        df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)
        df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
        df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    df["hours_elapsed"] = (ts - ts.min()).dt.total_seconds() / 3600
    return df


# ============================================================================
# MANILA SEASON FEATURES
# ============================================================================

def add_manila_season_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = df[TIMESTAMP_COL]

    df["is_wet_season"]        = ts.dt.month.isin(WET_SEASON_MONTHS).astype(int)
    df["season_progress"]      = ts.dt.dayofyear / 365.0
    df["days_since_wet_start"] = (ts.dt.dayofyear - 152) % 365
    df["season_sin"]           = np.sin(2 * np.pi * ts.dt.dayofyear / 365)
    df["season_cos"]           = np.cos(2 * np.pi * ts.dt.dayofyear / 365)
    return df


# ============================================================================
# CO2 FEATURES  (fixed — removed 8760h lag, added Keeling Curve features)
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

    # 2. Polynomial trend (growth slightly accelerating)
    df["co2_trend_squared"] = df["co2_years_elapsed"] ** 2

    # 3. Detrended CO2 (residuals from expected global trend)
    baseline_start  = df["co2_ppm"].iloc[:24].mean()
    df["co2_detrended"] = df["co2_ppm"] - (baseline_start + df["co2_trend_linear"])

    # 4. Diurnal plant respiration cycle (peaks 6am, troughs 2pm)
    df["co2_diurnal_sin"] = np.sin(2 * np.pi * (df["hour"] - 6) / 24)
    df["co2_diurnal_cos"] = np.cos(2 * np.pi * (df["hour"] - 6) / 24)

    # 5. Annual Keeling seasonal oscillation
    df["co2_seasonal_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["co2_seasonal_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # 6. Short-window rolling stats
    df["co2_trend_30d"] = df["co2_ppm"].rolling(720,  min_periods=1).mean()
    df["co2_trend_7d"]  = df["co2_ppm"].rolling(168,  min_periods=1).mean()
    df["co2_trend_24h"] = df["co2_ppm"].rolling(24,   min_periods=1).mean()

    # 7. 30-day z-score anomaly
    co2_std_30d = df["co2_ppm"].rolling(720, min_periods=1).std().fillna(1.0)
    df["co2_zscore_30d"] = (df["co2_ppm"] - df["co2_trend_30d"]) / (co2_std_30d + 1e-9)

    # 8. Rate of change
    df["co2_diff_1h"]  = df["co2_ppm"].diff(1).fillna(0)
    df["co2_diff_24h"] = df["co2_ppm"].diff(24).fillna(0)
    df["co2_diff_7d"]  = df["co2_ppm"].diff(168).fillna(0)

    # 9. Wet season x diurnal interaction
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
        df["temp_x_humidity"]     = df["temperature_c"] * df["humidity_percent"]
        df["temp_humidity_ratio"] = df["temperature_c"] / (df["humidity_percent"] + 1e-9)
        df["heat_stress_index"]   = (
            df["temperature_c"]
            + 0.33 * (df["humidity_percent"] / 100 * 6.105
            * np.exp(17.27 * df["temperature_c"] / (237.7 + df["temperature_c"])))
            - 4.0
        )

    if has_co2 and has_temp:
        df["co2_per_temp"] = df["co2_ppm"] / (df["temperature_c"] + 273.15)

    if has_co2 and has_hum:
        df["co2_humidity_ratio"] = df["co2_ppm"] / (df["humidity_percent"] + 1e-9)

    if has_co2 and has_temp and has_hum:
        df["co2_lag1_x_temp"] = df["co2_ppm"].shift(1) * df["temperature_c"]
        df["temp_lag1_x_hum"] = df["temperature_c"].shift(1) * df["humidity_percent"]

    return df


# ============================================================================
# LAG FEATURES
# ============================================================================

def add_lag_features(df: pd.DataFrame, target_cols: list = None) -> pd.DataFrame:
    if not USE_LAG_FEATURES:
        return df
    if target_cols is None:
        target_cols = TARGET_COLS

    df = df.copy()
    for col in target_cols:
        for lag in LAG_HOURS:
            df[f"{col}_lag_{lag}h"] = df[col].shift(lag)
    return df


# ============================================================================
# ROLLING FEATURES
# ============================================================================

def add_rolling_features(df: pd.DataFrame, target_cols: list = None) -> pd.DataFrame:
    if not USE_ROLLING_FEATURES:
        return df
    if target_cols is None:
        target_cols = TARGET_COLS

    df = df.copy()
    for col in target_cols:
        for window in ROLLING_WINDOWS:
            roll = df[col].rolling(window=window, min_periods=1)
            df[f"{col}_rollmean_{window}h"] = roll.mean()
            df[f"{col}_rollstd_{window}h"]  = roll.std().fillna(0)
            df[f"{col}_rollmin_{window}h"]  = roll.min()
            df[f"{col}_rollmax_{window}h"]  = roll.max()
            df[f"{col}_rollmom_{window}h"]  = df[col] - roll.mean()
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
