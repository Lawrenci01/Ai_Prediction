"""
Context Builder
===============
Builds structured context dicts from recent sensor readings.
Used to pass current + historical data to the insight engine.
"""

import pandas as pd
import numpy as np
from datetime import datetime

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))


def build_current_readings(df: pd.DataFrame) -> dict:
    """
    Extract the most recent readings as a clean dict.

    Parameters
    ----------
    df : DataFrame with timestamp, co2_ppm, temperature_c, humidity_percent

    Returns
    -------
    dict — { co2_ppm, temperature_c, humidity_percent }
    """
    latest = df.sort_values("timestamp").iloc[-1]
    return {
        "co2_ppm":          round(float(latest["co2_ppm"]), 2),
        "temperature_c":    round(float(latest["temperature_c"]), 2),
        "humidity_percent": round(float(latest["humidity_percent"]), 2),
    }


def build_historical_stats(df: pd.DataFrame, hours: int = 24) -> dict:
    """
    Build summary statistics from the last N hours of data.

    Parameters
    ----------
    df    : DataFrame with climate columns
    hours : how many hours to look back

    Returns
    -------
    dict of stats per variable
    """
    df = df.sort_values("timestamp").tail(hours)

    stats = {}
    for col in ["co2_ppm", "temperature_c", "humidity_percent"]:
        if col in df.columns:
            stats[col] = {
                "mean":  round(float(df[col].mean()), 2),
                "max":   round(float(df[col].max()), 2),
                "min":   round(float(df[col].min()), 2),
                "range": round(float(df[col].max() - df[col].min()), 2),
                "trend": "rising" if df[col].iloc[-1] > df[col].mean() else "falling",
            }

    return stats


def build_forecast_series(temp_fc: list,
                           hum_fc: list,
                           co2_fc: list) -> list:
    """
    Merge three separate forecast lists into one unified series.

    Parameters
    ----------
    temp_fc : list of {hour, predicted_value} for temperature
    hum_fc  : list of {hour, predicted_value} for humidity
    co2_fc  : list of {hour, predicted_value} for CO2

    Returns
    -------
    list of {hour, temperature_c, humidity_percent, co2_ppm}
    """
    series = []
    for i in range(min(len(temp_fc), len(hum_fc), len(co2_fc))):
        series.append({
            "hour":             temp_fc[i].get("hour", i + 1),
            "timestamp":        temp_fc[i].get("timestamp", ""),
            "temperature_c":    temp_fc[i].get("predicted_value", 0.0),
            "humidity_percent": hum_fc[i].get("predicted_value", 0.0),
            "co2_ppm":          co2_fc[i].get("predicted_value", 0.0),
        })
    return series
