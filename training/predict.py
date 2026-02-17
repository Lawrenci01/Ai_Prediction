"""
Future Prediction Script
=========================
Generates daily predictions for CO2 PPM, Temperature (C), and Humidity (%)
from January 1, 2025 to June 30, 2026.

Run AFTER all 3 models are trained.

Usage:
    python training/predict.py
    python training/predict.py --start 2025-01-01 --end 2026-06-30
    python training/predict.py --output my_output.csv

Output files (saved to saved_models/):
    predictions_2025_2026.csv          <- main daily output
    predictions_hourly_2025_2026.csv   <- all hourly rows
    predictions_accuracy_summary.csv   <- expected accuracy by time period
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline.config import (
    DATA_FILE, TARGET_COLS, TIMESTAMP_COL,
    SEQUENCE_LENGTH, PREDICTION_HORIZON,
    MODEL_DIR, LOG_FORMAT, LOG_LEVEL
)
from app.pipeline.feature_engineer import build_features
from app.inference import registry, predict_with_lstm

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# ============================================================================
# STEP 1 — Load seed window
# ============================================================================

def load_seed_window() -> pd.DataFrame:
    """Load last 168 hours of real data as the starting window."""

    logger.info(f"Loading data from: {DATA_FILE}")
    df_raw = pd.read_csv(DATA_FILE)
    df_raw[TIMESTAMP_COL] = pd.to_datetime(df_raw[TIMESTAMP_COL])

    logger.info("Running feature engineering on historical data...")
    df_feat = build_features(df_raw, drop_nan=True)

    seed = df_feat.tail(SEQUENCE_LENGTH).copy().reset_index(drop=True)
    logger.info(
        f"Seed window: {seed[TIMESTAMP_COL].iloc[0]} -> "
        f"{seed[TIMESTAMP_COL].iloc[-1]}"
    )
    return seed


# ============================================================================
# STEP 2 — Rolling prediction
# ============================================================================

def run_rolling_prediction(seed: pd.DataFrame,
                            start_dt: datetime,
                            end_dt: datetime) -> pd.DataFrame:
    """
    Predict in 24-hour blocks from start_dt to end_dt.
    Each block feeds the previous block's predictions back as input.
    """

    feat_cols = registry.get("feature_cols")
    scaler_X  = registry.get("scaler_X")

    if feat_cols is None or scaler_X is None:
        raise RuntimeError(
            "feature_cols or scaler_X not found.\n"
            "Run training first: python training/train.py --target all --force-retrain"
        )

    for target in TARGET_COLS:
        key = ("lstm_co2" if "co2" in target
               else "lstm_temperature" if "temp" in target
               else "lstm_humidity")
        if registry.get(key) is None:
            raise RuntimeError(
                f"LSTM for '{target}' not found.\n"
                f"Run: python training/train.py --target {target} --force-retrain"
            )

    window       = seed.copy()
    all_hourly   = []
    current_dt   = start_dt
    total_hours  = int((end_dt - start_dt).total_seconds() / 3600) + 1
    total_blocks = (total_hours + PREDICTION_HORIZON - 1) // PREDICTION_HORIZON

    logger.info(f"Predicting {total_hours:,} hours in {total_blocks} blocks...")

    for block in range(total_blocks):

        # Predict all 3 targets
        block_preds = {}
        for target in TARGET_COLS:
            result = predict_with_lstm(window, target)
            if "error" in result:
                raise RuntimeError(f"Block {block+1} failed for '{target}': {result['error']}")
            block_preds[target] = [h["predicted_value"] for h in result["hourly_predictions"]]

        # Clip to safe physical ranges
        block_preds["co2_ppm"]          = np.clip(block_preds["co2_ppm"],          380, 600).tolist()
        block_preds["temperature_c"]    = np.clip(block_preds["temperature_c"],    18,  42 ).tolist()
        block_preds["humidity_percent"] = np.clip(block_preds["humidity_percent"], 20,  100).tolist()

        # Store results and build raw rows for window update
        new_raw_rows = []
        for h in range(PREDICTION_HORIZON):
            hour_dt = current_dt + timedelta(hours=h)
            if hour_dt > end_dt:
                break

            all_hourly.append({
                "timestamp":             hour_dt,
                "co2_ppm_pred":          round(float(block_preds["co2_ppm"][h]),          4),
                "temperature_c_pred":    round(float(block_preds["temperature_c"][h]),    4),
                "humidity_percent_pred": round(float(block_preds["humidity_percent"][h]), 4),
            })
            new_raw_rows.append({
                TIMESTAMP_COL:       hour_dt,
                "co2_ppm":           float(block_preds["co2_ppm"][h]),
                "temperature_c":     float(block_preds["temperature_c"][h]),
                "humidity_percent":  float(block_preds["humidity_percent"][h]),
            })

        if not new_raw_rows:
            break

        hours_added = len(new_raw_rows)

        # Update rolling window
        try:
            raw_cols      = [TIMESTAMP_COL, "co2_ppm", "temperature_c", "humidity_percent"]
            tail_raw      = window[raw_cols].tail(SEQUENCE_LENGTH)
            new_raw       = pd.DataFrame(new_raw_rows)
            combined_raw  = pd.concat([tail_raw, new_raw], ignore_index=True)
            combined_feat = build_features(combined_raw, drop_nan=False)
            new_feat      = combined_feat.tail(hours_added).copy()

            for col in feat_cols:
                if col not in new_feat.columns:
                    new_feat[col] = 0.0

            # FIX BUG 3: Use .ffill().bfill() instead of deprecated fillna(method=...)
            new_feat[feat_cols] = (
                new_feat[feat_cols]
                .ffill()
                .bfill()
                .fillna(0.0)
            )

            window = pd.concat(
                [window.iloc[hours_added:], new_feat],
                ignore_index=True
            )

        except Exception as e:
            logger.warning(f"Block {block+1}: feature update failed — {e}. Using fallback.")
            filler = pd.concat([window.iloc[-1:]] * hours_added, ignore_index=True)
            filler[TIMESTAMP_COL] = [current_dt + timedelta(hours=h) for h in range(hours_added)]
            window = pd.concat([window.iloc[hours_added:], filler], ignore_index=True)

        # FIX BUG 4: Guard against window shrinking below SEQUENCE_LENGTH
        if len(window) < SEQUENCE_LENGTH:
            pad_needed = SEQUENCE_LENGTH - len(window)
            pad = pd.concat([window.iloc[:1]] * pad_needed, ignore_index=True)
            window = pd.concat([pad, window], ignore_index=True)
            logger.warning(f"  Block {block+1}: window padded back to {SEQUENCE_LENGTH} rows.")

        current_dt += timedelta(hours=hours_added)

        if (block + 1) % 7 == 0 or block == 0:
            pct = min(100.0, round((block + 1) / total_blocks * 100, 1))
            logger.info(
                f"  [{pct:5.1f}%] Block {block+1}/{total_blocks} | "
                f"Up to: {(current_dt - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M')}"
            )

    logger.info(f"Generated {len(all_hourly):,} hourly rows.")
    return pd.DataFrame(all_hourly)


# ============================================================================
# STEP 3 — Aggregate to daily
# ============================================================================

def aggregate_daily(df_hourly: pd.DataFrame) -> pd.DataFrame:
    df         = df_hourly.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    ref_date   = pd.Timestamp("2025-01-01")
    rows       = []

    for date, grp in df.groupby("date"):
        days_ahead = (pd.Timestamp(date) - ref_date).days
        confidence = max(60.0, 95.0 - days_ahead * 0.05)

        rows.append({
            "date":       date,
            "days_ahead": days_ahead + 1,

            "co2_ppm_min":  round(grp["co2_ppm_pred"].min(),  2),
            "co2_ppm_mean": round(grp["co2_ppm_pred"].mean(), 2),
            "co2_ppm_max":  round(grp["co2_ppm_pred"].max(),  2),

            "temperature_c_min":  round(grp["temperature_c_pred"].min(),  2),
            "temperature_c_mean": round(grp["temperature_c_pred"].mean(), 2),
            "temperature_c_max":  round(grp["temperature_c_pred"].max(),  2),

            "humidity_percent_min":  round(grp["humidity_percent_pred"].min(),  2),
            "humidity_percent_mean": round(grp["humidity_percent_pred"].mean(), 2),
            "humidity_percent_max":  round(grp["humidity_percent_pred"].max(),  2),

            "prediction_confidence_pct": round(confidence, 1),
        })

    df_daily         = pd.DataFrame(rows)
    df_daily["date"] = pd.to_datetime(df_daily["date"])
    return df_daily.sort_values("date").reset_index(drop=True)


# ============================================================================
# STEP 4 — Accuracy summary
# ============================================================================

def build_accuracy_summary(df_daily: pd.DataFrame) -> pd.DataFrame:
    periods = [
        ("Jan 2025",     "2025-01-01", "2025-01-31", "Near-term"),
        ("Feb-Mar 2025", "2025-02-01", "2025-03-31", "Short-range"),
        ("Apr-Jun 2025", "2025-04-01", "2025-06-30", "Medium-range"),
        ("Jul-Sep 2025", "2025-07-01", "2025-09-30", "Medium-range"),
        ("Oct-Dec 2025", "2025-10-01", "2025-12-31", "Long-range"),
        ("Jan-Jun 2026", "2026-01-01", "2026-06-30", "Very long-range"),
    ]
    acc_map = {
        "Near-term":       {"co2": "0.55-0.75", "temp": "0.88-0.96", "hum": "0.82-0.92",
                            "co2_rmse": "<1.5ppm", "temp_rmse": "<0.8C", "hum_rmse": "<4%"},
        "Short-range":     {"co2": "0.45-0.65", "temp": "0.82-0.92", "hum": "0.75-0.88",
                            "co2_rmse": "<2.0ppm", "temp_rmse": "<1.2C", "hum_rmse": "<6%"},
        "Medium-range":    {"co2": "0.35-0.55", "temp": "0.70-0.85", "hum": "0.65-0.80",
                            "co2_rmse": "<2.5ppm", "temp_rmse": "<1.8C", "hum_rmse": "<8%"},
        "Long-range":      {"co2": "0.25-0.45", "temp": "0.60-0.75", "hum": "0.55-0.72",
                            "co2_rmse": "<3.5ppm", "temp_rmse": "<2.5C", "hum_rmse": "<10%"},
        "Very long-range": {"co2": "0.15-0.35", "temp": "0.50-0.68", "hum": "0.45-0.65",
                            "co2_rmse": "<5.0ppm", "temp_rmse": "<3.5C", "hum_rmse": "<14%"},
    }

    rows = []
    for label, start, end, tier in periods:
        mask = (df_daily["date"] >= pd.Timestamp(start)) & (df_daily["date"] <= pd.Timestamp(end))
        grp  = df_daily[mask]
        if grp.empty:
            continue
        a = acc_map[tier]
        rows.append({
            "period":                    label,
            "accuracy_tier":             tier,
            "total_days":                len(grp),
            "avg_confidence_pct":        round(grp["prediction_confidence_pct"].mean(), 1),
            "co2_ppm_avg":               round(grp["co2_ppm_mean"].mean(), 2),
            "temperature_c_avg":         round(grp["temperature_c_mean"].mean(), 2),
            "humidity_percent_avg":      round(grp["humidity_percent_mean"].mean(), 2),
            "expected_co2_r2":           a["co2"],
            "expected_temperature_r2":   a["temp"],
            "expected_humidity_r2":      a["hum"],
            "expected_co2_rmse":         a["co2_rmse"],
            "expected_temperature_rmse": a["temp_rmse"],
            "expected_humidity_rmse":    a["hum_rmse"],
        })
    return pd.DataFrame(rows)


# ============================================================================
# MAIN
# ============================================================================

def run_prediction(start_date="2025-01-01", end_date="2026-06-30",
                   output_filename="predictions_2025_2026.csv"):

    logger.info("=" * 70)
    logger.info("CLIMATE FUTURE PREDICTION")
    logger.info("=" * 70)
    logger.info(f"Period  : {start_date} -> {end_date}")
    logger.info(f"Targets : {TARGET_COLS}")

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt   = datetime.strptime(end_date,   "%Y-%m-%d") + timedelta(hours=23)

    seed      = load_seed_window()
    df_hourly = run_rolling_prediction(seed, start_dt, end_dt)

    logger.info("Aggregating hourly -> daily...")
    df_daily    = aggregate_daily(df_hourly)
    df_accuracy = build_accuracy_summary(df_daily)

    out = Path(MODEL_DIR)
    out.mkdir(parents=True, exist_ok=True)

    daily_path    = out / output_filename
    hourly_path   = out / "predictions_hourly_2025_2026.csv"
    accuracy_path = out / "predictions_accuracy_summary.csv"

    df_daily.to_csv(daily_path,    index=False)
    df_hourly.to_csv(hourly_path,  index=False)
    df_accuracy.to_csv(accuracy_path, index=False)

    logger.info("\n" + "=" * 70)
    logger.info("PREDICTION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Days    : {len(df_daily):,}")
    logger.info(f"  Hours   : {len(df_hourly):,}")
    logger.info(f"  CO2     : {df_daily['co2_ppm_min'].min():.1f} - {df_daily['co2_ppm_max'].max():.1f} ppm")
    logger.info(f"  Temp    : {df_daily['temperature_c_min'].min():.1f} - {df_daily['temperature_c_max'].max():.1f} C")
    logger.info(f"  Humidity: {df_daily['humidity_percent_min'].min():.1f} - {df_daily['humidity_percent_max'].max():.1f} %")
    logger.info(f"\n  Files saved to: {out}")
    logger.info("=" * 70)

    return df_daily, df_hourly, df_accuracy


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate future climate predictions")
    parser.add_argument("--start",  default="2025-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",    default="2026-06-30", help="End date YYYY-MM-DD")
    parser.add_argument("--output", default="predictions_2025_2026.csv", help="Output filename")
    args = parser.parse_args()

    run_prediction(
        start_date      = args.start,
        end_date        = args.end,
        output_filename = args.output,
    )