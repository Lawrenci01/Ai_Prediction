"""
Daily Prediction Scheduler
============================
Runs once per day (default: 1 AM UTC).
Supports single sensor or all active sensors.

Flow per sensor:
  1. Fetch last 14 days from sensor_data (MySQL)
  2. Rename columns → co2_ppm, humidity_percent, timestamp
  3. Feature engineering (build_features)
  4. LSTM predictions → co2_ppm, temperature_c, humidity_percent
  5. Save to daily_predictions + hourly_predictions
  6. Groq insight → save to prediction_insights

Usage:
    python -m app.scheduler.daily_scheduler --run-now
    python -m app.scheduler.daily_scheduler --sensor-id 1 --run-now
    python -m app.scheduler.daily_scheduler   # starts loop
"""

import asyncio
import logging
import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

load_dotenv()
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.database.connection import AsyncSessionLocal, check_connection
from app.database.repository import (
    fetch_recent_sensor_data,
    fetch_sensor_info,
    fetch_all_active_sensors,
    save_daily_prediction,
    save_insight,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

SCHEDULER_HOUR   = int(os.getenv("SCHEDULER_HOUR")   or "1")
SCHEDULER_MINUTE = int(os.getenv("SCHEDULER_MINUTE") or "0")
DATA_FETCH_HOURS = int(os.getenv("DATA_FETCH_HOURS") or "336")
LLM_BACKEND      = os.getenv("LLM_BACKEND", "groq")

# Model target names (what model outputs)
MODEL_TARGETS = ["co2_ppm", "temperature_c", "humidity_percent"]

# DB column names for saving (maps back to sensor_data column names)
TARGET_LABEL_MAP = {
    "co2_ppm":          "co2_density",
    "temperature_c":    "temperature_c",
    "humidity_percent": "humidity",
}

UNIT_MAP = {
    "co2_ppm":          "ppm",
    "temperature_c":    "°C",
    "humidity_percent": "%",
}


# ============================================================================
# SINGLE SENSOR JOB
# ============================================================================

async def run_for_sensor(sensor_id: int):
    """Run full prediction pipeline for one sensor."""
    logger.info(f"── Sensor {sensor_id} ──────────────────────────────────")

    # 1. Fetch IoT data
    async with AsyncSessionLocal() as db:
        df_raw     = await fetch_recent_sensor_data(db, hours=DATA_FETCH_HOURS, sensor_id=sensor_id)
        sensor_info = await fetch_sensor_info(db, sensor_id)

    if df_raw.empty or len(df_raw) < 170:
        logger.warning(f"Sensor {sensor_id}: not enough data ({len(df_raw)} rows). Need 170+. Skipping.")
        return False

    logger.info(f"Sensor {sensor_id}: {len(df_raw):,} rows | {df_raw['timestamp'].min()} → {df_raw['timestamp'].max()}")

    # 2. Feature engineering
    try:
        from app.pipeline.feature_engineer import build_features
        df_featured = build_features(df_raw, drop_nan=True)
    except Exception as e:
        logger.error(f"Sensor {sensor_id}: feature engineering failed: {e}")
        return False

    if len(df_featured) < 168:
        logger.error(f"Sensor {sensor_id}: only {len(df_featured)} rows after feature eng. Need 168+.")
        return False

    # 3. Predictions
    try:
        from app.inference import daily_forecast
        forecast = daily_forecast(df_raw, df_featured)
        prediction_date = datetime.strptime(forecast["prediction_date"], "%Y-%m-%d").date()
    except Exception as e:
        logger.error(f"Sensor {sensor_id}: prediction failed: {e}")
        return False

    # 4. Save predictions to DB
    async with AsyncSessionLocal() as db:
        for model_target in MODEL_TARGETS:
            target_data = forecast.get(model_target, {})
            if not target_data or target_data.get("mean") is None:
                logger.warning(f"Sensor {sensor_id}: no prediction for {model_target}")
                continue

            # Use DB-friendly label for storage
            db_target = TARGET_LABEL_MAP[model_target]

            summary = {
                "mean": target_data["mean"],
                "min":  target_data["min"],
                "max":  target_data["max"],
                "unit": UNIT_MAP[model_target],
            }
            hourly = target_data.get("hourly", [])

            try:
                await save_daily_prediction(db, prediction_date, sensor_id, db_target, summary, hourly)
                logger.info(f"Sensor {sensor_id}: saved {db_target} mean={summary['mean']} {summary['unit']}")
            except Exception as e:
                logger.error(f"Sensor {sensor_id}: DB save failed for {db_target}: {e}")

    # 5. Groq insight
    try:
        from app.llm_engine import generate_insight

        sensor    = {"sensor_id": sensor_info["sensor_name"], "barangay": sensor_info["barangay_name"]}
        predicted = {
            "co2_ppm":          forecast.get("co2_ppm",          {}).get("mean", 415.0),
            "temperature_c":    forecast.get("temperature_c",    {}).get("mean", 28.0),
            "humidity_percent": forecast.get("humidity_percent", {}).get("mean", 75.0),
        }
        peak_temp_hour = _find_peak_hour(forecast, "temperature_c")
        peak_co2_hour  = _find_peak_hour(forecast, "co2_ppm")

        insight_text = await generate_insight(
            sensor=sensor,
            current={},
            predicted=predicted,
            peak_temp_hour=peak_temp_hour,
            peak_co2_hour=peak_co2_hour,
        )

        async with AsyncSessionLocal() as db:
            await save_insight(
                db,
                prediction_date=prediction_date,
                sensor_id=sensor_id,
                barangay=sensor_info["barangay_name"],
                insight_text=insight_text,
                llm_backend=LLM_BACKEND,
            )
        logger.info(f"Sensor {sensor_id}: insight → {insight_text[:80]}...")

    except Exception as e:
        logger.error(f"Sensor {sensor_id}: insight failed: {e}")

    return True


def _find_peak_hour(forecast: dict, target: str) -> int:
    hourly = forecast.get(target, {}).get("hourly", [])
    if not hourly:
        return 13 if "temp" in target else 12
    peak = max(hourly, key=lambda h: h["value"])
    try:
        return datetime.fromisoformat(peak["timestamp"]).hour
    except Exception:
        return peak.get("hour", 13)


# ============================================================================
# MAIN JOB — all sensors or one
# ============================================================================

async def run_prediction_job(sensor_id: Optional[int] = None):
    logger.info("=" * 60)
    logger.info(f"🚀 Daily prediction job — {datetime.utcnow().isoformat()}")
    logger.info("=" * 60)

    if sensor_id:
        sensors = [sensor_id]
    else:
        async with AsyncSessionLocal() as db:
            sensors = await fetch_all_active_sensors(db)

    if not sensors:
        logger.warning("No active sensors found in last 24h.")
        return False

    logger.info(f"Running predictions for {len(sensors)} sensor(s): {sensors}")
    for sid in sensors:
        await run_for_sensor(sid)

    logger.info("✅ Daily prediction job complete.")
    return True


# ============================================================================
# SCHEDULER LOOP
# ============================================================================

async def scheduler_loop(sensor_id=None):
    logger.info(f"📅 Scheduler started — daily at {SCHEDULER_HOUR:02d}:{SCHEDULER_MINUTE:02d} UTC")

    if not await check_connection():
        logger.error("❌ Cannot connect to MySQL. Check .env")
        return

    while True:
        now      = datetime.utcnow()
        next_run = now.replace(hour=SCHEDULER_HOUR, minute=SCHEDULER_MINUTE, second=0, microsecond=0)
        if next_run <= now:
            next_run += timedelta(days=1)

        wait = (next_run - now).total_seconds()
        logger.info(f"⏰ Next run at {next_run.isoformat()} UTC ({wait/3600:.1f}h away)")
        await asyncio.sleep(wait)
        await run_prediction_job(sensor_id)


# ============================================================================
# ENTRYPOINT
# ============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-now",   action="store_true", help="Run immediately and exit")
    parser.add_argument("--sensor-id", type=int, default=None, help="Run for a specific sensor_id only")
    args = parser.parse_args()

    if args.run_now:
        asyncio.run(run_prediction_job(args.sensor_id))
    else:
        asyncio.run(scheduler_loop(args.sensor_id))
