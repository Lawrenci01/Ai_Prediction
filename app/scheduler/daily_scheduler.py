import asyncio
import logging
import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

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

MODEL_TARGETS = ["co2_ppm", "temperature_c", "humidity_percent"]

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


async def run_for_sensor(sensor_id: int):
    logger.info(f"Starting prediction for sensor {sensor_id}")

    async with AsyncSessionLocal() as db:
        df_raw      = await fetch_recent_sensor_data(db, hours=DATA_FETCH_HOURS, sensor_id=sensor_id)
        sensor_info = await fetch_sensor_info(db, sensor_id)

    if df_raw.empty or len(df_raw) < 170:
        logger.warning(f"Sensor {sensor_id}: insufficient data ({len(df_raw)} rows, need 170+). Skipping.")
        return False

    logger.info(f"Sensor {sensor_id}: {len(df_raw):,} rows | {df_raw['timestamp'].min()} → {df_raw['timestamp'].max()}")

    try:
        from app.pipeline.feature_engineer import build_features
        df_featured = build_features(df_raw, drop_nan=True)
    except Exception as e:
        logger.error(f"Sensor {sensor_id}: feature engineering failed: {e}", exc_info=True)
        return False

    if len(df_featured) < 168:
        logger.error(f"Sensor {sensor_id}: only {len(df_featured)} rows after feature engineering, need 168+.")
        return False

    try:
        from app.inference import daily_forecast
        forecast        = daily_forecast(df_raw, df_featured)
        prediction_date = datetime.strptime(forecast["prediction_date"], "%Y-%m-%d").date()
    except Exception as e:
        logger.error(f"Sensor {sensor_id}: prediction failed: {e}", exc_info=True)
        return False

    async with AsyncSessionLocal() as db:
        for model_target in MODEL_TARGETS:
            target_data = forecast.get(model_target, {})
            if not target_data or target_data.get("mean") is None:
                logger.warning(f"Sensor {sensor_id}: no prediction for {model_target}, skipping.")
                continue

            db_target = TARGET_LABEL_MAP[model_target]
            summary   = {
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
                logger.error(f"Sensor {sensor_id}: DB save failed for {db_target}: {e}", exc_info=True)

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
        logger.info(f"Sensor {sensor_id}: insight saved — {insight_text[:80]}...")

    except Exception as e:
        logger.error(f"Sensor {sensor_id}: insight generation failed: {e}", exc_info=True)

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


async def run_prediction_job(sensor_id: Optional[int] = None):
    logger.info(f"Daily prediction job started — {datetime.utcnow().isoformat()}")

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

    logger.info("Daily prediction job complete.")
    return True


async def scheduler_loop(sensor_id=None):
    logger.info(f"Scheduler started — daily at {SCHEDULER_HOUR:02d}:{SCHEDULER_MINUTE:02d} UTC")

    if not await check_connection():
        logger.error("Cannot connect to MySQL — check .env DB credentials.")
        return

    while True:
        now      = datetime.utcnow()
        next_run = now.replace(hour=SCHEDULER_HOUR, minute=SCHEDULER_MINUTE, second=0, microsecond=0)
        if next_run <= now:
            next_run += timedelta(days=1)

        wait = (next_run - now).total_seconds()
        logger.info(f"Next run at {next_run.isoformat()} UTC ({wait/3600:.1f}h away)")
        await asyncio.sleep(wait)
        await run_prediction_job(sensor_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-now",   action="store_true")
    parser.add_argument("--sensor-id", type=int, default=None)
    args = parser.parse_args()

    if args.run_now:
        asyncio.run(run_prediction_job(args.sensor_id))
    else:
        asyncio.run(scheduler_loop(args.sensor_id))