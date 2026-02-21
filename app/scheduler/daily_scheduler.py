import asyncio
import logging
import os
import argparse
from datetime import datetime, timedelta, timezone
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from app.database.connection import AsyncSessionLocal, check_connection
from app.database.repository import (
    fetch_recent_sensor_data,
    fetch_sensor_full_info,
    fetch_all_active_sensors,
    save_daily_prediction,
    save_hourly_prediction,
    save_insight,
)
from app.llm_engine import generate_hourly_safety_insight, generate_insight

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

os.makedirs("logs", exist_ok=True)
_log_file = f"logs/prediction_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
_fh = logging.FileHandler(_log_file, encoding="utf-8")
_fh.setFormatter(logging.Formatter(LOG_FORMAT))
logging.getLogger().addHandler(_fh)

SCHEDULER_HOUR   = int(os.getenv("SCHEDULER_HOUR",   "1"))
SCHEDULER_MINUTE = int(os.getenv("SCHEDULER_MINUTE", "0"))
DATA_FETCH_HOURS = int(os.getenv("DATA_FETCH_HOURS", "336"))
LLM_BACKEND      = os.getenv("LLM_BACKEND", "groq")


def _peak_hour(hourly: list, key: str) -> int:
    if not hourly:
        return 13
    peak = max(hourly, key=lambda h: h.get(key) or 0)
    try:
        return datetime.fromisoformat(peak["timestamp"]).hour
    except Exception:
        return 13


def _summary(forecast: dict, target: str) -> dict:
    d = forecast.get(target, {})
    return {"mean": d.get("mean"), "min": d.get("min"), "max": d.get("max")}


def _merge_hourly(forecast: dict) -> list:
    co2_rows = forecast.get("co2_ppm",          {}).get("hourly", [])
    temp_map = {r["timestamp"]: r for r in forecast.get("temperature_c",    {}).get("hourly", [])}
    hum_map  = {r["timestamp"]: r for r in forecast.get("humidity_percent", {}).get("hourly", [])}

    rows = []
    for r in co2_rows:
        ts = r["timestamp"]
        t  = temp_map.get(ts, {})
        h  = hum_map.get(ts, {})
        rows.append({
            "timestamp":        ts,
            "co2_ppm":          r.get("value") or r.get("co2_ppm"),
            "temperature_c":    t.get("value") or t.get("temperature_c"),
            "humidity_percent": h.get("value") or h.get("humidity_percent"),
        })
    return rows


async def run_for_sensor(sensor_id: int) -> bool:
    async with AsyncSessionLocal() as db:
        df_raw      = await fetch_recent_sensor_data(db, hours=DATA_FETCH_HOURS, sensor_id=sensor_id)
        sensor_info = await fetch_sensor_full_info(db, sensor_id)

    if df_raw.empty or len(df_raw) < 170:
        logger.warning(f"Sensor {sensor_id}: insufficient data ({len(df_raw)} rows). Skipping.")
        return False

    establishment_name = sensor_info.get("establishment_name") or f"Sensor Node {sensor_id}"
    establishment_type = sensor_info.get("establishment_type") or "Monitoring Station"
    barangay_name      = sensor_info.get("barangay_name")      or "Naga City"
    sensor_name        = sensor_info.get("sensor_name")        or f"NODE-{sensor_id:02d}"

    try:
        from app.pipeline.feature_engineer import build_features
        df_featured = build_features(df_raw, drop_nan=True)
    except Exception as e:
        logger.error(f"Sensor {sensor_id}: feature engineering failed: {e}", exc_info=True)
        return False

    if len(df_featured) < 168:
        logger.error(f"Sensor {sensor_id}: only {len(df_featured)} rows after features. Skipping.")
        return False

    try:
        from app.inference import daily_forecast
        forecast        = daily_forecast(df_raw, df_featured)
        prediction_date = datetime.strptime(forecast["prediction_date"], "%Y-%m-%d").date()
    except Exception as e:
        logger.error(f"Sensor {sensor_id}: inference failed: {e}", exc_info=True)
        return False

    co2_sum  = _summary(forecast, "co2_ppm")
    temp_sum = _summary(forecast, "temperature_c")
    hum_sum  = _summary(forecast, "humidity_percent")

    async with AsyncSessionLocal() as db:
        try:
            daily_row = await save_daily_prediction(
                db                 = db,
                prediction_date    = prediction_date,
                sensor_id          = sensor_id,
                establishment_name = establishment_name,
                establishment_type = establishment_type,
                barangay_name      = barangay_name,
                co2                = co2_sum,
                temp               = temp_sum,
                humidity           = hum_sum,
            )
            logger.info(f"Sensor {sensor_id}: daily saved (id={daily_row.id})")
        except Exception as e:
            logger.error(f"Sensor {sensor_id}: daily save failed: {e}", exc_info=True)
            return False

    hourly_rows = _merge_hourly(forecast)
    if not hourly_rows:
        logger.warning(f"Sensor {sensor_id}: no hourly rows in forecast.")
        return False

    logger.info(f"Sensor {sensor_id}: processing {len(hourly_rows)} hourly predictions...")

    date_str = prediction_date.strftime("%Y-%m-%d")
    for row in hourly_rows:
        try:
            ts_dt    = datetime.fromisoformat(row["timestamp"])
            hour     = ts_dt.hour
            co2      = float(row["co2_ppm"]          or 415.0)
            temp     = float(row["temperature_c"]    or 28.0)
            humidity = float(row["humidity_percent"] or 75.0)
        except Exception as e:
            logger.warning(f"Sensor {sensor_id}: bad hourly row {row}: {e}")
            continue

        try:
            safe_status, insight_text = await generate_hourly_safety_insight(
                establishment_name = establishment_name,
                establishment_type = establishment_type,
                barangay_name      = barangay_name,
                hour               = hour,
                prediction_date    = date_str,
                co2_ppm            = co2,
                temperature_c      = temp,
                humidity_percent   = humidity,
            )
        except Exception as e:
            logger.error(f"Sensor {sensor_id} {hour:02d}:00 LLM failed: {e}")
            safe_status  = "CAUTION"
            insight_text = f"CO2 {co2:.0f}ppm, temp {temp:.1f}C, humidity {humidity:.0f}%."

        async with AsyncSessionLocal() as db:
            try:
                await save_hourly_prediction(
                    db                 = db,
                    daily_id           = daily_row.id,
                    sensor_id          = sensor_id,
                    establishment_name = establishment_name,
                    establishment_type = establishment_type,
                    barangay_name      = barangay_name,
                    hour               = hour,
                    timestamp          = ts_dt,
                    co2_ppm            = co2,
                    temperature_c      = temp,
                    humidity_percent   = humidity,
                    safe_status        = safe_status,
                    insight_text       = insight_text,
                    llm_backend        = LLM_BACKEND,
                )
                logger.info(f"Sensor {sensor_id} {hour:02d}:00 -> {safe_status}")
            except Exception as e:
                logger.error(f"Sensor {sensor_id} {hour:02d}:00 save failed: {e}", exc_info=True)

    try:
        insight_text = await generate_insight(
            sensor         = {"sensor_id": sensor_name, "barangay": barangay_name},
            current        = {},
            predicted      = {
                "co2_ppm":          co2_sum.get("mean", 415.0),
                "temperature_c":    temp_sum.get("mean", 28.0),
                "humidity_percent": hum_sum.get("mean", 75.0),
            },
            peak_temp_hour = _peak_hour(hourly_rows, "temperature_c"),
            peak_co2_hour  = _peak_hour(hourly_rows, "co2_ppm"),
        )
        async with AsyncSessionLocal() as db:
            await save_insight(
                db              = db,
                prediction_date = prediction_date,
                sensor_id       = sensor_id,
                barangay        = barangay_name,
                insight_text    = insight_text,
                llm_backend     = LLM_BACKEND,
            )
        logger.info(f"Sensor {sensor_id}: insight saved")
    except Exception as e:
        logger.error(f"Sensor {sensor_id}: insight failed: {e}", exc_info=True)

    return True


async def run_prediction_job(sensor_id: Optional[int] = None) -> bool:
    logger.info(f"Daily prediction job started - {datetime.now(timezone.utc).isoformat()}")

    sensors = [sensor_id] if sensor_id else []
    if not sensors:
        async with AsyncSessionLocal() as db:
            sensors = await fetch_all_active_sensors(db)

    if not sensors:
        logger.warning("No active sensors found.")
        return False

    logger.info(f"Running predictions for {len(sensors)} sensor(s): {sensors}")
    for sid in sensors:
        await run_for_sensor(sid)

    logger.info("Daily prediction job complete.")
    return True


async def scheduler_loop(sensor_id: Optional[int] = None):
    logger.info(f"Scheduler started - daily at {SCHEDULER_HOUR:02d}:{SCHEDULER_MINUTE:02d} UTC")

    if not await check_connection():
        logger.error("Cannot connect to database.")
        return

    while True:
        now      = datetime.now(timezone.utc)
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