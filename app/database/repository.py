"""
Database Repository
====================
All DB read/write operations.
Updated to match exact MySQL schema:
  sensor_data table: co2_density, humidity, recorded_at
  sensor table: sensor_id (INT), sensor_name, barangay_id
"""

import logging
from datetime import datetime, date, timedelta
from typing import Optional
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, and_

from app.database.models import (
    SensorData, Sensor, Barangay,
    DailyPrediction, HourlyPrediction, PredictionInsight
)

logger = logging.getLogger(__name__)


# ============================================================================
# SENSOR DATA — read IoT data and remap columns for model
# ============================================================================

async def fetch_recent_sensor_data(
    db: AsyncSession,
    hours: int = 336,
    sensor_id: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fetch recent rows from sensor_data.
    Renames columns to match feature_engineer.py:
        recorded_at  → timestamp
        co2_density  → co2_ppm
        humidity     → humidity_percent
    """
    since = datetime.utcnow() - timedelta(hours=hours)

    query = (
        select(
            SensorData.recorded_at,
            SensorData.co2_density,
            SensorData.temperature_c,
            SensorData.humidity,
            SensorData.sensor_id,
        )
        .where(SensorData.recorded_at >= since)
        .order_by(SensorData.recorded_at.asc())
    )
    if sensor_id:
        query = query.where(SensorData.sensor_id == sensor_id)

    result = await db.execute(query)
    rows = result.fetchall()

    if not rows:
        logger.warning(f"No sensor_data found in last {hours}h")
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        "recorded_at", "co2_density", "temperature_c", "humidity", "sensor_id"
    ])

    # Rename to match model feature names
    df = df.rename(columns={
        "recorded_at": "timestamp",
        "co2_density": "co2_ppm",
        "humidity":    "humidity_percent",
    })

    df["timestamp"]        = pd.to_datetime(df["timestamp"])
    df["co2_ppm"]          = pd.to_numeric(df["co2_ppm"],          errors="coerce")
    df["temperature_c"]    = pd.to_numeric(df["temperature_c"],    errors="coerce")
    df["humidity_percent"] = pd.to_numeric(df["humidity_percent"], errors="coerce")

    df = df.dropna(subset=["co2_ppm", "temperature_c", "humidity_percent"], how="all")
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info(f"Fetched {len(df):,} rows from sensor_data (last {hours}h)")
    return df


async def fetch_sensor_info(db: AsyncSession, sensor_id: int) -> dict:
    """Fetch sensor name + barangay name for insight generation."""
    result = await db.execute(
        select(Sensor.sensor_id, Sensor.sensor_name, Barangay.barangay_name)
        .join(Barangay, Sensor.barangay_id == Barangay.barangay_id)
        .where(Sensor.sensor_id == sensor_id)
    )
    row = result.fetchone()
    if not row:
        return {"sensor_id": sensor_id, "sensor_name": f"NODE-{sensor_id:02d}", "barangay_name": "Naga City"}
    return {
        "sensor_id":    row.sensor_id,
        "sensor_name":  row.sensor_name,
        "barangay_name": row.barangay_name,
    }


async def fetch_all_active_sensors(db: AsyncSession) -> list:
    """Return list of sensor_ids that have data in the last 24h."""
    since = datetime.utcnow() - timedelta(hours=24)
    result = await db.execute(
        select(SensorData.sensor_id)
        .where(SensorData.recorded_at >= since)
        .distinct()
    )
    return [row.sensor_id for row in result.fetchall()]


# ============================================================================
# DAILY PREDICTIONS — write & read
# ============================================================================

async def save_daily_prediction(
    db: AsyncSession,
    prediction_date: date,
    sensor_id: Optional[int],
    target: str,
    summary: dict,
    hourly: list,
) -> DailyPrediction:
    """Upsert daily prediction for one target + sensor."""
    pred_dt = datetime.combine(prediction_date, datetime.min.time())

    existing = await db.execute(
        select(DailyPrediction).where(
            and_(
                DailyPrediction.prediction_date == pred_dt,
                DailyPrediction.target == target,
                DailyPrediction.sensor_id == sensor_id,
            )
        )
    )
    existing_row = existing.scalar_one_or_none()
    if existing_row:
        await db.delete(existing_row)
        await db.flush()

    daily = DailyPrediction(
        prediction_date=pred_dt,
        sensor_id=sensor_id,
        target=target,
        unit=summary.get("unit", ""),
        mean_value=summary.get("mean"),
        min_value=summary.get("min"),
        max_value=summary.get("max"),
        model_used="LSTM",
    )
    db.add(daily)
    await db.flush()

    for h in hourly:
        db.add(HourlyPrediction(
            daily_id=daily.id,
            hour=h["hour"],
            timestamp=datetime.fromisoformat(h["timestamp"]),
            predicted_value=h["value"],
        ))

    await db.commit()
    logger.info(f"Saved prediction: sensor={sensor_id} target={target} date={prediction_date}")
    return daily


async def save_insight(
    db: AsyncSession,
    prediction_date: date,
    sensor_id: Optional[int],
    barangay: str,
    insight_text: str,
    llm_backend: str = "groq",
) -> PredictionInsight:
    """Save Groq insight for a prediction run."""
    pred_dt = datetime.combine(prediction_date, datetime.min.time())
    insight = PredictionInsight(
        prediction_date=pred_dt,
        sensor_id=sensor_id,
        barangay=barangay,
        insight_text=insight_text,
        llm_backend=llm_backend,
    )
    db.add(insight)
    await db.commit()
    logger.info(f"Saved insight for sensor={sensor_id} date={prediction_date}")
    return insight


# ============================================================================
# READ — for API endpoints
# ============================================================================

async def get_latest_predictions(db: AsyncSession, sensor_id: Optional[int] = None) -> list:
    latest_date_q = select(DailyPrediction.prediction_date).order_by(desc(DailyPrediction.prediction_date)).limit(1)
    if sensor_id:
        latest_date_q = latest_date_q.where(DailyPrediction.sensor_id == sensor_id)
    latest_date = (await db.execute(latest_date_q)).scalar_one_or_none()
    if not latest_date:
        return []

    q = select(DailyPrediction).where(DailyPrediction.prediction_date == latest_date)
    if sensor_id:
        q = q.where(DailyPrediction.sensor_id == sensor_id)
    return (await db.execute(q.order_by(DailyPrediction.target))).scalars().all()


async def get_predictions_by_date(db: AsyncSession, prediction_date: date, sensor_id: Optional[int] = None) -> list:
    pred_dt = datetime.combine(prediction_date, datetime.min.time())
    q = select(DailyPrediction).where(DailyPrediction.prediction_date == pred_dt)
    if sensor_id:
        q = q.where(DailyPrediction.sensor_id == sensor_id)
    return (await db.execute(q.order_by(DailyPrediction.target))).scalars().all()


async def get_hourly_predictions(db: AsyncSession, daily_id: int) -> list:
    result = await db.execute(
        select(HourlyPrediction)
        .where(HourlyPrediction.daily_id == daily_id)
        .order_by(HourlyPrediction.hour)
    )
    return result.scalars().all()


async def get_prediction_history(db: AsyncSession, days: int = 30, sensor_id: Optional[int] = None) -> list:
    since = datetime.utcnow() - timedelta(days=days)
    q = select(DailyPrediction).where(DailyPrediction.prediction_date >= since)
    if sensor_id:
        q = q.where(DailyPrediction.sensor_id == sensor_id)
    return (await db.execute(q.order_by(desc(DailyPrediction.prediction_date), DailyPrediction.target))).scalars().all()


async def get_latest_insight(db: AsyncSession, sensor_id: Optional[int] = None) -> Optional[PredictionInsight]:
    q = select(PredictionInsight).order_by(desc(PredictionInsight.prediction_date))
    if sensor_id:
        q = q.where(PredictionInsight.sensor_id == sensor_id)
    return (await db.execute(q.limit(1))).scalar_one_or_none()


async def get_insight_history(db: AsyncSession, days: int = 30, sensor_id: Optional[int] = None) -> list:
    since = datetime.utcnow() - timedelta(days=days)
    q = (select(PredictionInsight)
         .where(PredictionInsight.prediction_date >= since)
         .order_by(desc(PredictionInsight.prediction_date)))
    if sensor_id:
        q = q.where(PredictionInsight.sensor_id == sensor_id)
    return (await db.execute(q)).scalars().all()