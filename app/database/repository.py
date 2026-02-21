import logging
from datetime import datetime, date, timedelta, timezone
from typing import Optional

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, and_, func

from app.database.models import (
    SensorData, Sensor, Barangay, Establishment,
    DailyPrediction, HourlyPrediction, PredictionInsight,
)

logger = logging.getLogger(__name__)


async def fetch_recent_sensor_data(
    db: AsyncSession,
    hours: int = 336,
    sensor_id: Optional[int] = None,
) -> pd.DataFrame:
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    q = (
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
        q = q.where(SensorData.sensor_id == sensor_id)

    rows = (await db.execute(q)).fetchall()
    if not rows:
        logger.warning(f"No sensor_data found in last {hours}h")
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["recorded_at", "co2_density", "temperature_c", "humidity", "sensor_id"])
    df = df.rename(columns={"recorded_at": "timestamp", "co2_density": "co2_ppm", "humidity": "humidity_percent"})
    df["timestamp"]        = pd.to_datetime(df["timestamp"])
    df["co2_ppm"]          = pd.to_numeric(df["co2_ppm"],          errors="coerce")
    df["temperature_c"]    = pd.to_numeric(df["temperature_c"],    errors="coerce")
    df["humidity_percent"] = pd.to_numeric(df["humidity_percent"], errors="coerce")
    df = df.dropna(subset=["co2_ppm", "temperature_c", "humidity_percent"], how="all")
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"Fetched {len(df):,} rows from sensor_data (last {hours}h)")
    return df


async def fetch_sensor_info(db: AsyncSession, sensor_id: int) -> dict:
    result = await db.execute(
        select(Sensor.sensor_id, Sensor.sensor_name, Barangay.barangay_name)
        .join(Barangay, Sensor.barangay_id == Barangay.barangay_id)
        .where(Sensor.sensor_id == sensor_id)
    )
    row = result.fetchone()
    if not row:
        return {"sensor_id": sensor_id, "sensor_name": f"NODE-{sensor_id:02d}", "barangay_name": "Naga City"}
    return {"sensor_id": row.sensor_id, "sensor_name": row.sensor_name, "barangay_name": row.barangay_name}


async def fetch_sensor_full_info(db: AsyncSession, sensor_id: int) -> dict:
    result = await db.execute(
        select(
            Sensor.sensor_id,
            Sensor.sensor_name,
            Barangay.barangay_name,
            Establishment.establishment_name,
            Establishment.establishment_type,
        )
        .join(Barangay, Sensor.barangay_id == Barangay.barangay_id)
        .outerjoin(Establishment, Sensor.establishment_id == Establishment.establishment_id)
        .where(Sensor.sensor_id == sensor_id)
    )
    row = result.fetchone()
    if not row:
        return {
            "sensor_id": sensor_id, "sensor_name": f"NODE-{sensor_id:02d}",
            "barangay_name": "Naga City", "establishment_name": None, "establishment_type": None,
        }
    return {
        "sensor_id":          row.sensor_id,
        "sensor_name":        row.sensor_name,
        "barangay_name":      row.barangay_name,
        "establishment_name": row.establishment_name,
        "establishment_type": row.establishment_type,
    }


async def fetch_latest_sensor_reading(db: AsyncSession, sensor_id: int) -> Optional[dict]:
    result = await db.execute(
        select(
            SensorData.co2_density,
            SensorData.temperature_c,
            SensorData.humidity,
            SensorData.heat_index_c,
            SensorData.recorded_at,
        )
        .where(SensorData.sensor_id == sensor_id)
        .order_by(SensorData.recorded_at.desc())
        .limit(1)
    )
    row = result.fetchone()
    if not row:
        return None
    return {
        "co2_ppm":          float(row.co2_density  or 0),
        "temperature_c":    float(row.temperature_c or 0),
        "humidity_percent": float(row.humidity      or 0),
        "heat_index_c":     float(row.heat_index_c  or 0),
        "recorded_at":      row.recorded_at,
    }


async def fetch_all_active_sensors(db: AsyncSession) -> list:
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    result = await db.execute(
        select(SensorData.sensor_id)
        .where(SensorData.recorded_at >= since)
        .distinct()
    )
    return [row.sensor_id for row in result.fetchall()]


async def save_daily_prediction(
    db: AsyncSession,
    prediction_date: date,
    sensor_id: Optional[int],
    establishment_name: Optional[str],
    establishment_type: Optional[str],
    barangay_name: Optional[str],
    co2: dict,
    temp: dict,
    humidity: dict,
) -> DailyPrediction:
    pred_dt  = datetime.combine(prediction_date, datetime.min.time())
    existing = (await db.execute(
        select(DailyPrediction).where(
            and_(DailyPrediction.prediction_date == pred_dt, DailyPrediction.sensor_id == sensor_id)
        )
    )).scalar_one_or_none()

    if existing:
        await db.delete(existing)
        await db.flush()

    daily = DailyPrediction(
        prediction_date    = pred_dt,
        sensor_id          = sensor_id,
        establishment_name = establishment_name,
        establishment_type = establishment_type,
        barangay_name      = barangay_name,
        co2_mean           = co2.get("mean"),
        co2_min            = co2.get("min"),
        co2_max            = co2.get("max"),
        temp_mean          = temp.get("mean"),
        temp_min           = temp.get("min"),
        temp_max           = temp.get("max"),
        humidity_mean      = humidity.get("mean"),
        humidity_min       = humidity.get("min"),
        humidity_max       = humidity.get("max"),
        model_used         = "LSTM",
    )
    db.add(daily)
    await db.flush()
    await db.commit()
    await db.refresh(daily)
    logger.info(f"Saved daily prediction: sensor={sensor_id} date={prediction_date} id={daily.id}")
    return daily


async def save_hourly_prediction(
    db: AsyncSession,
    daily_id: int,
    sensor_id: int,
    establishment_name: Optional[str],
    establishment_type: Optional[str],
    barangay_name: Optional[str],
    hour: int,
    timestamp: datetime,
    co2_ppm: float,
    temperature_c: float,
    humidity_percent: float,
    safe_status: str,
    insight_text: str,
    llm_backend: str = "groq",
) -> HourlyPrediction:
    existing = (await db.execute(
        select(HourlyPrediction).where(
            and_(
                HourlyPrediction.sensor_id == sensor_id,
                HourlyPrediction.timestamp == timestamp,
            )
        )
    )).scalar_one_or_none()

    if existing:
        await db.delete(existing)
        await db.flush()

    row = HourlyPrediction(
        daily_id           = daily_id,
        sensor_id          = sensor_id,
        establishment_name = establishment_name,
        establishment_type = establishment_type,
        barangay_name      = barangay_name,
        hour               = hour,
        timestamp          = timestamp,
        co2_ppm            = co2_ppm,
        temperature_c      = temperature_c,
        humidity_percent   = humidity_percent,
        safe_status        = safe_status,
        insight_text       = insight_text,
        llm_backend        = llm_backend,
    )
    db.add(row)
    await db.commit()
    return row


async def save_insight(
    db: AsyncSession,
    prediction_date: date,
    sensor_id: Optional[int],
    barangay: str,
    insight_text: str,
    llm_backend: str = "groq",
) -> PredictionInsight:
    pred_dt  = datetime.combine(prediction_date, datetime.min.time())
    existing = (await db.execute(
        select(PredictionInsight).where(
            and_(
                PredictionInsight.prediction_date == pred_dt,
                PredictionInsight.sensor_id       == sensor_id,
            )
        )
    )).scalar_one_or_none()

    if existing:
        await db.delete(existing)
        await db.flush()

    insight = PredictionInsight(
        prediction_date = pred_dt,
        sensor_id       = sensor_id,
        barangay        = barangay,
        insight_text    = insight_text,
        llm_backend     = llm_backend,
    )
    db.add(insight)
    await db.commit()
    logger.info(f"Saved daily insight: sensor={sensor_id} date={prediction_date}")
    return insight


async def get_latest_daily(
    db: AsyncSession,
    sensor_id: Optional[int] = None,
) -> Optional[DailyPrediction]:
    q = select(DailyPrediction).order_by(desc(DailyPrediction.prediction_date))
    if sensor_id:
        q = q.where(DailyPrediction.sensor_id == sensor_id)
    return (await db.execute(q.limit(1))).scalar_one_or_none()


async def get_daily_by_date(
    db: AsyncSession,
    prediction_date: date,
    sensor_id: Optional[int] = None,
) -> list:
    pred_dt = datetime.combine(prediction_date, datetime.min.time())
    q = select(DailyPrediction).where(DailyPrediction.prediction_date == pred_dt)
    if sensor_id:
        q = q.where(DailyPrediction.sensor_id == sensor_id)
    return (await db.execute(q)).scalars().all()


async def get_daily_history(
    db: AsyncSession,
    days: int = 30,
    sensor_id: Optional[int] = None,
) -> list:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    q = select(DailyPrediction).where(DailyPrediction.prediction_date >= since)
    if sensor_id:
        q = q.where(DailyPrediction.sensor_id == sensor_id)
    return (await db.execute(q.order_by(desc(DailyPrediction.prediction_date)))).scalars().all()


async def get_hourly_by_daily(db: AsyncSession, daily_id: int) -> list:
    result = await db.execute(
        select(HourlyPrediction)
        .where(HourlyPrediction.daily_id == daily_id)
        .order_by(HourlyPrediction.hour)
    )
    return result.scalars().all()


async def get_hourly_by_sensor_date(
    db: AsyncSession,
    sensor_id: int,
    prediction_date: date,
) -> list:
    start  = datetime.combine(prediction_date, datetime.min.time())
    end    = datetime.combine(prediction_date, datetime.max.time())
    result = await db.execute(
        select(HourlyPrediction)
        .where(and_(
            HourlyPrediction.sensor_id == sensor_id,
            HourlyPrediction.timestamp >= start,
            HourlyPrediction.timestamp <= end,
        ))
        .order_by(HourlyPrediction.hour)
    )
    return result.scalars().all()


async def get_hourly_latest(db: AsyncSession, sensor_id: int) -> list:
    return await get_hourly_by_sensor_date(db, sensor_id, datetime.now(timezone.utc).date())


async def get_latest_insight(db: AsyncSession, sensor_id: Optional[int] = None) -> Optional[PredictionInsight]:
    q = select(PredictionInsight).order_by(desc(PredictionInsight.prediction_date))
    if sensor_id:
        q = q.where(PredictionInsight.sensor_id == sensor_id)
    return (await db.execute(q.limit(1))).scalar_one_or_none()


async def get_insight_history(db: AsyncSession, days: int = 30, sensor_id: Optional[int] = None) -> list:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    q = (
        select(PredictionInsight)
        .where(PredictionInsight.prediction_date >= since)
        .order_by(desc(PredictionInsight.prediction_date))
    )
    if sensor_id:
        q = q.where(PredictionInsight.sensor_id == sensor_id)
    return (await db.execute(q)).scalars().all()