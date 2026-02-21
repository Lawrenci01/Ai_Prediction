import logging
import os
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database.connection import engine, Base, get_db, check_connection
from app.database.repository import (
    fetch_recent_sensor_data,
    fetch_all_active_sensors,
    get_latest_daily,
    get_daily_by_date,
    get_daily_history,
    get_hourly_by_sensor_date,
    get_hourly_latest,
    get_latest_insight,
    get_insight_history,
)
from app.database.models import Sensor, Barangay, DailyPrediction, HourlyPrediction, PredictionInsight

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

_raw_origins = os.getenv("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()] or ["*"]

if ALLOWED_ORIGINS == ["*"]:
    logger.warning(
        "CORS is open to all origins (ALLOWED_ORIGINS=*). "
        "Set ALLOWED_ORIGINS to your frontend domain before deploying to production."
    )

# Browsers reject credentials=True with wildcard origin.
_allow_credentials = ALLOWED_ORIGINS != ["*"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    from app.scheduler.daily_scheduler import scheduler_loop as daily_loop

    logger.info("Starting AI Prediction API...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await check_connection()

    daily_task = asyncio.create_task(daily_loop())
    logger.info("Daily scheduler started in background")

    yield

    daily_task.cancel()
    try:
        await daily_task
    except asyncio.CancelledError:
        pass
    await engine.dispose()
    logger.info("Scheduler stopped cleanly")


app = FastAPI(
    title="AI Prediction API",
    description="IoT Climate Prediction — CO2, Temperature, Humidity | Naga City",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


def serialize_daily(row: DailyPrediction) -> dict:
    return {
        "id": row.id,
        "prediction_date": row.prediction_date.strftime("%Y-%m-%d") if row.prediction_date else None,
        "run_at": row.run_at.isoformat() if row.run_at else None,
        "sensor_id": row.sensor_id,
        "establishment": row.establishment_name,
        "barangay": row.barangay_name,
        "co2": {"mean": row.co2_mean, "min": row.co2_min, "max": row.co2_max},
        "temperature": {"mean": row.temp_mean, "min": row.temp_min, "max": row.temp_max},
        "humidity": {"mean": row.humidity_mean, "min": row.humidity_min, "max": row.humidity_max},
        "model_used": row.model_used,
    }


def serialize_hourly(row: HourlyPrediction) -> dict:
    return {
        "id": row.id,
        "hour": row.hour,
        "timestamp": row.timestamp.isoformat() if row.timestamp else None,
        "establishment": row.establishment_name,
        "barangay": row.barangay_name,
        "co2_ppm": row.co2_ppm,
        "temperature_c": row.temperature_c,
        "humidity_percent": row.humidity_percent,
        "safe_status": row.safe_status,
        "insight_text": row.insight_text,
    }


def serialize_insight(row: PredictionInsight) -> dict:
    return {
        "id": row.id,
        "prediction_date": row.prediction_date.strftime("%Y-%m-%d") if row.prediction_date else None,
        "run_at": row.run_at.isoformat() if row.run_at else None,
        "sensor_id": row.sensor_id,
        "barangay": row.barangay,
        "insight_text": row.insight_text,
        "llm_backend": row.llm_backend,
    }


@app.get("/")
async def root():
    return {"status": "ok", "service": "AI Prediction API", "time": datetime.now(timezone.utc).isoformat()}


@app.get("/api/sensors")
async def list_sensors(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Sensor.sensor_id, Sensor.sensor_name, Barangay.barangay_name, Barangay.city)
        .join(Barangay, Sensor.barangay_id == Barangay.barangay_id)
        .order_by(Sensor.sensor_id)
    )
    rows = result.fetchall()
    return {
        "sensors": [
            {"sensor_id": r.sensor_id, "sensor_name": r.sensor_name, "barangay": r.barangay_name, "city": r.city}
            for r in rows
        ],
        "count": len(rows),
    }


@app.get("/api/predictions/latest")
async def get_latest(sensor_id: Optional[int] = Query(default=None), db: AsyncSession = Depends(get_db)):
    row = await get_latest_daily(db, sensor_id=sensor_id)
    if not row:
        raise HTTPException(status_code=404, detail="No predictions found.")
    return serialize_daily(row)


@app.get("/api/predictions/date/{prediction_date}")
async def get_by_date(prediction_date: date, sensor_id: Optional[int] = Query(default=None), db: AsyncSession = Depends(get_db)):
    rows = await get_daily_by_date(db, prediction_date, sensor_id=sensor_id)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No predictions for {prediction_date}.")
    return {"predictions": [serialize_daily(r) for r in rows]}


@app.get("/api/predictions/history")
async def get_history(days: int = Query(default=30, ge=1, le=365), sensor_id: Optional[int] = Query(default=None), db: AsyncSession = Depends(get_db)):
    rows = await get_daily_history(db, days=days, sensor_id=sensor_id)
    return {"predictions": [serialize_daily(r) for r in rows], "count": len(rows)}


@app.get("/api/predictions/hourly/latest")
async def get_hourly_latest_endpoint(sensor_id: int = Query(...), db: AsyncSession = Depends(get_db)):
    rows = await get_hourly_latest(db, sensor_id=sensor_id)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No hourly predictions for sensor {sensor_id} today.")
    return {
        "sensor_id": sensor_id,
        "date": datetime.now(timezone.utc).date().isoformat(),
        "total_hours": len(rows),
        "hourly": [serialize_hourly(r) for r in rows],
    }


@app.get("/api/predictions/hourly/{sensor_id}/{prediction_date}")
async def get_hourly_by_date(sensor_id: int, prediction_date: date, db: AsyncSession = Depends(get_db)):
    rows = await get_hourly_by_sensor_date(db, sensor_id=sensor_id, prediction_date=prediction_date)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No hourly predictions for sensor {sensor_id} on {prediction_date}.")
    return {
        "sensor_id": sensor_id,
        "date": str(prediction_date),
        "total_hours": len(rows),
        "hourly": [serialize_hourly(r) for r in rows],
    }


@app.get("/api/insights/latest")
async def insight_latest(sensor_id: Optional[int] = Query(default=None), db: AsyncSession = Depends(get_db)):
    row = await get_latest_insight(db, sensor_id=sensor_id)
    if not row:
        raise HTTPException(status_code=404, detail="No insights yet.")
    return serialize_insight(row)


@app.get("/api/insights/history")
async def insight_history(days: int = Query(default=30, ge=1, le=365), sensor_id: Optional[int] = Query(default=None), db: AsyncSession = Depends(get_db)):
    rows = await get_insight_history(db, days=days, sensor_id=sensor_id)
    return {"insights": [serialize_insight(r) for r in rows], "count": len(rows)}


@app.get("/api/sensor/latest")
async def sensor_latest(hours: int = Query(default=24, ge=1, le=168), sensor_id: Optional[int] = Query(default=None), db: AsyncSession = Depends(get_db)):
    df = await fetch_recent_sensor_data(db, hours=hours, sensor_id=sensor_id)
    if df.empty:
        raise HTTPException(status_code=404, detail="No sensor data found.")
    df_out = df.tail(200).copy()
    df_out["timestamp"] = df_out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return {
        "count": len(df_out),
        "latest_timestamp": df["timestamp"].max().isoformat(),
        "readings": df_out.to_dict(orient="records"),
    }


@app.post("/api/predict/run-now")
async def trigger_prediction(background_tasks: BackgroundTasks, sensor_id: Optional[int] = Query(default=None)):
    from app.scheduler.daily_scheduler import run_prediction_job
    background_tasks.add_task(run_prediction_job, sensor_id)
    return {
        "status": "started",
        "sensor_id": sensor_id,
        "message": "Prediction triggered. Check /api/predictions/hourly/latest shortly.",
        "time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/status")
async def get_status(db: AsyncSession = Depends(get_db)):
    db_ok = await check_connection()
    try:
        from app.inference import registry
        model_status = registry.status()
    except Exception as e:
        model_status = {"error": str(e)}

    active_sensors = await fetch_all_active_sensors(db)
    return {
        "api": "ok",
        "db": "ok" if db_ok else "error",
        "active_sensors": active_sensors,
        "models": model_status,
        "time": datetime.now(timezone.utc).isoformat(),
    }