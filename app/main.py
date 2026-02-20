import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database.connection import engine, Base, get_db, check_connection
from app.database.repository import (
    fetch_recent_sensor_data,
    fetch_all_active_sensors,
    get_latest_predictions,
    get_predictions_by_date,
    get_prediction_history,
    get_hourly_predictions,
    get_latest_insight,
    get_insight_history,
)
from app.database.models import (
    Sensor, Barangay, DailyPrediction, HourlyPrediction, PredictionInsight
)
from app.report.realtime_insight import router as realtime_router, start_realtime_loop

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await check_connection()

    from app.scheduler.daily_scheduler import scheduler_loop
    scheduler_task = asyncio.create_task(scheduler_loop())
    realtime_task = asyncio.create_task(start_realtime_loop())

    yield

    scheduler_task.cancel()
    realtime_task.cancel()

    for task in (scheduler_task, realtime_task):
        try:
            await task
        except asyncio.CancelledError:
            pass

    await engine.dispose()


app = FastAPI(
    title="AI Prediction API",
    description="IoT Climate Prediction — CO2, Temperature, Humidity | Naga City",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(realtime_router)


def serialize_daily(row: DailyPrediction) -> dict:
    return {
        "id":              row.id,
        "prediction_date": row.prediction_date.strftime("%Y-%m-%d") if row.prediction_date else None,
        "run_at":          row.run_at.isoformat() if row.run_at else None,
        "sensor_id":       row.sensor_id,
        "target":          row.target,
        "unit":            row.unit,
        "mean":            row.mean_value,
        "min":             row.min_value,
        "max":             row.max_value,
        "model_used":      row.model_used,
    }


def serialize_hourly(row: HourlyPrediction) -> dict:
    return {
        "hour":      row.hour,
        "timestamp": row.timestamp.isoformat() if row.timestamp else None,
        "value":     row.predicted_value,
    }


def serialize_insight(row: PredictionInsight) -> dict:
    return {
        "id":              row.id,
        "prediction_date": row.prediction_date.strftime("%Y-%m-%d") if row.prediction_date else None,
        "run_at":          row.run_at.isoformat() if row.run_at else None,
        "sensor_id":       row.sensor_id,
        "barangay":        row.barangay,
        "insight_text":    row.insight_text,
        "llm_backend":     row.llm_backend,
    }


@app.get("/")
async def root():
    return {"status": "ok", "service": "AI Prediction API", "time": datetime.utcnow().isoformat()}


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
            {"sensor_id": r.sensor_id, "sensor_name": r.sensor_name,
             "barangay": r.barangay_name, "city": r.city}
            for r in rows
        ],
        "count": len(rows),
    }


@app.get("/api/predictions/latest")
async def get_latest(
    sensor_id: Optional[int] = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    rows = await get_latest_predictions(db, sensor_id=sensor_id)
    if not rows:
        raise HTTPException(status_code=404, detail="No predictions found. Run the scheduler first.")

    grouped = {
        "prediction_date": rows[0].prediction_date.strftime("%Y-%m-%d"),
        "sensor_id": sensor_id,
    }
    for row in rows:
        grouped[row.target] = {
            "unit": row.unit,
            "mean": row.mean_value,
            "min":  row.min_value,
            "max":  row.max_value,
            "id":   row.id,
        }
    return grouped


@app.get("/api/predictions/date/{prediction_date}")
async def get_by_date(
    prediction_date: date,
    sensor_id: Optional[int] = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    rows = await get_predictions_by_date(db, prediction_date, sensor_id=sensor_id)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No predictions for {prediction_date}")
    grouped = {"prediction_date": str(prediction_date)}
    for row in rows:
        grouped[row.target] = serialize_daily(row)
    return grouped


@app.get("/api/predictions/history")
async def get_history(
    days: int = Query(default=30, ge=1, le=365),
    sensor_id: Optional[int] = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    rows = await get_prediction_history(db, days=days, sensor_id=sensor_id)
    return {"predictions": [serialize_daily(r) for r in rows], "count": len(rows)}


@app.get("/api/predictions/{prediction_id}/hourly")
async def get_hourly(prediction_id: int, db: AsyncSession = Depends(get_db)):
    rows = await get_hourly_predictions(db, daily_id=prediction_id)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No hourly data for id={prediction_id}")
    return {"daily_id": prediction_id, "hourly_values": [serialize_hourly(r) for r in rows]}


@app.get("/api/insights/latest")
async def insight_latest(
    sensor_id: Optional[int] = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    row = await get_latest_insight(db, sensor_id=sensor_id)
    if not row:
        raise HTTPException(status_code=404, detail="No insights yet. Run the scheduler first.")
    return serialize_insight(row)


@app.get("/api/insights/history")
async def insight_history(
    days: int = Query(default=30, ge=1, le=365),
    sensor_id: Optional[int] = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    rows = await get_insight_history(db, days=days, sensor_id=sensor_id)
    return {"insights": [serialize_insight(r) for r in rows], "count": len(rows)}


@app.get("/api/sensor/latest")
async def sensor_latest(
    hours: int = Query(default=24, ge=1, le=168),
    sensor_id: Optional[int] = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    df = await fetch_recent_sensor_data(db, hours=hours, sensor_id=sensor_id)
    if df.empty:
        raise HTTPException(status_code=404, detail="No sensor data found.")

    df_out = df.tail(200).copy()
    df_out["timestamp"] = df_out["timestamp"].dt.isoformat()

    return {
        "count":            len(df_out),
        "latest_timestamp": df["timestamp"].max().isoformat(),
        "readings":         df_out.to_dict(orient="records"),
    }


@app.post("/api/predict/run-now")
async def trigger_prediction(
    background_tasks: BackgroundTasks,
    sensor_id: Optional[int] = Query(default=None),
):
    from app.scheduler.daily_scheduler import run_prediction_job
    background_tasks.add_task(run_prediction_job, sensor_id)
    return {
        "status":    "started",
        "sensor_id": sensor_id,
        "message":   "Prediction job triggered. Check logs for progress.",
        "time":      datetime.utcnow().isoformat(),
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
        "api":            "ok",
        "db":             "ok" if db_ok else "error",
        "active_sensors": active_sensors,
        "models":         model_status,
        "time":           datetime.utcnow().isoformat(),
    }