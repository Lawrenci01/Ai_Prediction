"""
Realtime Barangay Insight
==========================
GET /api/realtime/insight        — latest cached insight + averages
GET /api/realtime/insight/history — last N rows from realtime_reports table

Every 5 minutes (background task started in lifespan):
  1. Query sensor_data — get latest reading per sensor via minute_stamp
  2. JOIN sensor → barangay → compute per-barangay + city-wide averages
  3. Call Groq for LLM insight
  4. Save snapshot to realtime_reports table (minute_stamp granularity)
  5. Cache result in memory — API serves from cache instantly

Table: realtime_reports
  - minute_stamp   DATETIME  (truncated to minute, UNIQUE — no duplicate snapshots)
  - co2_avg, temp_avg, hum_avg, heat_index_avg
  - top_barangay, top_carbon_level
  - very_high_count, total_sensors, total_barangays
  - insight_text, llm_success

Add to main.py lifespan (BEFORE yield):
    from app.report.realtime_insight import start_realtime_loop
    realtime_task = asyncio.create_task(start_realtime_loop())

Add to main.py lifespan (AFTER yield, cleanup):
    realtime_task.cancel()
    try:
        await realtime_task
    except asyncio.CancelledError:
        pass

Add router to main.py:
    from app.report.realtime_insight import router as realtime_router
    app.include_router(realtime_router)
"""

import asyncio
import logging
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy import Column, Integer, Float, String, DateTime, Text, UniqueConstraint, select, desc, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.connection import get_db, AsyncSessionLocal, engine, Base
from app.llm_engine import _call_groq, heat_index, co2_risk_level, comfort_label

logger = logging.getLogger(__name__)
router = APIRouter()

INTERVAL_SECONDS = 300  # 5 minutes


# ============================================================================
# MODEL — realtime_reports table
# ============================================================================

class RealtimeReport(Base):
    __tablename__ = "realtime_reports"
    __table_args__ = (
        UniqueConstraint("minute_stamp", name="uq_realtime_minute"),
    )

    id               = Column(Integer, primary_key=True, autoincrement=True)
    minute_stamp     = Column(DateTime, nullable=False, index=True)  # truncated to minute
    generated_at     = Column(DateTime, default=datetime.utcnow)

    # City-wide averages
    co2_avg          = Column(Float, nullable=True)
    temp_avg         = Column(Float, nullable=True)
    hum_avg          = Column(Float, nullable=True)
    heat_index_avg   = Column(Float, nullable=True)

    # Context
    top_barangay     = Column(String(120), nullable=True)
    top_carbon_level = Column(String(30),  nullable=True)
    very_high_count  = Column(Integer, default=0)
    total_sensors    = Column(Integer, default=0)
    total_barangays  = Column(Integer, default=0)

    # LLM
    insight_text     = Column(Text, nullable=True)
    llm_success      = Column(Integer, default=1)  # 1 = Groq, 0 = fallback


# ============================================================================
# IN-MEMORY CACHE — serves API instantly between DB writes
# ============================================================================

_cache: dict = {
    "report":    None,   # latest RealtimeReport-like dict
    "barangays": None,   # per-barangay breakdown
    "cached_at": None,
}


def _seconds_until_refresh() -> int:
    if _cache["cached_at"] is None:
        return 0
    elapsed = (datetime.utcnow() - _cache["cached_at"]).total_seconds()
    return max(0, int(INTERVAL_SECONDS - elapsed))


# ============================================================================
# QUERY — latest reading per sensor via minute_stamp, grouped by barangay
# ============================================================================

async def _fetch_barangay_averages(db: AsyncSession) -> list[dict]:
    result = await db.execute(text("""
        SELECT
            b.barangay_name,
            AVG(sd.co2_density)   AS co2_avg,
            AVG(sd.temperature_c) AS temp_avg,
            AVG(sd.humidity)      AS hum_avg,
            AVG(sd.heat_index_c)  AS hi_avg,
            COUNT(sd.sensor_id)   AS sensor_count,
            MAX(sd.carbon_level)  AS carbon_level,
            MAX(sd.recorded_at)   AS last_seen
        FROM sensor_data sd
        JOIN (
            SELECT sensor_id, MAX(minute_stamp) AS max_ts
            FROM sensor_data
            GROUP BY sensor_id
        ) latest ON sd.sensor_id   = latest.sensor_id
               AND sd.minute_stamp = latest.max_ts
        JOIN sensor   s ON sd.sensor_id  = s.sensor_id
        JOIN barangay b ON s.barangay_id = b.barangay_id
        GROUP BY b.barangay_name
        ORDER BY co2_avg DESC
    """))

    return [
        {
            "barangay_name": r.barangay_name,
            "co2_avg":       round(float(r.co2_avg),  4) if r.co2_avg  is not None else None,
            "temp_avg":      round(float(r.temp_avg), 2) if r.temp_avg is not None else None,
            "hum_avg":       round(float(r.hum_avg),  1) if r.hum_avg  is not None else None,
            "hi_avg":        round(float(r.hi_avg),   1) if r.hi_avg   is not None else None,
            "sensor_count":  int(r.sensor_count),
            "carbon_level":  r.carbon_level,
            "last_seen":     r.last_seen.isoformat() if r.last_seen else None,
        }
        for r in result.fetchall()
    ]


def _city_averages(barangays: list[dict]) -> dict:
    co2s  = [b["co2_avg"]  for b in barangays if b["co2_avg"]  is not None]
    temps = [b["temp_avg"] for b in barangays if b["temp_avg"] is not None]
    hums  = [b["hum_avg"]  for b in barangays if b["hum_avg"]  is not None]

    co2_avg  = round(sum(co2s)  / len(co2s),  4) if co2s  else 0.0
    temp_avg = round(sum(temps) / len(temps), 2) if temps else 28.0
    hum_avg  = round(sum(hums)  / len(hums),  1) if hums  else 75.0
    hi       = round(heat_index(temp_avg, hum_avg), 1)

    severity = {"VERY HIGH": 4, "HIGH": 3, "MODERATE": 2, "LOW": 1, "NORMAL": 0}
    top = max(barangays, key=lambda b: (
        severity.get((b["carbon_level"] or "NORMAL").upper(), 0),
        b["co2_avg"] or 0
    ))

    very_high = sum(1 for b in barangays if (b["carbon_level"] or "").upper() == "VERY HIGH")

    return {
        "co2_avg":         co2_avg,
        "temp_avg":        temp_avg,
        "hum_avg":         hum_avg,
        "heat_index":      hi,
        "co2_risk":        co2_risk_level(co2_avg),
        "comfort":         comfort_label(hi),
        "top_barangay":    top["barangay_name"],
        "top_co2":         top["co2_avg"],
        "top_level":       top["carbon_level"],
        "very_high_count": very_high,
        "total_barangays": len(barangays),
        "total_sensors":   sum(b["sensor_count"] for b in barangays),
    }


# ============================================================================
# LLM PROMPT
# ============================================================================

def _build_prompt(city: dict, barangays: list[dict]) -> str:
    now_str   = datetime.utcnow().strftime("%B %d, %Y %H:%M UTC")
    top3_lines = "\n".join([
        f"  - {b['barangay_name']}: CO2={b['co2_avg']} ppm, "
        f"Temp={b['temp_avg']}°C, Hum={b['hum_avg']}%, "
        f"Level={b['carbon_level'] or 'NORMAL'}"
        for b in barangays[:3]
    ])
    alert = (
        f"{city['very_high_count']} barangay(s) at VERY HIGH — urgent."
        if city["very_high_count"] > 0
        else "No barangays at VERY HIGH level right now."
    )

    return f"""You are a real-time environmental monitoring AI for Naga City, Philippines.

Live sensor data as of {now_str} — {city['total_sensors']} sensors across {city['total_barangays']} barangays:

CITY-WIDE AVERAGES:
- CO2 Density : {city['co2_avg']} ppm — {city['co2_risk']}
- Temperature : {city['temp_avg']}°C — feels like {city['heat_index']}°C ({city['comfort']})
- Humidity    : {city['hum_avg']}%

ALERT: {alert}

TOP 3 BARANGAYS BY CO2:
{top3_lines}

HIGHEST RISK: {city['top_barangay']} ({city['top_level'] or 'NORMAL'})

Task:
- Reason about what these live readings mean RIGHT NOW for Naga City.
- Consider CO2 + temperature + humidity interactions.
- Give exactly 2 immediate actionable recommendations: RESIDENTS and HEALTH OFFICIALS.
- Be concise — this is a real-time dashboard update.
- Use the actual numbers in every sentence.

Format EXACTLY (no preamble):

STATUS: [One sentence on current city-wide conditions using actual values.]

RESIDENTS: [What residents should do right now.]

HEALTH OFFICIALS: [What officials should act on immediately.]
"""


def _fallback(city: dict) -> str:
    if city["very_high_count"] > 0:
        res = "avoid prolonged outdoor exposure near affected barangays"
        off = f"deploy teams to {city['top_barangay']} — VERY HIGH CO2 detected"
    elif city["comfort"] in ["danger", "extreme danger"]:
        res = "stay indoors, hydrate, and avoid midday outdoor activity"
        off = "ensure cooling centers are open and check on vulnerable residents"
    elif city["co2_risk"] in ["moderately concentrated", "high enough to affect focus indoors"]:
        res = "open windows and improve indoor ventilation"
        off = f"inspect {city['top_barangay']} for CO2 sources"
    else:
        res = "conditions are manageable — maintain normal ventilation"
        off = "continue routine monitoring, no immediate action required"

    return (
        f"STATUS: City-wide CO2 at {city['co2_avg']} ppm ({city['co2_risk']}), "
        f"temperature {city['temp_avg']}°C (feels like {city['heat_index']}°C — {city['comfort']}), "
        f"humidity {city['hum_avg']}%.\n\n"
        f"RESIDENTS: {res.capitalize()}.\n\n"
        f"HEALTH OFFICIALS: {off.capitalize()}."
    )


# ============================================================================
# CORE JOB — runs every 5 minutes
# ============================================================================

async def _run_realtime_job():
    """Fetch averages, call Groq, save to DB, update cache."""
    try:
        async with AsyncSessionLocal() as db:
            barangays = await _fetch_barangay_averages(db)

        if not barangays:
            logger.warning("Realtime job: no sensor data found.")
            return

        city = _city_averages(barangays)

        # Groq insight
        llm_success = 1
        try:
            insight = await _call_groq(_build_prompt(city, barangays))
        except Exception as e:
            logger.error(f"Groq failed in realtime job: {e}")
            insight     = _fallback(city)
            llm_success = 0

        # Truncate to current minute for minute_stamp
        now          = datetime.utcnow()
        minute_stamp = now.replace(second=0, microsecond=0)

        report = RealtimeReport(
            minute_stamp     = minute_stamp,
            generated_at     = now,
            co2_avg          = city["co2_avg"],
            temp_avg         = city["temp_avg"],
            hum_avg          = city["hum_avg"],
            heat_index_avg   = city["heat_index"],
            top_barangay     = city["top_barangay"],
            top_carbon_level = city["top_level"],
            very_high_count  = city["very_high_count"],
            total_sensors    = city["total_sensors"],
            total_barangays  = city["total_barangays"],
            insight_text     = insight,
            llm_success      = llm_success,
        )

        # Upsert — skip if same minute already saved (unique constraint)
        async with AsyncSessionLocal() as db:
            existing = await db.execute(
                select(RealtimeReport).where(RealtimeReport.minute_stamp == minute_stamp)
            )
            if existing.scalar_one_or_none() is None:
                db.add(report)
                await db.commit()
                logger.info(
                    f"Realtime report saved — {minute_stamp} | "
                    f"CO2={city['co2_avg']} | Temp={city['temp_avg']}°C | "
                    f"Hum={city['hum_avg']}% | LLM={'OK' if llm_success else 'fallback'}"
                )
            else:
                logger.info(f"Realtime report already exists for {minute_stamp}, skipping.")

        # Update in-memory cache
        _cache["report"]    = {
            "minute_stamp":     minute_stamp.isoformat(),
            "generated_at":     now.isoformat(),
            "co2_avg":          city["co2_avg"],
            "temp_avg":         city["temp_avg"],
            "hum_avg":          city["hum_avg"],
            "heat_index_avg":   city["heat_index"],
            "co2_risk":         city["co2_risk"],
            "comfort":          city["comfort"],
            "top_barangay":     city["top_barangay"],
            "top_carbon_level": city["top_level"],
            "very_high_count":  city["very_high_count"],
            "total_sensors":    city["total_sensors"],
            "total_barangays":  city["total_barangays"],
            "insight_text":     insight,
            "llm_success":      bool(llm_success),
        }
        _cache["barangays"] = barangays
        _cache["cached_at"] = now

    except Exception as e:
        logger.error(f"Realtime job failed: {e}", exc_info=True)


# ============================================================================
# BACKGROUND LOOP — started from main.py lifespan
# ============================================================================

async def start_realtime_loop():
    """
    Infinite loop that runs the realtime job every 5 minutes.
    Align to the nearest 5-minute boundary on the clock
    (e.g. :00, :05, :10 ... :55) so reports are predictable.
    """
    logger.info(f"⏱  Realtime insight loop started — interval={INTERVAL_SECONDS}s")

    # Ensure table exists
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Run once immediately on startup
    await _run_realtime_job()

    while True:
        # Sleep until next 5-minute boundary
        now          = datetime.utcnow()
        minutes_past = now.minute % 5
        seconds_past = now.second
        wait         = (5 - minutes_past) * 60 - seconds_past
        if wait <= 0:
            wait += 300

        logger.info(f"⏰ Next realtime snapshot in {wait}s")
        await asyncio.sleep(wait)
        await _run_realtime_job()


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("/api/realtime/insight")
async def get_realtime_insight(
    force_refresh: bool = False,
    db: AsyncSession = Depends(get_db),
):
    """
    Latest city-wide realtime insight.
    Served from in-memory cache — refreshes every 5 minutes automatically.
    Use ?force_refresh=true to trigger an immediate recalculation.
    """
    if force_refresh:
        await _run_realtime_job()

    if _cache["report"] is None:
        # No data yet — run once synchronously
        await _run_realtime_job()

    if _cache["report"] is None:
        return {
            "cached":            False,
            "refreshes_in_secs": INTERVAL_SECONDS,
            "report":            None,
            "barangays":         [],
            "message":           "No sensor data available yet.",
        }

    return {
        "cached":            not force_refresh,
        "refreshes_in_secs": _seconds_until_refresh(),
        "report":            _cache["report"],
        "barangays":         _cache["barangays"],
    }


@router.get("/api/realtime/insight/history")
async def get_realtime_history(
    limit: int = Query(default=12, ge=1, le=1440, description="Number of snapshots — default 12 (60 min), max 1440 (5 days)"),
    from_ts: Optional[datetime] = Query(default=None, alias="from", description="Start timestamp e.g. 2026-02-19T20:00:00"),
    to_ts:   Optional[datetime] = Query(default=None, alias="to",   description="End timestamp   e.g. 2026-02-19T22:00:00"),
    db: AsyncSession = Depends(get_db),
):
    """
    Realtime snapshots from realtime_reports table.

    Filter options (use one or combine):
      ?limit=12                                          → last 60 minutes
      ?from=2026-02-19T20:00:00&to=2026-02-19T22:00:00  → specific range
      ?from=2026-02-19T20:00:00                          → from a point until now
    """
    q = select(RealtimeReport)

    if from_ts and to_ts:
        q = q.where(RealtimeReport.minute_stamp >= from_ts)              .where(RealtimeReport.minute_stamp <= to_ts)
    elif from_ts:
        q = q.where(RealtimeReport.minute_stamp >= from_ts)
    elif to_ts:
        q = q.where(RealtimeReport.minute_stamp <= to_ts)
    else:
        # No range given — fall back to latest N snapshots
        q = q.order_by(desc(RealtimeReport.minute_stamp)).limit(limit)

    if from_ts or to_ts:
        q = q.order_by(RealtimeReport.minute_stamp.asc())

    result = await db.execute(q)
    rows   = result.scalars().all()

    return {
        "count":    len(rows),
        "from":     from_ts.isoformat() if from_ts else None,
        "to":       to_ts.isoformat()   if to_ts   else None,
        "snapshots": [
            {
                "minute_stamp":     r.minute_stamp.isoformat(),
                "co2_avg":          r.co2_avg,
                "temp_avg":         r.temp_avg,
                "hum_avg":          r.hum_avg,
                "heat_index_avg":   r.heat_index_avg,
                "top_barangay":     r.top_barangay,
                "top_carbon_level": r.top_carbon_level,
                "very_high_count":  r.very_high_count,
                "total_sensors":    r.total_sensors,
                "insight_text":     r.insight_text,
                "llm_success":      bool(r.llm_success),
            }
            for r in rows
        ],
    }