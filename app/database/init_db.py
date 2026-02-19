"""
Database Initialization Script
================================
Run this ONCE to create all prediction tables in MySQL.
Your existing IoT sensor_readings table is NOT touched.

Usage:
    python -m app.database.init_db
"""

import asyncio
import sys
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.database.connection import engine, check_connection
from app.database.models import Base

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


async def init_db():
    logger.info("Checking database connection...")
    ok = await check_connection()
    if not ok:
        logger.error("Cannot connect to MySQL. Check your .env file.")
        sys.exit(1)

    logger.info("Creating tables (if not exist)...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("✅ Done! Tables created:")
    logger.info("   - sensor_readings      (IoT input — already exists, not modified)")
    logger.info("   - daily_predictions    (LSTM daily forecast summaries)")
    logger.info("   - hourly_predictions   (24 hourly values per forecast)")
    logger.info("   - prediction_insights  (Groq AI-generated insights)")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(init_db())