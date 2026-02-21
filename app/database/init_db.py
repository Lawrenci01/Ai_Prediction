import asyncio
import logging
from dotenv import load_dotenv
load_dotenv()
from app.database.connection import engine, Base
from app.database.models import (
    Barangay,
    Establishment,
    Sensor,
    SensorData,
    DailyPrediction,
    HourlyPrediction,
    PredictionInsight,
    LocationSafetyInsight,
)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
async def create_tables():
    logger.info("Connecting to database...")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Tables created / verified successfully.")
    except Exception as e:
        logger.error(f"Table creation failed: {e}")
        raise
    finally:
        await engine.dispose()
if __name__ == "__main__":
    asyncio.run(create_tables())