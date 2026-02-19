"""
Monthly Retrain Scheduler
=========================
Retrains LSTM models using newly collected IoT sensor data.
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from app.services.export_service import export_new_iot_data
from training.train import train_model

# ==========================================
# CONFIG
# ==========================================

RETRAIN_LOG_FILE = Path("retrain_log.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ==========================================
# MAIN RETRAIN JOB
# ==========================================

async def run_retrain_job(evaluate_only: bool = False):

    # ======================================
    # 🔎 ENVIRONMENT DEBUG (IMPORTANT)
    # ======================================
    print("\n========== ENV DEBUG ==========")
    print("DB_HOST =", os.getenv("DB_HOST"))
    print("DB_PORT =", os.getenv("DB_PORT"))
    print("DB_NAME =", os.getenv("DB_NAME"))
    print("DB_USER =", os.getenv("DB_USER"))
    print("DB_PASS exists =", os.getenv("DB_PASS") is not None)
    print("DB_PASSWORD exists =", os.getenv("DB_PASSWORD") is not None)
    print("================================\n")

    logger.info("=" * 60)
    logger.info(f"🔄 Monthly retrain job — {datetime.utcnow().isoformat()}")
    logger.info("=" * 60)

    # Explicit model key map (avoid fragile substring matching)
    MODEL_KEY_MAP = {
        "co2_ppm": "lstm_co2",
        "temperature_c": "lstm_temperature",
        "humidity_percent": "lstm_humidity",
    }

    # ======================================
    # 1️⃣ Determine Last Retrain Date
    # ======================================

    last_retrain = datetime.utcnow() - timedelta(days=60)

    if RETRAIN_LOG_FILE.exists():
        df_log = pd.read_csv(RETRAIN_LOG_FILE)
        if not df_log.empty:
            last_retrain = pd.to_datetime(df_log["timestamp"].iloc[-1])
            logger.info(f"Last retrain date: {last_retrain.date()}")

    # ======================================
    # 2️⃣ Export New IoT Data
    # ======================================

    logger.info("Exporting new IoT data...")
    df_new = await export_new_iot_data(since=last_retrain)

    if df_new.empty:
        logger.warning("No new data available. Retrain skipped.")
        return

    logger.info(f"New data shape: {df_new.shape}")

    # ======================================
    # 3️⃣ Retrain Models
    # ======================================

    for target, model_key in MODEL_KEY_MAP.items():

        logger.info("-" * 50)
        logger.info(f"Training target: {target}")

        try:
            train_model(
                df=df_new,
                target_column=target,
                model_key=model_key,
                evaluate_only=evaluate_only
            )

            logger.info(f"✅ Completed training for {target}")

        except Exception as e:
            logger.error(f"❌ Failed training for {target}: {str(e)}")

    # ======================================
    # 4️⃣ Log Retrain Timestamp
    # ======================================

    now = datetime.utcnow()

    if RETRAIN_LOG_FILE.exists():
        df_log = pd.read_csv(RETRAIN_LOG_FILE)
    else:
        df_log = pd.DataFrame(columns=["timestamp"])

    df_log.loc[len(df_log)] = [now.isoformat()]
    df_log.to_csv(RETRAIN_LOG_FILE, index=False)

    logger.info("Retrain log updated.")
    logger.info("🎯 Monthly retrain job completed.")


# ==========================================
# CLI ENTRY POINT
# ==========================================

if __name__ == "__main__":
    asyncio.run(run_retrain_job())
