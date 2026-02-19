"""
Monthly retrain scheduler (MySQL version)
=========================================

Retrains or evaluates AI model using data from the sensor_data table.
"""

import argparse
import logging
import sys
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine
import os

# Replace with your actual model module
from app.model.train import train_model, evaluate_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# --- Config ---
MODEL_DIR = Path(os.getenv("MODEL_DIR", "saved_models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --- Database connection (MySQL) ---
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    if not all([DB_HOST, DB_NAME, DB_USER, DB_PASS]):
        logging.error("Missing required DB environment variables: DB_HOST, DB_NAME, DB_USER, DB_PASS")
        sys.exit(1)
    DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def load_data() -> pd.DataFrame | None:
    """Load sensor data from MySQL sensor_data table."""
    try:
        engine = create_engine(DATABASE_URL)
        query = "SELECT * FROM sensor_data"
        df = pd.read_sql(query, engine)
        logging.info(f"Loaded {len(df)} rows from sensor_data table")
        return df
    except Exception as e:
        logging.error(f"Failed to load sensor data from DB: {e}")
        return None


def retrain(df: pd.DataFrame):
    """Run full model retraining."""
    logging.info("Starting full retrain...")
    try:
        train_model(df, save_dir=MODEL_DIR)
        logging.info("Retrain complete!")
    except Exception as e:
        logging.error(f"Retrain failed: {e}")
        sys.exit(1)


def evaluate(df: pd.DataFrame):
    """Run model evaluation only."""
    logging.info("Starting model evaluation...")
    try:
        evaluate_model(df, model_dir=MODEL_DIR)
        logging.info("Evaluation complete!")
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Monthly retrain scheduler")
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Evaluate models without retraining",
    )
    parser.add_argument(
        "--run-now",
        action="store_true",
        help="Force full retraining immediately",
    )
    args = parser.parse_args()

    if not args.evaluate_only and not args.run_now:
        logging.error("No mode specified. Use --evaluate-only or --run-now.")
        sys.exit(1)

    df = load_data()
    if df is None or df.empty:
        logging.warning("No data available for training or evaluation. Exiting.")
        sys.exit(1)

    if args.evaluate_only:
        evaluate(df)
    elif args.run_now:
        retrain(df)


if __name__ == "__main__":
    main()