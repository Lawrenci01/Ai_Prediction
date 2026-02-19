"""
Automatic Monthly Retraining Scheduler
========================================
Runs on the 1st of every month at 2:00 AM.
Fetches new IoT data from DB, merges with historical data,
retrains LSTM models, evaluates, and replaces if better.

Flow:
  1. Export new IoT data from DB since last retrain
  2. Merge with existing training CSV
  3. Retrain all 3 LSTM models
  4. Evaluate new vs old model on holdout set
  5. If new model R² > old model R² → replace model files
  6. Log results to retrain_history table

Usage:
    python retrain_scheduler.py --run-now        # retrain immediately
    python retrain_scheduler.py                  # start monthly loop
    python retrain_scheduler.py --evaluate-only  # just compare models
"""

import argparse
import asyncio
import logging
import os
import shutil
import sys
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.pipeline.config import (
    DATA_FILE, TARGET_COLS, TIMESTAMP_COL,
    MODEL_DIR, SEQUENCE_LENGTH, PREDICTION_HORIZON
)
from app.pipeline.feature_engineer import build_features
from app.database.connection import AsyncSessionLocal
from app.database.repository import fetch_recent_sensor_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

RETRAIN_DAY      = int(os.getenv("RETRAIN_DAY",    "1"))   # day of month
RETRAIN_HOUR     = int(os.getenv("RETRAIN_HOUR",   "2"))   # 2 AM
RETRAIN_MINUTE   = int(os.getenv("RETRAIN_MINUTE", "0"))
MIN_NEW_ROWS     = int(os.getenv("MIN_NEW_ROWS",   "720")) # at least 30 days new data
BACKUP_DIR       = MODEL_DIR / "backups"
RETRAIN_LOG_FILE = MODEL_DIR / "retrain_history.csv"


# ============================================================================
# STEP 1 — Export new IoT data from DB
# ============================================================================

async def export_new_iot_data(since: datetime = None) -> pd.DataFrame:
    """Fetch all new sensor data from DB since last retrain."""
    if since is None:
        since = datetime.utcnow() - timedelta(days=60)  # default: last 60 days

    logger.info(f"Fetching new IoT data from DB since {since.date()}...")

    async with AsyncSessionLocal() as db:
        from sqlalchemy import select, text
        from app.database.models import SensorData

        result = await db.execute(
            select(
                SensorData.minute_stamp,
                SensorData.co2_density,
                SensorData.temperature_c,
                SensorData.humidity,
            )
            .where(SensorData.minute_stamp >= since)
            .where(SensorData.co2_density.isnot(None))
            .where(SensorData.temperature_c.isnot(None))
            .where(SensorData.humidity.isnot(None))
            .order_by(SensorData.minute_stamp.asc())
        )
        rows = result.fetchall()

    if not rows:
        logger.warning("No new IoT data found in DB.")
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["timestamp", "co2_ppm", "temperature_c", "humidity_percent"])
    df["timestamp"]        = pd.to_datetime(df["timestamp"])
    df["co2_ppm"]          = pd.to_numeric(df["co2_ppm"],          errors="coerce")
    df["temperature_c"]    = pd.to_numeric(df["temperature_c"],    errors="coerce")
    df["humidity_percent"] = pd.to_numeric(df["humidity_percent"], errors="coerce")
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)

    # Resample to hourly (model was trained on hourly data)
    df = df.set_index("timestamp")
    df = df.resample("1h").mean().dropna().reset_index()
    # reset_index() already restores "timestamp" as column name — no rename needed

    logger.info(f"Exported {len(df):,} hourly rows from DB.")
    return df


# ============================================================================
# STEP 2 — Merge with historical training data
# ============================================================================

def merge_with_historical(df_new: pd.DataFrame) -> pd.DataFrame:
    """Merge new IoT data with existing historical CSV."""
    logger.info(f"Loading historical data from {DATA_FILE}...")

    if not Path(DATA_FILE).exists():
        logger.warning("No historical CSV found — using new data only.")
        return df_new

    df_hist = pd.read_csv(DATA_FILE)
    df_hist[TIMESTAMP_COL] = pd.to_datetime(df_hist[TIMESTAMP_COL])
    df_hist = df_hist.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    # Rename new data columns to match historical
    df_new_renamed = df_new.rename(columns={"timestamp": TIMESTAMP_COL})

    # Combine and deduplicate by timestamp
    df_combined = pd.concat([df_hist, df_new_renamed], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=[TIMESTAMP_COL], keep="last")
    df_combined = df_combined.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    logger.info(
        f"Historical: {len(df_hist):,} rows | "
        f"New: {len(df_new_renamed):,} rows | "
        f"Combined: {len(df_combined):,} rows"
    )
    return df_combined


# ============================================================================
# STEP 3 — Evaluate current model
# ============================================================================

def evaluate_model(model, scaler_X, scaler_y, df_featured: pd.DataFrame,
                   target: str, feat_cols: list) -> dict:
    """Evaluate model R² and RMSE on holdout set (last 20% of data)."""
    from sklearn.metrics import r2_score, mean_squared_error

    n         = len(df_featured)
    holdout_n = int(n * 0.2)
    df_test   = df_featured.iloc[-holdout_n:].reset_index(drop=True)

    if len(df_test) < SEQUENCE_LENGTH + PREDICTION_HORIZON:
        return {"r2": None, "rmse": None, "samples": 0}

    X_vals = df_test[feat_cols].values
    y_vals = df_test[target].values

    y_true, y_pred = [], []
    for i in range(SEQUENCE_LENGTH, len(X_vals) - PREDICTION_HORIZON, PREDICTION_HORIZON):
        X_seq    = X_vals[i - SEQUENCE_LENGTH:i]
        X_scaled = scaler_X.transform(X_seq)
        X_inp    = X_scaled.reshape(1, SEQUENCE_LENGTH, -1)
        y_scaled = model.predict(X_inp, verbose=0)
        val      = float(scaler_y.inverse_transform([[y_scaled[0, 0]]])[0][0])
        y_pred.append(val)
        y_true.append(float(y_vals[i]))

    if not y_true:
        return {"r2": None, "rmse": None, "samples": 0}

    r2   = round(r2_score(y_true, y_pred), 4)
    rmse = round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4)
    return {"r2": r2, "rmse": rmse, "samples": len(y_true)}


# ============================================================================
# STEP 4 — Retrain models
# ============================================================================

def retrain_models(df_combined: pd.DataFrame) -> bool:
    """Retrain all 3 LSTM models using combined historical + new data."""
    logger.info("Starting retraining pipeline...")

    # Save combined data to temp CSV for train.py to use
    temp_csv = MODEL_DIR / "retrain_temp.csv"
    df_combined.to_csv(temp_csv, index=False)
    logger.info(f"Saved combined data to {temp_csv} ({len(df_combined):,} rows)")

    import subprocess
    results = {}

    for target in TARGET_COLS:
        logger.info(f"Retraining model for {target}...")
        cmd = [
            sys.executable, "training/train.py",
            "--target", target,
            "--force-retrain",
            "--data-file", str(temp_csv),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).parent))
        if result.returncode != 0:
            logger.error(f"Retraining failed for {target}:\n{result.stderr}")
            results[target] = False
        else:
            logger.info(f"Retraining complete for {target}")
            results[target] = True

    temp_csv.unlink(missing_ok=True)
    return all(results.values())


# ============================================================================
# STEP 5 — Backup + Replace model files
# ============================================================================

def backup_current_models():
    """Backup existing model files before replacing."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / timestamp
    backup_path.mkdir(parents=True, exist_ok=True)

    model_files = list(MODEL_DIR.glob("*.keras")) + \
                  list(MODEL_DIR.glob("*.pkl"))

    for f in model_files:
        shutil.copy2(f, backup_path / f.name)

    logger.info(f"Backed up {len(model_files)} model files to {backup_path}")
    return backup_path


def restore_backup(backup_path: Path):
    """Restore previous model files if new model is worse."""
    logger.warning(f"Restoring backup from {backup_path}...")
    for f in backup_path.glob("*"):
        shutil.copy2(f, MODEL_DIR / f.name)
    logger.info("Backup restored successfully.")


# ============================================================================
# STEP 6 — Log retrain results
# ============================================================================

def log_retrain_result(results: dict):
    """Append retrain results to CSV log."""
    row = {
        "timestamp":        datetime.now().isoformat(),
        "status":           results.get("status", "unknown"),
        "new_rows":         results.get("new_rows", 0),
        "total_rows":       results.get("total_rows", 0),
    }
    for target in TARGET_COLS:
        old = results.get("old_metrics", {}).get(target, {})
        new = results.get("new_metrics", {}).get(target, {})
        row[f"{target}_old_r2"]   = old.get("r2")
        row[f"{target}_new_r2"]   = new.get("r2")
        row[f"{target}_old_rmse"] = old.get("rmse")
        row[f"{target}_new_rmse"] = new.get("rmse")
        row[f"{target}_improved"] = (
            new.get("r2", 0) > old.get("r2", 0)
            if new.get("r2") and old.get("r2") else None
        )

    df_log = pd.DataFrame([row])
    if RETRAIN_LOG_FILE.exists():
        df_log.to_csv(RETRAIN_LOG_FILE, mode="a", header=False, index=False)
    else:
        df_log.to_csv(RETRAIN_LOG_FILE, index=False)

    logger.info(f"Retrain result logged to {RETRAIN_LOG_FILE}")


# ============================================================================
# MAIN RETRAIN JOB
# ============================================================================

async def run_retrain_job(evaluate_only: bool = False):
    logger.info("=" * 60)
    logger.info(f"🔄 Monthly retrain job — {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # Explicit model key map — avoids fragile substring matching
    MODEL_KEY_MAP = {
        "co2_ppm":          "lstm_co2",
        "temperature_c":    "lstm_temperature",
        "humidity_percent": "lstm_humidity",
    }

    # 1. Get last retrain date
    last_retrain = datetime.utcnow() - timedelta(days=60)
    if RETRAIN_LOG_FILE.exists():
        df_log = pd.read_csv(RETRAIN_LOG_FILE)
        if not df_log.empty:
            last_retrain = pd.to_datetime(df_log["timestamp"].iloc[-1])
            logger.info(f"Last retrain: {last_retrain.date()}")

    # 2. Export new data
    df_new = await export_new_iot_data(since=last_retrain)

    # 3. For evaluate-only: run even if data is below MIN_NEW_ROWS threshold
    #    This lets you test model evaluation at any time without needing a full month of data
    if evaluate_only:
        if df_new.empty:
            logger.warning("No new data found — cannot evaluate without any data.")
            return False
        df_combined = merge_with_historical(df_new)
        logger.info("Building features for evaluation...")
        df_featured = build_features(df_combined, drop_nan=True)
        feat_cols_path = MODEL_DIR / "feature_cols.pkl"
        with open(feat_cols_path, "rb") as f:
            feat_cols = pickle.load(f)

        from app.inference import _load_scaler_X, registry
        logger.info("Evaluating current models (evaluate-only mode)...")
        for target in TARGET_COLS:
            try:
                model_key = MODEL_KEY_MAP[target]
                model     = registry.get(model_key)
                scaler_X  = _load_scaler_X(target)
                scaler_y_path = MODEL_DIR / f"scaler_{target}.pkl"
                with open(scaler_y_path, "rb") as f:
                    scaler_y = pickle.load(f)
                metrics = evaluate_model(model, scaler_X, scaler_y, df_featured, target, feat_cols)
                logger.info(f"  {target}: R²={metrics['r2']}  RMSE={metrics['rmse']}  samples={metrics['samples']}")
            except Exception as e:
                logger.error(f"  Could not evaluate {target}: {e}")
        logger.info("Evaluate-only mode — skipping retraining.")
        return True

    # 4. Full retrain: enforce minimum new data requirement
    if df_new.empty or len(df_new) < MIN_NEW_ROWS:
        logger.warning(
            f"Not enough new data to retrain: {len(df_new)} rows "
            f"(need {MIN_NEW_ROWS}). Skipping."
        )
        return False

    # 5. Merge with historical
    df_combined = merge_with_historical(df_new)

    # 6. Feature engineering for evaluation
    logger.info("Building features for evaluation...")
    df_featured = build_features(df_combined, drop_nan=True)
    feat_cols_path = MODEL_DIR / "feature_cols.pkl"
    with open(feat_cols_path, "rb") as f:
        feat_cols = pickle.load(f)

    # 7. Evaluate CURRENT models
    logger.info("Evaluating current models...")
    old_metrics = {}
    from tensorflow.keras.models import load_model as keras_load
    from app.inference import _load_scaler_X, registry

    for target in TARGET_COLS:
        try:
            model_key = MODEL_KEY_MAP[target]
            model    = registry.get(model_key)
            scaler_X = _load_scaler_X(target)
            scaler_y_path = MODEL_DIR / f"scaler_{target}.pkl"
            with open(scaler_y_path, "rb") as f:
                scaler_y = pickle.load(f)

            metrics = evaluate_model(model, scaler_X, scaler_y, df_featured, target, feat_cols)
            old_metrics[target] = metrics
            logger.info(f"  Current {target}: R²={metrics['r2']}  RMSE={metrics['rmse']}")
        except Exception as e:
            logger.error(f"  Could not evaluate current {target}: {e}")
            old_metrics[target] = {"r2": None, "rmse": None}

    if evaluate_only:
        logger.info("Evaluate-only mode — skipping retraining.")
        return True

    # 8. Backup current models
    backup_path = backup_current_models()

    # 9. Retrain
    logger.info("Retraining models with combined data...")
    success = retrain_models(df_combined)

    if not success:
        logger.error("Retraining failed — restoring backup.")
        restore_backup(backup_path)
        log_retrain_result({
            "status":      "failed",
            "new_rows":    len(df_new),
            "total_rows":  len(df_combined),
            "old_metrics": old_metrics,
            "new_metrics": {},
        })
        return False

    # 10. Evaluate NEW models
    logger.info("Evaluating new models...")
    registry.reload_all()  # clear cache to load new models
    new_metrics = {}

    for target in TARGET_COLS:
        try:
            model_key = MODEL_KEY_MAP[target]
            model    = registry.get(model_key)
            scaler_X = _load_scaler_X(target)
            scaler_y_path = MODEL_DIR / f"scaler_{target}.pkl"
            with open(scaler_y_path, "rb") as f:
                scaler_y = pickle.load(f)

            metrics = evaluate_model(model, scaler_X, scaler_y, df_featured, target, feat_cols)
            new_metrics[target] = metrics
            old_r2 = old_metrics.get(target, {}).get("r2", "N/A")
            logger.info(
                f"  New {target}: R²={metrics['r2']}  RMSE={metrics['rmse']}  "
                f"(was R²={old_r2})"
            )
        except Exception as e:
            logger.error(f"  Could not evaluate new {target}: {e}")
            new_metrics[target] = {"r2": None, "rmse": None}

    # 11. Compare — restore if worse
    any_improved = False
    any_worse    = False
    for target in TARGET_COLS:
        old_r2 = old_metrics.get(target, {}).get("r2")
        new_r2 = new_metrics.get(target, {}).get("r2")
        if old_r2 and new_r2:
            if new_r2 >= old_r2 - 0.01:  # allow 0.01 tolerance
                any_improved = True
                logger.info(f"  ✅ {target}: improved or maintained ({old_r2} → {new_r2})")
            else:
                any_worse = True
                logger.warning(f"  ⚠️  {target}: degraded ({old_r2} → {new_r2})")

    if any_worse and not any_improved:
        logger.warning("All models degraded — restoring backup.")
        restore_backup(backup_path)
        status = "restored"
    else:
        logger.info("✅ New models accepted.")
        status = "improved" if any_improved else "unchanged"

    # 12. Log result
    log_retrain_result({
        "status":      status,
        "new_rows":    len(df_new),
        "total_rows":  len(df_combined),
        "old_metrics": old_metrics,
        "new_metrics": new_metrics,
    })

    logger.info("=" * 60)
    logger.info(f"✅ Retrain job complete — status: {status}")
    logger.info("=" * 60)
    return True


# ============================================================================
# SCHEDULER LOOP — runs 1st of every month
# ============================================================================

async def retrain_loop():
    logger.info(f"📅 Retrain scheduler — runs day {RETRAIN_DAY} of month at {RETRAIN_HOUR:02d}:{RETRAIN_MINUTE:02d} UTC")

    while True:
        now = datetime.utcnow()

        # Find next retrain date
        next_run = now.replace(
            day=RETRAIN_DAY, hour=RETRAIN_HOUR,
            minute=RETRAIN_MINUTE, second=0, microsecond=0
        )
        if next_run <= now:
            # Move to next month
            if now.month == 12:
                next_run = next_run.replace(year=now.year + 1, month=1)
            else:
                next_run = next_run.replace(month=now.month + 1)

        wait = (next_run - now).total_seconds()
        logger.info(f"⏰ Next retrain: {next_run.date()} ({wait/3600/24:.1f} days away)")
        await asyncio.sleep(wait)
        await run_retrain_job()


# ============================================================================
# ENTRYPOINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-now",       action="store_true", help="Retrain immediately")
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate current models only")
    args = parser.parse_args()

    if args.run_now or args.evaluate_only:
        asyncio.run(run_retrain_job(evaluate_only=args.evaluate_only))
    else:
        asyncio.run(retrain_loop())