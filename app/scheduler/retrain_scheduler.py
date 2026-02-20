import argparse
import logging
import os
import sys
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

from app.pipeline.config import (
    TARGET_COLS, SEQUENCE_LENGTH, PREDICTION_HORIZON,
    EPOCHS, BATCH_SIZE, TRAIN_SPLIT, MODEL_DIR, get_model_path
)
from app.pipeline.feature_engineer import build_features, load_feature_cols
from app.pipeline.sequence_builder import build_sequences_for_target, _scaler_X_path
from app.model.lstm import load_lstm, save_lstm, get_callbacks

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _get_env(key: str, default: str) -> str:
    value = os.getenv(key, "").strip()
    if not value:
        logger.warning(f"Env var '{key}' is empty or missing, using default: {default!r}")
        return default
    return value


def _get_env_int(key: str, default: int) -> int:
    value = _get_env(key, str(default))
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Env var '{key}' has invalid value {value!r}, using default: {default}")
        return default


DB_HOST          = _get_env("DB_HOST",          "")
DB_PORT          = _get_env("DB_PORT",          "3306")
DB_NAME          = _get_env("DB_NAME",          "")
DB_USER          = _get_env("DB_USER",          "")
DB_PASS          = _get_env("DB_PASS",          "")
DATABASE_URL     = _get_env("DATABASE_URL",     "")
MIN_NEW_ROWS     = _get_env_int("MIN_NEW_ROWS",     720)
DATA_FETCH_HOURS = _get_env_int("DATA_FETCH_HOURS", 336)
CA_CERT_PATH     = _get_env(
    "CA_CERT_PATH",
    str(Path(__file__).resolve().parent.parent.parent / "ca.pem")
)

if not DATABASE_URL:
    if not all([DB_HOST, DB_NAME, DB_USER, DB_PASS]):
        logger.error(
            "Missing required DB env vars — "
            f"DB_HOST={'SET' if DB_HOST else 'MISSING'}, "
            f"DB_NAME={'SET' if DB_NAME else 'MISSING'}, "
            f"DB_USER={'SET' if DB_USER else 'MISSING'}, "
            f"DB_PASS={'SET' if DB_PASS else 'MISSING'}"
        )
        sys.exit(1)
    DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def _test_engine(engine) -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.warning(f"Connection test failed: {e}")
        return False


def get_engine():
    ca_path = Path(CA_CERT_PATH)

    if ca_path.exists():
        logger.info(f"Trying SSL with CA cert: {ca_path}")
        engine = create_engine(
            DATABASE_URL,
            connect_args={
                "ssl_ca":              str(ca_path),
                "ssl_verify_cert":     True,
                "ssl_verify_identity": False,
            }
        )
        if _test_engine(engine):
            logger.info("Connected via CA cert + verify")
            return engine

        logger.info("Trying CA cert without verification")
        engine = create_engine(
            DATABASE_URL,
            connect_args={
                "ssl_ca":          str(ca_path),
                "ssl_verify_cert": False,
            }
        )
        if _test_engine(engine):
            logger.info("Connected via CA cert, no verify")
            return engine

    logger.info("Trying SSL without cert")
    engine = create_engine(
        DATABASE_URL,
        connect_args={
            "ssl_disabled":        False,
            "ssl_verify_cert":     False,
            "ssl_verify_identity": False,
        }
    )
    if _test_engine(engine):
        logger.info("Connected via SSL, no cert")
        return engine

    logger.error("All SSL connection methods failed.")
    sys.exit(1)


def load_last_30_days() -> pd.DataFrame | None:
    try:
        engine = get_engine()
        since  = (datetime.utcnow() - timedelta(hours=DATA_FETCH_HOURS)).strftime("%Y-%m-%d %H:%M:%S")
        query  = f"""
            SELECT recorded_at, co2_density, temperature_c, humidity
            FROM sensor_data
            WHERE recorded_at >= '{since}'
            ORDER BY recorded_at ASC
        """
        df = pd.read_sql(query, engine)
        logger.info(f"Loaded {len(df):,} rows from sensor_data (last {DATA_FETCH_HOURS}h)")

        df = df.rename(columns={
            "recorded_at": "timestamp",
            "co2_density": "co2_ppm",
            "humidity":    "humidity_percent",
        })
        df["timestamp"]        = pd.to_datetime(df["timestamp"])
        df["co2_ppm"]          = pd.to_numeric(df["co2_ppm"],          errors="coerce")
        df["temperature_c"]    = pd.to_numeric(df["temperature_c"],    errors="coerce")
        df["humidity_percent"] = pd.to_numeric(df["humidity_percent"], errors="coerce")
        df = df.dropna(subset=["co2_ppm", "temperature_c", "humidity_percent"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        logger.info(f"Clean rows after dropna: {len(df):,}")
        return df

    except Exception as e:
        logger.error(f"Failed to load data from DB: {e}", exc_info=True)
        return None


def run_evaluate(df: pd.DataFrame):
    logger.info("Starting evaluation")

    try:
        feature_cols = load_feature_cols()
    except Exception as e:
        logger.error(f"Failed to load feature_cols: {e}", exc_info=True)
        sys.exit(1)

    try:
        df_feat = build_features(df)
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}", exc_info=True)
        sys.exit(1)

    results = {}
    for target in TARGET_COLS:
        logger.info(f"Evaluating: {target}")
        try:
            with open(_scaler_X_path(target), "rb") as f:
                scaler_X = pickle.load(f)
            with open(get_model_path(f"scaler_{target}"), "rb") as f:
                scaler_y = pickle.load(f)

            data   = build_sequences_for_target(
                df_feat, feature_cols, target,
                fit=False, scaler_X=scaler_X, scaler_y=scaler_y,
            )
            X_test = data["X_test"]
            y_test = data["y_test"]

            if len(X_test) == 0:
                logger.warning(f"No test sequences for {target}, skipping.")
                continue

            model         = load_lstm(target)
            y_pred_scaled = model.predict(X_test, verbose=0)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(y_pred_scaled.shape)
            y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

            mae  = float(np.mean(np.abs(y_pred - y_true)))
            rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
            results[target] = {"MAE": round(mae, 4), "RMSE": round(rmse, 4)}
            logger.info(f"{target}: MAE={mae:.4f}  RMSE={rmse:.4f}")

        except Exception as e:
            logger.error(f"Evaluation failed for {target}: {e}", exc_info=True)

    logger.info("Evaluation summary:")
    for t, m in results.items():
        logger.info(f"  {t:25s}  MAE={m['MAE']}  RMSE={m['RMSE']}")


def run_finetune(df: pd.DataFrame):
    logger.info("Starting fine-tune")

    if len(df) < MIN_NEW_ROWS:
        logger.warning(f"Only {len(df)} rows available, minimum is {MIN_NEW_ROWS}. Aborting.")
        sys.exit(1)

    try:
        feature_cols = load_feature_cols()
    except Exception as e:
        logger.error(f"Failed to load feature_cols: {e}", exc_info=True)
        sys.exit(1)

    try:
        df_feat = build_features(df)
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}", exc_info=True)
        sys.exit(1)

    for target in TARGET_COLS:
        logger.info(f"Fine-tuning: {target}")
        try:
            with open(_scaler_X_path(target), "rb") as f:
                scaler_X = pickle.load(f)
            with open(get_model_path(f"scaler_{target}"), "rb") as f:
                scaler_y = pickle.load(f)

            data    = build_sequences_for_target(
                df_feat, feature_cols, target,
                fit=False, scaler_X=scaler_X, scaler_y=scaler_y,
            )
            X_train = data["X_train"]
            y_train = data["y_train"]
            X_val   = data["X_val"]
            y_val   = data["y_val"]

            if len(X_train) == 0:
                logger.warning(f"No training sequences for {target}, skipping.")
                continue

            logger.info(f"Sequences — train: {len(X_train):,}, val: {len(X_val):,}")

            model           = load_lstm(target)
            callbacks       = get_callbacks(target)
            FINETUNE_EPOCHS = min(50, EPOCHS)

            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=FINETUNE_EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=callbacks,
                verbose=1,
            )

            save_lstm(model, target)
            logger.info(f"Fine-tune saved: {target}")

        except Exception as e:
            logger.error(f"Fine-tune failed for {target}: {e}", exc_info=True)
            sys.exit(1)

    _save_retrain_history(df)
    logger.info("All targets fine-tuned successfully.")


def _save_retrain_history(df: pd.DataFrame):
    history_path = MODEL_DIR / "retrain_history.csv"
    record = {
        "retrain_date": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "rows_used":    len(df),
        "date_from":    str(df["timestamp"].min()),
        "date_to":      str(df["timestamp"].max()),
    }
    history_df = pd.DataFrame([record])
    if history_path.exists():
        history_df.to_csv(history_path, mode="a", header=False, index=False)
    else:
        history_df.to_csv(history_path, index=False)
    logger.info(f"Retrain history updated: {history_path}")


def main():
    parser = argparse.ArgumentParser(description="Monthly retrain scheduler")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--run-now",       action="store_true")
    args = parser.parse_args()

    if not args.evaluate_only and not args.run_now:
        logger.error("No mode specified. Use --evaluate-only or --run-now.")
        sys.exit(1)

    df = load_last_30_days()
    if df is None or df.empty:
        logger.warning("No data loaded from DB. Exiting.")
        sys.exit(1)

    if args.evaluate_only:
        run_evaluate(df)
    elif args.run_now:
        run_finetune(df)


if __name__ == "__main__":
    main()