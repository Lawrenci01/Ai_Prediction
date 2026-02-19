"""
LSTM Training Script
=====================
Trains LSTM models for CO2, temperature, and humidity prediction.
Saves models and scalers to saved_models/ directory.

Usage:
    python training/train.py --target all
    python training/train.py --target co2_ppm
    python training/train.py --target temperature_c
    python training/train.py --target humidity_percent
    python training/train.py --target all --force-retrain
    python training/train.py --target all --data-file custom_data.csv
"""

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline.config import (
    DATA_FILE, TARGET_COLS, TIMESTAMP_COL,
    SEQUENCE_LENGTH, PREDICTION_HORIZON,
    MODEL_DIR, get_model_path,
    LOG_FORMAT, LOG_LEVEL
)
from app.pipeline.feature_engineer import (
    build_features, get_feature_columns, save_feature_cols
)
from app.pipeline.sequence_builder import (
    build_sequences, split_sequences,
    fit_scalers, scale_sequences,
    save_scalers
)

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# ============================================================================
# MODEL DEFINITION
# ============================================================================

def build_lstm_model(input_shape: tuple, output_steps: int):
    """Build LSTM model architecture."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        LSTM, Dense, Dropout, BatchNormalization
    )
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(output_steps),
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )
    return model


# ============================================================================
# TRAINING
# ============================================================================

def train_target(target: str, df_featured: pd.DataFrame,
                 feat_cols: list, force_retrain: bool = False) -> dict:
    """Train LSTM for a single target."""
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    )

    model_key = (
        "lstm_co2"         if "co2"   in target else
        "lstm_temperature" if "temp"  in target else
        "lstm_humidity"
    )

    model_path = get_model_path(model_key)

    # Skip if model already exists and not force retrain
    if model_path.exists() and not force_retrain:
        logger.info(f"Model for {target} already exists. Use --force-retrain to retrain.")
        return {"skipped": True, "target": target}

    logger.info(f"\n{'='*60}")
    logger.info(f"Training: {target}")
    logger.info(f"{'='*60}")

    # Build sequences
    X, y = build_sequences(df_featured, target, feat_cols)
    logger.info(f"Sequences: X={X.shape}, y={y.shape}")

    # Split — interleaved to avoid data leakage
    X_train, X_val, X_test, y_train, y_val, y_test = split_sequences(X, y)
    logger.info(
        f"Split — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}"
    )

    # Scale
    scaler_X, scaler_y = fit_scalers(X_train, y_train)
    X_train_s, X_val_s, X_test_s = scale_sequences(scaler_X, X_train, X_val, X_test)
    y_train_s = scaler_y.transform(y_train.reshape(-1, 1)).flatten()
    y_val_s   = scaler_y.transform(y_val.reshape(-1,   1)).flatten()

    # Reshape y for multi-step output
    y_train_s = y_train_s.reshape(-1, 1)
    y_val_s   = y_val_s.reshape(-1, 1)

    # Save scalers
    save_scalers(target, scaler_X, scaler_y)
    logger.info(f"Scalers saved for {target}")

    # Build model
    model = build_lstm_model(
        input_shape=(SEQUENCE_LENGTH, len(feat_cols)),
        output_steps=PREDICTION_HORIZON
    )
    model.summary(print_fn=logger.info)

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
    ]

    # Train
    history = model.fit(
        X_train_s, y_train_s,
        validation_data=(X_val_s, y_val_s),
        epochs=200,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on test set
    y_test_s  = scaler_y.transform(y_test.reshape(-1, 1)).flatten().reshape(-1, 1)
    y_pred_s  = model.predict(X_test_s, verbose=0)
    y_pred    = scaler_y.inverse_transform(y_pred_s)
    y_true    = scaler_y.inverse_transform(y_test_s)

    from sklearn.metrics import r2_score, mean_squared_error
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100

    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTS: {target}")
    logger.info(f"  R²   = {r2:.4f}")
    logger.info(f"  RMSE = {rmse:.4f}")
    logger.info(f"  MAPE = {mape:.2f}%")
    logger.info(f"  Epochs trained: {len(history.history['loss'])}")
    logger.info(f"{'='*60}\n")

    return {
        "target": target,
        "r2":     round(r2,   4),
        "rmse":   round(rmse, 4),
        "mape":   round(mape, 2),
        "epochs": len(history.history["loss"]),
    }


# ============================================================================
# MAIN
# ============================================================================

def run_training(targets: list, data_file: str = None,
                 force_retrain: bool = False):
    logger.info("=" * 60)
    logger.info("LSTM TRAINING PIPELINE")
    logger.info("=" * 60)

    # Load data
    csv_path = Path(data_file) if data_file else DATA_FILE
    logger.info(f"Loading data from: {csv_path}")
    df_raw = pd.read_csv(csv_path)
    df_raw[TIMESTAMP_COL] = pd.to_datetime(df_raw[TIMESTAMP_COL])
    df_raw = df_raw.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    logger.info(f"Loaded {len(df_raw):,} rows")

    # Feature engineering
    logger.info("Running feature engineering...")
    df_featured = build_features(df_raw, drop_nan=True)

    # Get and save feature columns
    feat_cols = get_feature_columns(df_featured)
    save_feature_cols(feat_cols)
    logger.info(f"Feature columns: {len(feat_cols)}")

    # Create model directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Train each target
    results = []
    for target in targets:
        try:
            result = train_target(target, df_featured, feat_cols, force_retrain)
            results.append(result)
        except Exception as e:
            logger.error(f"Training failed for {target}: {e}")
            results.append({"target": target, "error": str(e)})

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    for r in results:
        if r.get("skipped"):
            logger.info(f"  {r['target']}: SKIPPED (already exists)")
        elif r.get("error"):
            logger.info(f"  {r['target']}: FAILED — {r['error']}")
        else:
            logger.info(
                f"  {r['target']}: R²={r['r2']}  "
                f"RMSE={r['rmse']}  MAPE={r['mape']}%  "
                f"Epochs={r['epochs']}"
            )
    logger.info("=" * 60)
    return results


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM climate models")
    parser.add_argument(
        "--target",
        default="all",
        choices=["all"] + TARGET_COLS,
        help="Which target to train (default: all)"
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retrain even if model already exists"
    )
    parser.add_argument(
        "--data-file",
        default=None,
        help="Path to custom data CSV (default: uses DATA_FILE from config)"
    )
    args = parser.parse_args()

    targets = TARGET_COLS if args.target == "all" else [args.target]

    results = run_training(
        targets       = targets,
        data_file     = args.data_file,
        force_retrain = args.force_retrain,
    )

    # ── Exit code for retrain_scheduler.py ──────────────────────────────────
    # retrain_scheduler checks returncode != 0 to detect failures.
    # Exit 1 if ANY target failed (not skipped — skipped is intentional).
    any_failed = any(r.get("error") for r in results)
    sys.exit(1 if any_failed else 0)