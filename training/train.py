"""
Main Training Script
=====================
Trains LSTM models for CO2, temperature, and humidity prediction.
RF and XGBoost are disabled — LSTM only.

Usage:
    python training/train.py --target temperature_c --force-retrain
    python training/train.py --target humidity_percent --force-retrain
    python training/train.py --target co2_ppm --force-retrain
    python training/train.py --target all --force-retrain
    python training/train.py --target co2_ppm --force-retrain --bidirectional

Recommended order:
    1. temperature_c
    2. humidity_percent
    3. co2_ppm
"""

import argparse
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    precision_score, recall_score, f1_score, confusion_matrix
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline.config import (
    DATA_FILE, TARGET_COLS, TIMESTAMP_COL,
    EPOCHS, BATCH_SIZE, get_model_path, LOG_FORMAT, LOG_LEVEL, MODEL_DIR,
    LSTM_UNITS
)
from app.pipeline.feature_engineer import (
    build_features, get_feature_columns, save_feature_cols
)
from app.pipeline.sequence_builder import build_sequences_for_target

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

METRICS_REPORT_PATH = MODEL_DIR / "metrics_report.txt"


# ============================================================================
# BINS
# ============================================================================

TARGET_BINS = {
    "co2_ppm": {
        "bins":   [0, 380, 400, 420, 440, float("inf")],
        "labels": ["Very Low (<380)", "Low (380-400)", "Normal (400-420)",
                   "High (420-440)", "Very High (>440)"],
    },
    "temperature_c": {
        "bins":   [0, 24, 28, 32, 36, float("inf")],
        "labels": ["Cool (<24)", "Mild (24-28)", "Warm (28-32)",
                   "Hot (32-36)", "Very Hot (>36)"],
    },
    "humidity_percent": {
        "bins":   [0, 40, 60, 75, 90, float("inf")],
        "labels": ["Dry (<40%)", "Comfortable (40-60%)", "Humid (60-75%)",
                   "Very Humid (75-90%)", "Extreme (>90%)"],
    },
}


def bin_values(values: np.ndarray, target: str) -> np.ndarray:
    bins = TARGET_BINS[target]["bins"]
    return np.digitize(values, bins[1:-1])


# ============================================================================
# METRICS
# ============================================================================

def compute_regression_metrics(y_true, y_pred) -> dict:
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100),
    }


def compute_classification_metrics(y_true, y_pred, target) -> dict:
    true_bins = bin_values(y_true.flatten(), target)
    pred_bins = bin_values(y_pred.flatten(), target)
    labels    = list(range(len(TARGET_BINS[target]["labels"])))
    return {
        "precision":        float(precision_score(true_bins, pred_bins, labels=labels, average="weighted", zero_division=0)),
        "recall":           float(recall_score(true_bins, pred_bins, labels=labels, average="weighted", zero_division=0)),
        "f1_score":         float(f1_score(true_bins, pred_bins, labels=labels, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(true_bins, pred_bins, labels=labels).tolist(),
        "bin_labels":       TARGET_BINS[target]["labels"],
    }


def compute_all_metrics(y_true, y_pred, model_name, target, train_time_s=0.0, extra=None) -> dict:
    reg = compute_regression_metrics(y_true, y_pred)
    cls = compute_classification_metrics(y_true, y_pred, target)
    result = {"model": model_name, "target": target, "train_time_s": round(train_time_s, 1), **reg, **cls}
    if extra:
        result.update(extra)
    logger.info(
        f"  [{model_name}] RMSE={reg['rmse']:.4f}  MAE={reg['mae']:.4f}  "
        f"R2={reg['r2']:.4f}  MAPE={reg['mape']:.2f}%  F1={cls['f1_score']:.4f}"
    )
    return result


# ============================================================================
# METRICS REPORT
# ============================================================================

def format_confusion_matrix(cm, labels) -> str:
    col_w  = max(max(len(l) for l in labels), 10)
    header = " " * (col_w + 2) + "  ".join(f"{l:>{col_w}}" for l in labels)
    lines  = [header, "-" * len(header)]
    for i, row in enumerate(cm):
        lines.append(f"{labels[i]:>{col_w}}  " + "  ".join(f"{v:>{col_w}}" for v in row))
    return "\n".join(lines)


def save_metrics_report(all_metrics: dict, report_path: Path):
    now   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = ["=" * 70, "  CLIMATE PREDICTION — TRAINING METRICS REPORT",
             f"  Generated: {now}", "=" * 70]

    for target, model_results in all_metrics.items():
        if not model_results:
            continue
        lines += [f"\n{'='*70}", f"  TARGET: {target.upper()}",
                  f"  Bins: {', '.join(TARGET_BINS[target]['labels'])}", f"{'='*70}"]

        for r in model_results:
            lines += [f"\n  [ {r.get('model')} ]", f"  {'─'*50}",
                      "  REGRESSION",
                      f"    RMSE  : {r.get('rmse', 0):.4f}",
                      f"    MAE   : {r.get('mae',  0):.4f}",
                      f"    R2    : {r.get('r2',   0):.4f}",
                      f"    MAPE  : {r.get('mape', 0):.2f}%",
                      "  CLASSIFICATION",
                      f"    Precision : {r.get('precision', 0):.4f}",
                      f"    Recall    : {r.get('recall',    0):.4f}",
                      f"    F1 Score  : {r.get('f1_score',  0):.4f}",
                      "  TRAINING INFO",
                      f"    Train time : {r.get('train_time_s', 0):.1f}s"]
            if "epochs_trained" in r:
                lines.append(f"    Epochs     : {r['epochs_trained']}")
            cm     = r.get("confusion_matrix")
            labels = r.get("bin_labels", TARGET_BINS[target]["labels"])
            if cm:
                lines.append("  CONFUSION MATRIX")
                for l in format_confusion_matrix(cm, labels).split("\n"):
                    lines.append(f"    {l}")

        valid = [r for r in model_results if "rmse" in r]
        if valid:
            best = min(valid, key=lambda x: x["rmse"])
            lines.append(
                f"\n  Best model: {best['model']}  "
                f"RMSE={best['rmse']:.4f}  R2={best.get('r2', 0):.4f}"
            )

    lines += [f"\n{'='*70}", "  END OF REPORT", "=" * 70]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Metrics report saved: {report_path}")


# ============================================================================
# MODEL CHECKS
# ============================================================================

def is_keras_model_valid(model_key: str) -> bool:
    path = get_model_path(model_key)
    if not Path(path).exists() or Path(path).stat().st_size == 0:
        return False
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(path) is not None
    except Exception:
        return False


def delete_model(model_key: str):
    path = Path(get_model_path(model_key))
    if path.exists():
        path.unlink()
        logger.warning(f"  [CLEANUP] Deleted: {path.name}")


def get_model_key(prefix: str, target: str) -> str:
    if target.startswith("co2"):
        return f"{prefix}_co2"
    elif target.startswith("temp"):
        return f"{prefix}_temperature"
    elif target.startswith("humid"):
        return f"{prefix}_humidity"
    else:
        raise ValueError(f"Unknown target: '{target}'")


# ============================================================================
# LSTM TRAINING
# ============================================================================

def train_lstm_model(df_featured, feature_cols, target,
                     force_retrain=False, bidirectional=False) -> dict:
    try:
        from app.model.lstm import build_lstm_model, get_callbacks
    except ImportError:
        logger.warning("TensorFlow not installed. Skipping LSTM.")
        return {}

    logger.info(f"\n--- LSTM for {target} (bidirectional={bidirectional}) ---")
    lstm_key = get_model_key("lstm", target)

    if not force_retrain and is_keras_model_valid(lstm_key):
        logger.info("  [SKIP] LSTM already valid. Use --force-retrain to override.")
        return {}

    delete_model(lstm_key)

    seq_data    = build_sequences_for_target(df_featured, feature_cols, target, fit=True)
    X_train     = seq_data["X_train"]
    y_train     = seq_data["y_train"]
    X_val       = seq_data["X_val"]
    y_val       = seq_data["y_val"]
    X_test      = seq_data["X_test"]
    y_test      = seq_data["y_test"]
    scaler_y    = seq_data["scaler_y"]
    input_shape = (X_train.shape[1], X_train.shape[2])

    logger.info(
        f"  Input: {input_shape}, train: {len(X_train):,}, "
        f"val: {len(X_val):,}, test: {len(X_test):,}"
    )

    model = build_lstm_model(input_shape, target, bidirectional=bidirectional)

    logger.info(f"  Training (max {EPOCHS} epochs, batch {BATCH_SIZE})...")
    t0      = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks(target),
        verbose=1,
    )
    elapsed = time.time() - t0

    y_pred_scaled = model.predict(X_test, verbose=0)
    n, h   = y_pred_scaled.shape
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(n, h)
    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).reshape(n, h)

    epochs_trained = len(history.history["loss"])
    m = compute_all_metrics(
        y_true, y_pred, "LSTM", target, elapsed,
        extra={"epochs_trained": epochs_trained}
    )
    logger.info(f"  LSTM done in {elapsed:.1f}s ({epochs_trained} epochs)")
    return m


# ============================================================================
# MAIN
# ============================================================================

def run_training(targets=None, train_lstm=True, train_sklearn=False,
                 force_retrain=False, bidirectional=False):

    if targets is None:
        targets = TARGET_COLS

    logger.info("=" * 70)
    logger.info("CLIMATE PREDICTION MODEL TRAINING  —  LSTM ONLY")
    logger.info("=" * 70)
    logger.info(f"Data       : {DATA_FILE}")
    logger.info(f"Targets    : {targets}")
    logger.info(f"Retrain    : {force_retrain}")
    logger.info(f"Batch size : {BATCH_SIZE}")
    logger.info(f"LSTM units : {LSTM_UNITS}")

    logger.info("\nLoading data...")
    df = pd.read_csv(DATA_FILE)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    logger.info(f"Loaded {len(df):,} records.")

    logger.info("Engineering features...")
    df_featured  = build_features(df, drop_nan=True)
    feature_cols = get_feature_columns(df_featured)
    save_feature_cols(feature_cols)
    logger.info(f"Features: {len(feature_cols)} | Rows: {len(df_featured):,}")

    all_results = {}

    for i, target in enumerate(targets, 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"TARGET {i}/{len(targets)}: {target.upper()}")
        logger.info(f"{'='*70}")

        target_results = []

        if train_lstm:
            result = train_lstm_model(
                df_featured, feature_cols, target,
                force_retrain=force_retrain,
                bidirectional=bidirectional
            )
            if result:
                target_results.append(result)

        all_results[target] = target_results
        logger.info(f"\n  Done: {target}")

    # Save ensemble weights (LSTM only)
    weights = {}
    for target, results in all_results.items():
        weights[target] = {"lstm": 1.0}
    with open(get_model_path("ensemble_weights"), "wb") as f:
        pickle.dump(weights, f)

    save_metrics_report(all_results, METRICS_REPORT_PATH)

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE — SUMMARY")
    logger.info("=" * 70)
    for target, results in all_results.items():
        logger.info(f"\n{target.upper()}")
        if not results:
            logger.info("  No new models trained.")
            continue
        for r in results:
            logger.info(
                f"  {r['model']:20s}  RMSE={r['rmse']:.4f}  "
                f"MAE={r['mae']:.4f}  R2={r['r2']:.4f}  MAPE={r['mape']:.2f}%"
            )
    logger.info(f"\nReport: {METRICS_REPORT_PATH}")
    return all_results


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    valid_targets = TARGET_COLS + ["all"]
    parser = argparse.ArgumentParser(description="Train LSTM climate prediction models")
    parser.add_argument("--target",        choices=valid_targets, default="all")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--no-lstm",       action="store_true")
    args = parser.parse_args()

    targets = TARGET_COLS if args.target == "all" else [args.target]
    run_training(
        targets=targets,
        train_lstm=not args.no_lstm,
        train_sklearn=False,
        force_retrain=args.force_retrain,
        bidirectional=args.bidirectional,
    )
