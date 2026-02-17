"""
Sequence Builder
=================
Builds sliding-window sequences for LSTM training and inference.
"""

import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.preprocessing import RobustScaler

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from app.pipeline.config import (
    SEQUENCE_LENGTH, PREDICTION_HORIZON,
    TRAIN_SPLIT, TARGET_COLS, get_model_path
)

logger = logging.getLogger(__name__)


def make_sequences(X_scaled: np.ndarray,
                   y_scaled: np.ndarray,
                   seq_len: int = SEQUENCE_LENGTH,
                   horizon: int = PREDICTION_HORIZON) -> tuple:
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X_scaled) - horizon + 1):
        X_seq.append(X_scaled[i - seq_len: i])
        y_seq.append(y_scaled[i: i + horizon])
    return np.array(X_seq), np.array(y_seq)


def time_split(X_seq: np.ndarray,
               y_seq: np.ndarray,
               train_ratio: float = TRAIN_SPLIT,
               val_ratio: float = 0.1) -> dict:
    n         = len(X_seq)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))
    return {
        "X_train": X_seq[:train_end],
        "y_train": y_seq[:train_end],
        "X_val":   X_seq[train_end:val_end],
        "y_val":   y_seq[train_end:val_end],
        "X_test":  X_seq[val_end:],
        "y_test":  y_seq[val_end:],
        "train_end": train_end,
        "val_end":   val_end,
    }


def build_sequences_for_target(df_featured: pd.DataFrame,
                                feature_cols: list,
                                target: str,
                                fit: bool = True,
                                scaler_X=None,
                                scaler_y=None) -> dict:
    logger.info(f"Building sequences for target: {target}")

    X_raw = df_featured[feature_cols].values
    y_raw = df_featured[target].values

    if fit:
        scaler_X = RobustScaler()
        scaler_y = RobustScaler()
        X_scaled = scaler_X.fit_transform(X_raw)
        y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()

        with open(get_model_path("scaler_X"), "wb") as f:
            pickle.dump(scaler_X, f)
        with open(get_model_path(f"scaler_{target}"), "wb") as f:
            pickle.dump(scaler_y, f)

        logger.info("RobustScaler fitted and saved.")
    else:
        if scaler_X is None or scaler_y is None:
            raise ValueError("Provide scaler_X and scaler_y when fit=False")
        X_scaled = scaler_X.transform(X_raw)
        y_scaled = scaler_y.transform(y_raw.reshape(-1, 1)).flatten()

    X_seq, y_seq = make_sequences(X_scaled, y_scaled)
    logger.info(f"Sequences: X={X_seq.shape}, y={y_seq.shape}")

    splits = time_split(X_seq, y_seq)
    logger.info(
        f"Split sizes — train: {len(splits['X_train']):,}, "
        f"val: {len(splits['X_val']):,}, "
        f"test: {len(splits['X_test']):,}"
    )

    return {
        **splits,
        "scaler_X":   scaler_X,
        "scaler_y":   scaler_y,
        "n_features": X_scaled.shape[1],
        "seq_len":    SEQUENCE_LENGTH,
        "target":     target,
    }


def load_scalers() -> tuple:
    with open(get_model_path("scaler_X"), "rb") as f:
        scaler_X = pickle.load(f)
    scalers_y = {}
    for target in TARGET_COLS:
        with open(get_model_path(f"scaler_{target}"), "rb") as f:
            scalers_y[target] = pickle.load(f)
    return scaler_X, scalers_y


def inverse_transform_predictions(y_pred_scaled: np.ndarray, scaler_y) -> np.ndarray:
    if y_pred_scaled.ndim == 1:
        return scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    else:
        n, h = y_pred_scaled.shape
        return scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(n, h)
