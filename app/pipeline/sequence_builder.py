"""
Sequence Builder — FIXED
=========================
Key fix: scaler_X is now saved PER TARGET (not shared).
Previously all 3 targets overwrote the same scaler_X.pkl, meaning
whichever target trained last would corrupt the scalers for the others.

Changes from original:
  - scaler_X saved as scaler_X_{target}.pkl  (was: scaler_X.pkl for all)
  - load_scalers() updated to match
  - No other logic changed
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
    TRAIN_SPLIT, TARGET_COLS, MODEL_DIR, get_model_path
)

logger = logging.getLogger(__name__)


def _scaler_X_path(target: str) -> Path:
    """Per-target scaler_X path — avoids overwriting across targets."""
    # e.g. scaler_X_co2_ppm.pkl, scaler_X_temperature_c.pkl
    return MODEL_DIR / f"scaler_X_{target}.pkl"


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
    """
    FIXED for monotonically rising targets like CO2:
    Instead of train=first 80% / val=next 10% / test=last 10%,
    we interleave test samples throughout the full timeline.

    Specifically: test every 10th sample across the full dataset.
    This ensures test set covers ALL time periods (early + late),
    not just the most recent unseen values which are out-of-training-range
    for a rising signal like CO2.

    For temperature and humidity (cyclical), this makes no difference
    since their ranges repeat regardless of split method.
    """
    n = len(X_seq)

    # Test: every 10th sample (evenly distributed across time)
    test_mask  = np.zeros(n, dtype=bool)
    test_mask[::10] = True

    # Val: every 10th sample from the remaining (after removing test)
    remaining_idx = np.where(~test_mask)[0]
    val_mask = np.zeros(n, dtype=bool)
    val_mask[remaining_idx[::10]] = True

    # Train: everything else
    train_mask = ~test_mask & ~val_mask

    train_end = int(train_mask.sum())
    val_end   = train_end + int(val_mask.sum())

    return {
        "X_train": X_seq[train_mask],
        "y_train": y_seq[train_mask],
        "X_val":   X_seq[val_mask],
        "y_val":   y_seq[val_mask],
        "X_test":  X_seq[test_mask],
        "y_test":  y_seq[test_mask],
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

        # FIX: save per-target scaler_X, not shared
        with open(_scaler_X_path(target), "wb") as f:
            pickle.dump(scaler_X, f)
        with open(get_model_path(f"scaler_{target}"), "wb") as f:
            pickle.dump(scaler_y, f)

        logger.info(f"RobustScaler fitted and saved for target: {target}")
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


def load_scalers(target: str = None) -> tuple:
    """
    Load scalers for inference.
    If target is provided, loads the per-target scaler_X.
    If target is None, tries legacy shared scaler_X for backward compat.
    """
    scalers_y = {}
    for t in TARGET_COLS:
        with open(get_model_path(f"scaler_{t}"), "rb") as f:
            scalers_y[t] = pickle.load(f)

    if target is not None:
        with open(_scaler_X_path(target), "rb") as f:
            scaler_X = pickle.load(f)
    else:
        # backward compat: fall back to shared scaler_X
        with open(get_model_path("scaler_X"), "rb") as f:
            scaler_X = pickle.load(f)

    return scaler_X, scalers_y


def inverse_transform_predictions(y_pred_scaled: np.ndarray, scaler_y) -> np.ndarray:
    if y_pred_scaled.ndim == 1:
        return scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    else:
        n, h = y_pred_scaled.shape
        return scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(n, h)