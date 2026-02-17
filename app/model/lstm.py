"""
LSTM Model Architecture
========================
Builds, saves, and loads the LSTM for climate forecasting.
"""

import logging
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from app.pipeline.config import (
    LSTM_UNITS, DROPOUT_RATE, DENSE_UNITS,
    LEARNING_RATE, PREDICTION_HORIZON, PATIENCE, get_model_path
)

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import (
        LSTM, Dense, Dropout, BatchNormalization, Input, Bidirectional
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
    )
    TF_AVAILABLE = True
    logger.info(f"TensorFlow {tf.__version__} available.")
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not installed.")


def get_lstm_key(target: str) -> str:
    if "co2" in target:
        return "lstm_co2"
    elif "temp" in target:
        return "lstm_temperature"
    elif "humid" in target:
        return "lstm_humidity"
    else:
        raise ValueError(f"Unknown target: {target}")


def build_lstm_model(input_shape: tuple,
                     target: str,
                     lstm_units: list = None,
                     dropout: float = None,
                     dense_units: list = None,
                     lr: float = None,
                     bidirectional: bool = False):

    if not TF_AVAILABLE:
        raise ImportError("TensorFlow required.")

    lstm_units  = lstm_units  or LSTM_UNITS
    dropout     = dropout     or DROPOUT_RATE
    dense_units = dense_units or DENSE_UNITS
    lr          = lr          or LEARNING_RATE

    model = Sequential(name=f"LSTM_{target}")
    model.add(Input(shape=input_shape))

    for i, units in enumerate(lstm_units):
        return_seq = (i < len(lstm_units) - 1)
        layer = LSTM(units, return_sequences=return_seq, name=f"lstm_{i+1}")
        if i == 0 and bidirectional:
            model.add(Bidirectional(layer, name=f"bilstm_{i+1}"))
        else:
            model.add(layer)
        model.add(BatchNormalization(name=f"bn_{i+1}"))
        model.add(Dropout(dropout, name=f"drop_{i+1}"))

    for i, units in enumerate(dense_units):
        model.add(Dense(units, activation="relu", name=f"dense_{i+1}"))
        model.add(Dropout(dropout / 2, name=f"drop_dense_{i+1}"))

    model.add(Dense(PREDICTION_HORIZON, activation="linear", name="output"))

    # FIX: Removed CosineDecayRestarts schedule — using a plain float lr
    # so that ReduceLROnPlateau in get_callbacks() can adjust it freely.
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="huber",
        metrics=["mae", "mse"]
    )

    logger.info(
        f"LSTM built for '{target}'. "
        f"Input: {input_shape}, Output: {PREDICTION_HORIZON}h, "
        f"Bidirectional: {bidirectional}"
    )
    model.summary(print_fn=logger.info)
    return model


def get_callbacks(target: str, log_dir: str = None) -> list:
    if not TF_AVAILABLE:
        return []

    ckpt_path = get_model_path(get_lstm_key(target))
    logger.info(f"  ModelCheckpoint: {ckpt_path}")

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        # FIX: This now works correctly because the optimizer uses a plain
        # float lr instead of a LearningRateSchedule object.
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        ),
    ]
    if log_dir:
        callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=1))
    return callbacks


def save_lstm(model, target: str):
    if not TF_AVAILABLE:
        return
    path = get_model_path(get_lstm_key(target))
    model.save(path)
    logger.info(f"LSTM saved: {path}")


def load_lstm(target: str):
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow required.")
    path = get_model_path(get_lstm_key(target))
    model = load_model(path)
    logger.info(f"LSTM loaded: {path}")
    return model


def predict_lstm(model, X_seq: np.ndarray, scaler_y) -> np.ndarray:
    y_scaled = model.predict(X_seq, verbose=0)
    n, h     = y_scaled.shape
    return scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).reshape(n, h)