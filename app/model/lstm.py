import logging
import numpy as np
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


_KEY_MAP = {
    "co2_ppm":          "lstm_co2",
    "temperature_c":    "lstm_temperature",
    "humidity_percent": "lstm_humidity",
}


def build_lstm_model(n_features: int, seq_len: int = 168):
    """Build and compile a Bidirectional LSTM model."""
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not installed.")

    model = Sequential([
        Input(shape=(seq_len, n_features)),
        Bidirectional(LSTM(LSTM_UNITS[0], return_sequences=True)),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),
        Bidirectional(LSTM(LSTM_UNITS[1], return_sequences=False)),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),
        Dense(DENSE_UNITS[0], activation="relu"),
        Dropout(DROPOUT_RATE),
        Dense(PREDICTION_HORIZON),
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="huber",
        metrics=["mae"],
    )
    logger.info(f"Built LSTM model: input=({seq_len}, {n_features}), output={PREDICTION_HORIZON}")
    return model


def load_lstm(target: str):
    """Load a saved LSTM model for the given target column."""
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not installed.")
    if target not in _KEY_MAP:
        raise ValueError(f"Unknown target: '{target}'. Valid: {list(_KEY_MAP.keys())}")
    path = get_model_path(_KEY_MAP[target])
    logger.info(f"Loading LSTM model for '{target}' from {path}")
    return load_model(str(path))


def save_lstm(model, target: str):
    """Save an LSTM model for the given target column."""
    if target not in _KEY_MAP:
        raise ValueError(f"Unknown target: '{target}'. Valid: {list(_KEY_MAP.keys())}")
    path = get_model_path(_KEY_MAP[target])
    model.save(str(path))
    logger.info(f"Saved LSTM model for '{target}' to {path}")


def get_callbacks(target: str) -> list:
    """Return standard training callbacks for the given target."""
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not installed.")
    if target not in _KEY_MAP:
        raise ValueError(f"Unknown target: '{target}'. Valid: {list(_KEY_MAP.keys())}")
    checkpoint_path = get_model_path(_KEY_MAP[target])
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(5, PATIENCE // 3),
            min_lr=1e-6,
            verbose=1,
        ),
    ]