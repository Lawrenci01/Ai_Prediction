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