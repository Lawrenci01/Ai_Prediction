"""
Pipeline Configuration
=======================
Central config for all model hyperparameters, paths, and feature flags.
"""

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR  = Path(__file__).resolve().parent.parent.parent
DATA_FILE = BASE_DIR / "Manila_HOURLY_20140101_20241231.csv"
MODEL_DIR = BASE_DIR / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_LEVEL  = "INFO"

# ============================================================================
# TARGETS
# ============================================================================

TARGET_COLS   = ["co2_ppm", "temperature_c", "humidity_percent"]
TIMESTAMP_COL = "timestamp"
LOCATION      = "Manila"

# ============================================================================
# SEQUENCE / PREDICTION SETTINGS
# ============================================================================

SEQUENCE_LENGTH    = 168   # 7-day lookback window (hours)
PREDICTION_HORIZON = 24    # predict next 24 hours at once
TRAIN_SPLIT        = 0.80  # 80% train | 10% val | 10% test

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

EPOCHS        = 200
BATCH_SIZE    = 128      # 2x faster than 64, minimal accuracy tradeoff
LEARNING_RATE = 0.0003   # stable convergence
PATIENCE      = 15       # enough chances to improve, cuts dead time


# ============================================================================
# LSTM ARCHITECTURE
# ============================================================================

LSTM_UNITS   = [128, 64]   # right-sized for signal strength
DENSE_UNITS  = [64]
DROPOUT_RATE = 0.2

# ============================================================================
# RANDOM FOREST (kept in config but not used in training)
# ============================================================================

RF_N_ESTIMATORS      = 200
RF_MAX_DEPTH         = 20
RF_MIN_SAMPLES_SPLIT = 5
RF_MIN_SAMPLES_LEAF  = 2

# ============================================================================
# XGBOOST (kept in config but not used in training)
# ============================================================================

XGB_N_ESTIMATORS     = 500
XGB_MAX_DEPTH        = 6
XGB_LEARNING_RATE    = 0.05
XGB_SUBSAMPLE        = 0.8
XGB_COLSAMPLE_BYTREE = 0.8
XGB_EARLY_STOPPING   = 30

# ============================================================================
# FEATURE FLAGS
# ============================================================================

USE_LAG_FEATURES      = True
USE_ROLLING_FEATURES  = True
USE_CYCLICAL_ENCODING = True

# No 8760h lag — was dropping entire first year of training data
LAG_HOURS       = [1, 3, 6, 12, 24, 48, 72, 168, 336]
ROLLING_WINDOWS = [6, 12, 24, 48, 168]

# Manila wet season months (June–November)
WET_SEASON_MONTHS = [6, 7, 8, 9, 10, 11]

# ============================================================================
# MODEL FILE NAMES
# ============================================================================

MODEL_NAMES = {
    "lstm_co2":               "lstm_co2_ppm.keras",
    "lstm_temperature":       "lstm_temperature_c.keras",
    "lstm_humidity":          "lstm_humidity_percent.keras",
    "rf_co2":                 "rf_co2_ppm.pkl",
    "rf_temperature":         "rf_temperature_c.pkl",
    "rf_humidity":            "rf_humidity_percent.pkl",
    "xgb_co2":                "xgb_co2_ppm.pkl",
    "xgb_temperature":        "xgb_temperature_c.pkl",
    "xgb_humidity":           "xgb_humidity_percent.pkl",
    "scaler_X":               "scaler_X.pkl",
    "scaler_co2_ppm":         "scaler_co2_ppm.pkl",
    "scaler_temperature_c":   "scaler_temperature_c.pkl",
    "scaler_humidity_percent": "scaler_humidity_percent.pkl",
    "feature_cols":           "feature_cols.pkl",
    "ensemble_weights":       "ensemble_weights.pkl",
}


def get_model_path(key: str) -> Path:
    """Return full path for a model file by key."""
    if key not in MODEL_NAMES:
        raise KeyError(
            f"Unknown model key: '{key}'. "
            f"Valid keys: {list(MODEL_NAMES.keys())}"
        )
    return MODEL_DIR / MODEL_NAMES[key]
