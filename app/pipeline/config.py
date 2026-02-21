import os
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent.parent.parent
_data_file_env = os.environ.get("DATA_FILE", "")
DATA_FILE      = Path(_data_file_env) if _data_file_env else BASE_DIR / "Manila_HOURLY_20140101_20241231.csv"

if not DATA_FILE.exists():
    import warnings
    warnings.warn(
        f"DATA_FILE not found: {DATA_FILE}\n"
        "Set the DATA_FILE environment variable or place the CSV at the expected path.",
        stacklevel=2,
    )
MODEL_DIR = BASE_DIR / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_LEVEL  = "INFO"

TARGET_COLS   = ["co2_ppm", "temperature_c", "humidity_percent"]
TIMESTAMP_COL = "timestamp"
LOCATION      = "Manila"

SEQUENCE_LENGTH    = 168
PREDICTION_HORIZON = 24
TRAIN_SPLIT        = 0.80

EPOCHS        = 200
BATCH_SIZE    = 128
LEARNING_RATE = 0.0003
PATIENCE      = 15

LSTM_UNITS   = [128, 64]
DENSE_UNITS  = [64]
DROPOUT_RATE = 0.2

RF_N_ESTIMATORS      = 200
RF_MAX_DEPTH         = 20
RF_MIN_SAMPLES_SPLIT = 5
RF_MIN_SAMPLES_LEAF  = 2

XGB_N_ESTIMATORS     = 500
XGB_MAX_DEPTH        = 6
XGB_LEARNING_RATE    = 0.05
XGB_SUBSAMPLE        = 0.8
XGB_COLSAMPLE_BYTREE = 0.8
XGB_EARLY_STOPPING   = 30

USE_LAG_FEATURES      = True
USE_ROLLING_FEATURES  = True
USE_CYCLICAL_ENCODING = True

LAG_HOURS         = [1, 3, 6, 12, 24, 48, 72, 168, 336]
ROLLING_WINDOWS   = [6, 12, 24, 48, 168]
WET_SEASON_MONTHS = [6, 7, 8, 9, 10, 11]

MODEL_NAMES = {
    "lstm_co2":                "lstm_co2_ppm.keras",
    "lstm_temperature":        "lstm_temperature_c.keras",
    "lstm_humidity":           "lstm_humidity_percent.keras",
    "scaler_co2_ppm":          "scaler_co2_ppm.pkl",
    "scaler_temperature_c":    "scaler_temperature_c.pkl",
    "scaler_humidity_percent": "scaler_humidity_percent.pkl",
    "feature_cols":            "feature_cols.pkl",
    "ensemble_weights":        "ensemble_weights.pkl",
}


def get_model_path(key: str) -> Path:
    if key not in MODEL_NAMES:
        raise KeyError(
            f"Unknown model key: '{key}'. "
            f"Valid keys: {list(MODEL_NAMES.keys())}"
        )
    return MODEL_DIR / MODEL_NAMES[key]