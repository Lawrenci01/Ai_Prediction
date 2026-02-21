import numpy as np
import pandas as pd
import pickle
import logging
import time
from datetime import timedelta
from pathlib import Path

from app.pipeline.config import (
    TARGET_COLS, SEQUENCE_LENGTH, PREDICTION_HORIZON,
    MODEL_DIR, get_model_path, LOCATION
)
from app.pipeline.feature_engineer import build_features, get_feature_columns
from app.pipeline.sequence_builder import inverse_transform_predictions

logger = logging.getLogger(__name__)

def get_model_key(target: str) -> str:
    if target.startswith("co2"):
        return "lstm_co2"
    elif target.startswith("temp"):
        return "lstm_temperature"
    elif target.startswith("humid"):
        return "lstm_humidity"
    else:
        raise ValueError(f"Unknown target: {target}")

def get_unit(target: str) -> str:
    if "temp"  in target: return "°C"
    if "humid" in target: return "%"
    if "co2"   in target: return "ppm"
    logger.warning(f"get_unit: unknown target '{target}' — returning empty unit.")
    return ""

def is_model_file_valid(path: str) -> tuple:
    p = Path(path)
    if not p.exists():
        return False, f"File not found: {path}"
    if p.stat().st_size == 0:
        return False, f"File is empty: {path}"
    return True, "ok"

def _load_scaler_X(target: str):
    per_target_path = MODEL_DIR / f"scaler_X_{target}.pkl"
    if per_target_path.exists():
        with open(per_target_path, "rb") as f:
            logger.info(f"Loaded per-target scaler_X for {target}")
            return pickle.load(f)

    try:
        shared_path = get_model_path("scaler_X")
        if shared_path.exists():
            logger.warning(f"Per-target scaler_X not found for {target}, falling back to shared scaler_X")
            with open(shared_path, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load shared scaler_X for target '{target}': {e}")

    logger.error(f"No scaler_X found for target '{target}' — train.py may not have run")
    return None

_MODEL_TTL_SECONDS = 3600

class ModelRegistry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded    = {}
            cls._instance._load_time = {}
        return cls._instance

    def get(self, key: str):
        age = time.time() - self._load_time.get(key, 0)
        if key not in self._loaded or age > _MODEL_TTL_SECONDS:
            self._loaded[key]    = self._load(key)
            self._load_time[key] = time.time()
        return self._loaded[key]

    def _load(self, key: str):
        try:
            path = get_model_path(key)
        except KeyError:
            logger.error(f"Model key not in config: '{key}'")
            return None

        valid, reason = is_model_file_valid(path)
        if not valid:
            logger.error(f"Cannot load '{key}': {reason}")
            return None

        suffix = Path(path).suffix.lower()
        if suffix in (".keras", ".h5"):
            try:
                from tensorflow.keras.models import load_model
                logger.info(f"Loading Keras model [{key}]")
                return load_model(path)
            except Exception as e:
                logger.error(f"Failed to load Keras model '{key}': {e}")
                return None
        else:
            try:
                with open(path, "rb") as f:
                    logger.info(f"Loading pickle [{key}]")
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Failed to load pickle '{key}': {e}")
                return None

    def reload_all(self):
        self._loaded.clear()
        self._load_time.clear()
        logger.info("Model registry cleared.")

    def status(self) -> dict:
        from app.pipeline.config import MODEL_NAMES
        report = {}
        for key, filename in MODEL_NAMES.items():
            try:
                path  = get_model_path(key)
                valid, reason = is_model_file_valid(path)
                report[key] = {
                    "file":   filename,
                    "exists": valid,
                    "loaded": key in self._loaded and self._loaded[key] is not None,
                    "note":   reason if not valid else "ok",
                }
            except Exception as e:
                report[key] = {"file": filename, "exists": False, "error": str(e)}
        return report

registry = ModelRegistry()

def predict_with_lstm(df_featured: pd.DataFrame, target: str) -> dict:
    model_key    = get_model_key(target)
    scaler_y_key = f"scaler_{target}"

    model     = registry.get(model_key)
    scaler_y  = registry.get(scaler_y_key)
    feat_cols = registry.get("feature_cols")
    scaler_X  = _load_scaler_X(target)

    if any(x is None for x in [model, scaler_X, scaler_y, feat_cols]):
        missing = []
        if model     is None: missing.append("model")
        if scaler_X  is None: missing.append("scaler_X")
        if scaler_y  is None: missing.append("scaler_y")
        if feat_cols is None: missing.append("feature_cols")
        logger.error(f"Missing components for '{target}': {missing}")
        return {"error": f"Missing for '{target}': {missing}. Run train.py first."}

    missing_cols = [c for c in feat_cols if c not in df_featured.columns]
    if missing_cols:
        logger.error(f"Missing feature columns for '{target}': {missing_cols}")
        return {"error": f"Missing feature columns for '{target}': {missing_cols[:5]}..."}

    X = df_featured[feat_cols].values[-SEQUENCE_LENGTH:]
    if len(X) < SEQUENCE_LENGTH:
        logger.error(f"Insufficient rows for '{target}': need {SEQUENCE_LENGTH}, got {len(X)}")
        return {"error": f"Need {SEQUENCE_LENGTH} rows. Got {len(X)}."}

    X_scaled = scaler_X.transform(X)
    X_seq    = X_scaled.reshape(1, SEQUENCE_LENGTH, -1)
    y_scaled = model.predict(X_seq, verbose=0)
    last_ts  = pd.to_datetime(df_featured["timestamp"].iloc[-1])

    hourly = []
    for h in range(y_scaled.shape[1]):
        val = float(scaler_y.inverse_transform([[y_scaled[0, h]]])[0][0])
        hourly.append({
            "hour":            h + 1,
            "timestamp":       (last_ts + timedelta(hours=h + 1)).isoformat(),
            "predicted_value": round(val, 2),
            "unit":            get_unit(target),
        })

    values = [h["predicted_value"] for h in hourly]
    return {
        "model":              "LSTM",
        "target":             target,
        "unit":               get_unit(target),
        "prediction_date":    (last_ts + timedelta(days=1)).strftime("%Y-%m-%d"),
        "hourly_predictions": hourly,
        "daily_summary": {
            "mean": round(float(np.mean(values)), 2),
            "min":  round(float(np.min(values)),  2),
            "max":  round(float(np.max(values)),  2),
        },
    }

def predict_all(df_recent: pd.DataFrame, df_featured: pd.DataFrame) -> dict:

    last_ts = pd.to_datetime(df_featured["timestamp"].iloc[-1])
    results = {target: predict_with_lstm(df_featured, target) for target in TARGET_COLS}

    return {
        "prediction_date": (last_ts + timedelta(days=1)).strftime("%Y-%m-%d"),
        **results,
    }

def daily_forecast(df_recent: pd.DataFrame, df_featured: pd.DataFrame) -> dict:
    raw   = predict_all(df_recent, df_featured)
    clean = {"prediction_date": raw["prediction_date"]}

    for target in TARGET_COLS:
        r       = raw.get(target, {})
        summary = r.get("daily_summary", {})
        hourly  = r.get("hourly_predictions", [])

        clean[target] = {
            "mean":   summary.get("mean"),
            "min":    summary.get("min"),
            "max":    summary.get("max"),
            "unit":   r.get("unit", ""),
            "hourly": [
                {
                    "hour":      h["hour"],
                    "timestamp": h["timestamp"],
                    "value":     h["predicted_value"],
                }
                for h in hourly
            ],
        }

    return clean