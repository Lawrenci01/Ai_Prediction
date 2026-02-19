"""
Test that all models load and can make predictions
without needing enough rows for full sequences.
"""
import pickle
import numpy as np
import sys
import os
from dotenv import load_dotenv
load_dotenv()

sys.path.append(".")

from app.pipeline.config import TARGET_COLS, get_model_path, SEQUENCE_LENGTH
from app.pipeline.sequence_builder import _scaler_X_path
from app.model.lstm import load_lstm

print("=" * 50)
print("STEP 1 - Check saved_models files exist")
print("=" * 50)

files_to_check = []
for target in TARGET_COLS:
    files_to_check.append((_scaler_X_path(target), f"scaler_X_{target}"))
    files_to_check.append((get_model_path(f"scaler_{target}"), f"scaler_{target}"))
files_to_check.append((get_model_path("feature_cols"), "feature_cols"))

all_ok = True
for path, name in files_to_check:
    if path.exists():
        print(f"  OK      {name}")
    else:
        print(f"  MISSING {name} <- {path}")
        all_ok = False

if not all_ok:
    print("\nSome files are missing. Cannot continue.")
    sys.exit(1)

print("\n" + "=" * 50)
print("STEP 2 - Load models and run dummy prediction")
print("=" * 50)

try:
    feature_cols = pickle.load(open(get_model_path("feature_cols"), "rb"))
    print(f"  feature_cols loaded: {len(feature_cols)} features")
except Exception as e:
    print(f"  FAILED loading feature_cols: {e}")
    sys.exit(1)

n_features = len(feature_cols)

for target in TARGET_COLS:
    print(f"\n  --- {target} ---")
    try:
        with open(_scaler_X_path(target), "rb") as f:
            scaler_X = pickle.load(f)
        with open(get_model_path(f"scaler_{target}"), "rb") as f:
            scaler_y = pickle.load(f)
        print(f"    scalers loaded OK")
    except Exception as e:
        print(f"    FAILED loading scalers: {e}")
        continue

    try:
        model = load_lstm(target)
        print(f"    model loaded OK")
    except Exception as e:
        print(f"    FAILED loading model: {e}")
        continue

    try:
        dummy_input = np.random.rand(1, SEQUENCE_LENGTH, n_features).astype(np.float32)
        pred = model.predict(dummy_input, verbose=0)
        pred_actual = scaler_y.inverse_transform(
            pred.reshape(-1, 1)).reshape(pred.shape)
        print(f"    dummy prediction OK - shape: {pred_actual.shape}")
        print(f"    sample values: {pred_actual[0][:3].round(2)} ...")
    except Exception as e:
        print(f"    FAILED prediction: {e}")

print("\n" + "=" * 50)
print("STEP 3 - Check DB connection and row count")
print("=" * 50)

try:
    from sqlalchemy import create_engine, text
    url = (
        f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT', '3306')}/{os.getenv('DB_NAME')}"
    )
    engine = create_engine(url)
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT COUNT(*), MIN(recorded_at), MAX(recorded_at)
            FROM sensor_data
        """)).fetchone()
        print(f"  rows   : {row[0]}")
        print(f"  oldest : {row[1]}")
        print(f"  newest : {row[2]}")
        print(f"  DB OK")
except Exception as e:
    print(f"  DB FAILED: {e}")

print("\n" + "=" * 50)
print("DONE - if all steps show OK, workflow is ready")
print("=" * 50)