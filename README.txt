================================================================
  AiPrediction — Complete Clean Project
  Manila Hourly Climate Prediction (CO2, Temperature, Humidity)
  LSTM Only
================================================================

PROJECT STRUCTURE
==================
AiPrediction/
├── Manila_HOURLY_20140101_20241231.csv   <- YOU provide this
├── saved_models/                          <- auto-created during training
├── app/
│   ├── __init__.py
│   ├── inference.py
│   ├── llm_engine.py
│   ├── context_builder.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── feature_engineer.py
│   │   └── sequence_builder.py
│   └── model/
│       ├── __init__.py
│       └── lstm.py
└── training/
    ├── __init__.py
    ├── train.py
    └── predict.py


SETUP
======
1. Place your CSV file in the root folder:
   Manila_HOURLY_20140101_20241231.csv

2. Install dependencies:
   pip install tensorflow numpy pandas scikit-learn xgboost python-dotenv httpx

3. Create a .env file in the root folder for LLM insights:
   LLM_BACKEND=ollama
   OLLAMA_MODEL=llama3.2:3b
   OLLAMA_URL=http://localhost:11434
   # OR for Groq:
   # LLM_BACKEND=groq
   # GROQ_API_KEY=your_key_here


TRAINING (run in this exact order)
=====================================
Step 1 — Temperature (fastest, ~1.5-3 hours):
   python training/train.py --target temperature_c --force-retrain

Step 2 — Humidity (~2-4 hours):
   python training/train.py --target humidity_percent --force-retrain

Step 3 — CO2 (~3-5 hours):
   python training/train.py --target co2_ppm --force-retrain

Check results:
   type saved_models\metrics_report.txt

Expected R2 scores:
   temperature_c    > 0.85
   humidity_percent > 0.80
   co2_ppm          > 0.50


PREDICTION
===========
Run after all 3 models are trained:
   python training/predict.py

Output files in saved_models/:
   predictions_2025_2026.csv          <- daily (547 rows)
   predictions_hourly_2025_2026.csv   <- hourly (13,128 rows)
   predictions_accuracy_summary.csv   <- accuracy by period

CSV columns (daily):
   date, days_ahead,
   co2_ppm_min, co2_ppm_mean, co2_ppm_max,
   temperature_c_min, temperature_c_mean, temperature_c_max,
   humidity_percent_min, humidity_percent_mean, humidity_percent_max,
   prediction_confidence_pct


WHAT CHANGED FROM ORIGINAL
============================
config.py
   BATCH_SIZE    32  → 64       (2x faster training)
   LEARNING_RATE 0.001 → 0.0003 (stable convergence)
   LSTM_UNITS    [256,128,64] → [128,64]  (less overfit)
   DENSE_UNITS   [128,64] → [64]
   DROPOUT_RATE  0.3 → 0.2
   Added LOCATION = "Manila"

feature_engineer.py
   Removed co2_lag_8760h  (was dropping all of 2014)
   Added Keeling Curve trend features
   Added diurnal plant cycle features
   Added CO2 z-score anomaly detection

train.py
   train_sklearn = False by default (LSTM only)
   No RF or XGBoost training

inference.py
   Removed LOCATION import bug
   daily_forecast() uses lstm directly

================================================================
