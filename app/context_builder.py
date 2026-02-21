import pandas as pd


def build_current_readings(df: pd.DataFrame) -> dict:
    latest = df.sort_values("timestamp").iloc[-1]
    return {
        "co2_ppm":          round(float(latest["co2_ppm"]), 2),
        "temperature_c":    round(float(latest["temperature_c"]), 2),
        "humidity_percent": round(float(latest["humidity_percent"]), 2),
    }


def build_historical_stats(df: pd.DataFrame, hours: int = 24) -> dict:
    df     = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    cutoff = df["timestamp"].max() - pd.Timedelta(hours=hours)
    df     = df[df["timestamp"] >= cutoff]
    stats  = {}

    for col in ["co2_ppm", "temperature_c", "humidity_percent"]:
        if col in df.columns:
            stats[col] = {
                "mean":  round(float(df[col].mean()), 2),
                "max":   round(float(df[col].max()), 2),
                "min":   round(float(df[col].min()), 2),
                "range": round(float(df[col].max() - df[col].min()), 2),
                "trend": "rising" if df[col].iloc[-1] > df[col].mean() else "falling",
            }

    return stats


def build_forecast_series(temp_fc: list, hum_fc: list, co2_fc: list) -> list:
    series = []
    for i in range(min(len(temp_fc), len(hum_fc), len(co2_fc))):
        series.append({
            "hour":             temp_fc[i].get("hour", i + 1),
            "timestamp":        temp_fc[i].get("timestamp", ""),
            "temperature_c":    temp_fc[i].get("predicted_value", 0.0),
            "humidity_percent": hum_fc[i].get("predicted_value", 0.0),
            "co2_ppm":          co2_fc[i].get("predicted_value", 0.0),
        })
    return series