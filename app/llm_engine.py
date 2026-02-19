"""
LLM Insight Engine
==================
AI-powered insight engine for IoT sensor climate predictions.
Uses Ollama (local) or Groq (production) to generate genuine
reasoning-based insights — not if/else templates.

Backends:
    - Ollama (default): local, offline, free
    - Groq: production, fast, free tier available

Set in .env:
    LLM_BACKEND=ollama        # or groq
    OLLAMA_MODEL=llama3.2:3b
    OLLAMA_URL=http://localhost:11434
    GROQ_API_KEY=your_key_here
    GROQ_MODEL=llama-3.3-70b-versatile
"""

import logging
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIG
# ============================================================================

LLM_BACKEND  = os.getenv("LLM_BACKEND",  "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL",   "llama-3.3-70b-versatile")


# ============================================================================
# CLIMATE SCIENCE CALCULATORS
# ============================================================================

def heat_index(temp_c: float, humidity: float) -> float:
    try:
        t  = temp_c * 9 / 5 + 32
        rh = humidity
        hi = (-42.379
              + 2.04901523  * t
              + 10.14333127 * rh
              - 0.22475541  * t * rh
              - 0.00683783  * t * t
              - 0.05481717  * rh * rh
              + 0.00122874  * t * t * rh
              + 0.00085282  * t * rh * rh
              - 0.00000199  * t * t * rh * rh)
        return (hi - 32) * 5 / 9
    except Exception:
        return temp_c


def discomfort_index(temp_c: float, humidity: float) -> float:
    try:
        return temp_c - 0.55 * (1 - humidity / 100) * (temp_c - 14.5)
    except Exception:
        return temp_c


def co2_risk_level(co2_ppm: float) -> str:
    thresholds = [
        (400,  "below atmospheric baseline"),
        (450,  "normal outdoor levels"),
        (600,  "slightly elevated for an urban area"),
        (800,  "moderately concentrated"),
        (1000, "high enough to affect focus indoors"),
        (1500, "well above safe indoor thresholds"),
        (9999, "critically high — immediate action needed"),
    ]
    for threshold, desc in thresholds:
        if co2_ppm < threshold:
            return desc
    return "critically high — immediate action needed"


def comfort_label(hi: float) -> str:
    if hi < 27:  return "comfortable"
    if hi < 32:  return "caution"
    if hi < 41:  return "extreme caution"
    if hi < 54:  return "danger"
    return "extreme danger"


# ============================================================================
# PROMPT BUILDER
# ============================================================================

def _build_insight_prompt(sensor: dict,
                           predicted: dict,
                           peak_temp_hour: int,
                           peak_co2_hour: int) -> str:

    sensor_id = sensor.get("sensor_id", "NODE-01")
    barangay  = sensor.get("barangay", "Naga").title()

    co2_pred  = float(predicted.get("co2_ppm",          415.0))
    temp_pred = float(predicted.get("temperature_c",    28.0))
    hum_pred  = float(predicted.get("humidity_percent", 75.0))
    hi_pred   = heat_index(temp_pred, hum_pred)
    co2_risk  = co2_risk_level(co2_pred)
    comfort   = comfort_label(hi_pred)

    # Determine worst condition to drive recommendation
    if co2_pred >= 440:
        recommendation = "immediately improve ventilation and reduce indoor occupancy"
    elif co2_pred >= 420:
        recommendation = "open windows and increase ventilation especially during peak CO2 hours"
    elif hi_pred >= 41:
        recommendation = "avoid all outdoor activities during peak heat and ensure cooling"
    elif hi_pred >= 32:
        recommendation = "limit outdoor exposure during midday and stay hydrated"
    elif hum_pred >= 90:
        recommendation = "ensure proper airflow in enclosed spaces to reduce moisture"
    else:
        recommendation = "no major concerns today, maintain normal ventilation"

    return f"""You are a climate monitoring AI for IoT sensors in Naga City, Philippines.

Write exactly ONE sentence using ONLY these predicted values:

PREDICTED VALUES (use these exact numbers):
- Sensor: {sensor_id} in {barangay}
- CO2: {co2_pred:.0f} ppm ({co2_risk})
- Temperature peak: {temp_pred:.1f}°C at {peak_temp_hour:02d}:00 (feels like {hi_pred:.1f}°C — {comfort})
- Humidity: {hum_pred:.0f}%
- Recommended action: {recommendation}

REQUIRED OUTPUT FORMAT (fill in the brackets with the values above):
"{sensor_id} in {barangay}: CO2 is at {co2_pred:.0f} ppm ({co2_risk}), temperature peaks at {temp_pred:.1f}°C at {peak_temp_hour:02d}:00 (feels like {hi_pred:.1f}°C — {comfort}), and humidity is at {hum_pred:.0f}% — {recommendation}."

Output that exact sentence with those exact numbers. Do not change the values. Do not add anything else.
"""


# ============================================================================
# LLM BACKENDS
# ============================================================================

async def _call_ollama(prompt: str) -> str:
    import httpx
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p":       0.9,
                    "num_predict": 100,
                }
            }
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()


async def _call_groq(prompt: str) -> str:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set in .env file.")
    import httpx
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens":  100,
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()


async def _call_llm(prompt: str) -> str:
    if LLM_BACKEND.lower() == "groq":
        return await _call_groq(prompt)
    return await _call_ollama(prompt)


# ============================================================================
# FALLBACK — used when LLM is unavailable
# ============================================================================

def _fallback_insight(sensor: dict,
                       predicted: dict,
                       peak_temp_hour: int,
                       peak_co2_hour: int) -> str:
    sensor_id = sensor.get("sensor_id", "NODE-01")
    barangay  = sensor.get("barangay", "Naga").title()
    co2       = float(predicted.get("co2_ppm",          415))
    temp      = float(predicted.get("temperature_c",    28))
    hum       = float(predicted.get("humidity_percent", 75))
    hi        = heat_index(temp, hum)

    if co2 >= 440:
        rec = "immediately improve ventilation and reduce indoor occupancy"
    elif co2 >= 420:
        rec = "open windows and increase ventilation especially during peak CO2 hours"
    elif hi >= 41:
        rec = "avoid all outdoor activities during peak heat and ensure cooling"
    elif hi >= 32:
        rec = "limit outdoor exposure during midday and stay hydrated"
    elif hum >= 90:
        rec = "ensure proper airflow in enclosed spaces to reduce moisture"
    else:
        rec = "no major concerns today, maintain normal ventilation"

    return (
        f"{sensor_id} in {barangay}: CO2 is at {co2:.0f} ppm ({co2_risk_level(co2)}), "
        f"temperature peaks at {temp:.1f}°C at {peak_temp_hour:02d}:00 "
        f"(feels like {hi:.1f}°C — {comfort_label(hi)}), "
        f"and humidity is at {hum:.0f}% — {rec}."
    )


# ============================================================================
# PUBLIC API
# ============================================================================

async def generate_insight(
    sensor:         dict,
    current:        dict,
    predicted:      dict,
    peak_temp_hour: int = 13,
    peak_co2_hour:  int = 12,
    hours_ahead:    int = 24,
) -> str:
    try:
        prompt  = _build_insight_prompt(sensor, predicted, peak_temp_hour, peak_co2_hour)
        insight = await _call_llm(prompt)
        logger.info(f"Insight generated via {LLM_BACKEND} for sensor {sensor.get('sensor_id')}")
        return insight
    except Exception as e:
        logger.error(f"Insight generation failed ({LLM_BACKEND}): {e}")
        return _fallback_insight(sensor, predicted, peak_temp_hour, peak_co2_hour)


async def generate_forecast_summary(
    sensor:          dict,
    current:         dict,
    forecast_series: list,
) -> str:
    return ""