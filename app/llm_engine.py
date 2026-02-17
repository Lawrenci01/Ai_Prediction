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
from datetime import datetime
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
    """Rothfusz heat index — feels like temperature in Celsius."""
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
    """Thom's discomfort index."""
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


# ============================================================================
# PROMPT BUILDERS
# ============================================================================

def _build_prompt(sensor: dict,
                  current: dict,
                  predicted: dict,
                  hours_ahead: int) -> str:

    sensor_id     = sensor.get("sensor_id", "?")
    barangay      = sensor.get("barangay", "unknown").title()
    establishment = sensor.get("establishment", "unknown").title()

    temp_now  = float(current.get("temperature_c",    28.0))
    hum_now   = float(current.get("humidity_percent", 75.0))
    co2_now   = float(current.get("co2_ppm",          415.0))

    temp_pred = float(predicted.get("temperature_c",    28.0))
    hum_pred  = float(predicted.get("humidity_percent", 75.0))
    co2_pred  = float(predicted.get("co2_ppm",          415.0))

    hi_now  = heat_index(temp_now,  hum_now)
    hi_pred = heat_index(temp_pred, hum_pred)
    di_now  = discomfort_index(temp_now,  hum_now)
    di_pred = discomfort_index(temp_pred, hum_pred)

    temp_delta = temp_pred - temp_now
    hum_delta  = hum_pred  - hum_now
    co2_delta  = co2_pred  - co2_now

    hour_of_day = (datetime.now().hour + hours_ahead) % 24
    time_of_day = (
        "early morning" if 5  <= hour_of_day < 9  else
        "mid-morning"   if 9  <= hour_of_day < 12 else
        "afternoon"     if 12 <= hour_of_day < 17 else
        "evening"       if 17 <= hour_of_day < 21 else
        "nighttime"
    )

    return f"""You are an environmental monitoring AI analyst for IoT climate sensors in Manila, Philippines.

SENSOR INFORMATION:
- Node ID: {sensor_id}
- Location: {establishment}, Barangay {barangay}, Manila

CURRENT READINGS:
- Temperature: {temp_now:.1f}°C (feels like {hi_now:.1f}°C)
- Humidity: {hum_now:.0f}%
- CO2: {co2_now:.0f} ppm ({co2_risk_level(co2_now)})
- Discomfort Index: {di_now:.1f}

PREDICTED READINGS (in {hours_ahead} hours — {time_of_day}):
- Temperature: {temp_pred:.1f}°C (feels like {hi_pred:.1f}°C, change: {temp_delta:+.1f}°C)
- Humidity: {hum_pred:.0f}% (change: {hum_delta:+.1f}%)
- CO2: {co2_pred:.0f} ppm ({co2_risk_level(co2_pred)}, change: {co2_delta:+.1f} ppm)
- Discomfort Index: {di_pred:.1f} (change: {di_pred - di_now:+.1f})

TASK:
Write exactly 2 sentences as a climate insight for this sensor location.

Sentence 1: Describe what the sensor is currently detecting and where conditions are heading over the next {hours_ahead} hours. Be specific — mention actual values and trends.

Sentence 2: Reason about what this combination of changes means for people at {establishment} and give one clear, specific, actionable recommendation.

Rules:
- Use natural, professional language
- Reference the specific location ({establishment}, Brgy. {barangay}) and Node {sensor_id}
- Do NOT use bullet points, headers, or lists
- Do NOT start with "I" or "The sensor"
- Output only the 2 sentences, nothing else
"""


def _build_forecast_prompt(sensor: dict,
                            current: dict,
                            forecast_series: list) -> str:

    sensor_id     = sensor.get("sensor_id", "?")
    barangay      = sensor.get("barangay", "unknown").title()
    establishment = sensor.get("establishment", "unknown").title()

    temps = [float(f.get("temperature_c",    0)) for f in forecast_series]
    hums  = [float(f.get("humidity_percent", 0)) for f in forecast_series]
    co2s  = [float(f.get("co2_ppm",          0)) for f in forecast_series]
    hours = [f.get("hour", i + 1) for i, f in enumerate(forecast_series)]

    max_temp      = max(temps)
    min_temp      = min(temps)
    max_co2       = max(co2s)
    max_hum       = max(hums)
    max_temp_hour = hours[temps.index(max_temp)]
    max_co2_hour  = hours[co2s.index(max_co2)]

    stress_scores = [heat_index(t, h) + (c / 100) for t, h, c in zip(temps, hums, co2s)]
    worst_idx     = stress_scores.index(max(stress_scores))
    worst_hour    = hours[worst_idx]
    worst_hi      = heat_index(temps[worst_idx], hums[worst_idx])

    series_summary = "\n".join([
        f"  +{hours[i]}h: {temps[i]:.1f}°C, {hums[i]:.0f}% humidity, {co2s[i]:.0f} ppm CO2"
        for i in range(len(forecast_series))
    ])

    return f"""You are an environmental monitoring AI analyst for IoT climate sensors in Manila, Philippines.

SENSOR: Node {sensor_id} at {establishment}, Barangay {barangay}

CURRENT READINGS:
- Temperature: {current.get('temperature_c', 0):.1f}°C
- Humidity: {current.get('humidity_percent', 0):.0f}%
- CO2: {current.get('co2_ppm', 0):.0f} ppm

FORECAST SERIES (next {max(hours)} hours):
{series_summary}

KEY STATISTICS:
- Temperature range: {min_temp:.1f}°C to {max_temp:.1f}°C (peak at +{max_temp_hour}h)
- Peak CO2: {max_co2:.0f} ppm at +{max_co2_hour}h ({co2_risk_level(max_co2)})
- Peak humidity: {max_hum:.0f}%
- Worst combined stress: +{worst_hour}h (apparent temperature {worst_hi:.1f}°C)

TASK:
Write exactly 2 sentences summarizing this forecast for {establishment}.

Sentence 1: Summarize the overall trend and key peaks across the forecast window with specific values and timing.

Sentence 2: Identify the most critical period and give one actionable recommendation for building management or occupants.

Rules:
- Natural, professional language
- Reference Node {sensor_id} and {establishment} specifically
- No bullet points, headers, or lists
- Output only the 2 sentences, nothing else
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
                    "temperature": 0.7,
                    "top_p":       0.9,
                    "num_predict": 200,
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
                "temperature": 0.7,
                "max_tokens":  200,
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()


async def _call_llm(prompt: str) -> str:
    if LLM_BACKEND.lower() == "groq":
        return await _call_groq(prompt)
    return await _call_ollama(prompt)


# ============================================================================
# FALLBACK
# ============================================================================

def _fallback_insight(sensor: dict, current: dict, backend: str) -> str:
    return (
        f"Node {sensor.get('sensor_id')} at {sensor.get('establishment', 'the monitored area')} "
        f"recorded {current.get('temperature_c', '?')}°C, "
        f"{current.get('humidity_percent', '?')}% humidity, "
        f"and {current.get('co2_ppm', '?')} ppm CO2. "
        f"Insight generation is temporarily unavailable — please check the {backend} service."
    )


# ============================================================================
# PUBLIC API
# ============================================================================

async def generate_insight(
    sensor:      dict,
    current:     dict,
    predicted:   dict,
    hours_ahead: int = 24
) -> str:
    try:
        prompt  = _build_prompt(sensor, current, predicted, hours_ahead)
        insight = await _call_llm(prompt)
        logger.info(f"Insight generated via {LLM_BACKEND} for sensor {sensor.get('sensor_id')}")
        return insight
    except Exception as e:
        logger.error(f"Insight generation failed ({LLM_BACKEND}): {e}")
        return _fallback_insight(sensor, current, LLM_BACKEND)


async def generate_forecast_summary(
    sensor:          dict,
    current:         dict,
    forecast_series: list,
) -> str:
    if not forecast_series:
        return "No forecast data available."
    try:
        prompt  = _build_forecast_prompt(sensor, current, forecast_series)
        summary = await _call_llm(prompt)
        logger.info(f"Forecast summary generated via {LLM_BACKEND} for sensor {sensor.get('sensor_id')}")
        return summary
    except Exception as e:
        logger.error(f"Forecast summary generation failed ({LLM_BACKEND}): {e}")
        return _fallback_insight(sensor, current, LLM_BACKEND)
