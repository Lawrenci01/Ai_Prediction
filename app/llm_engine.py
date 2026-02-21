import logging
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

LLM_BACKEND  = os.getenv("LLM_BACKEND",  "groq")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL",   "llama-3.3-70b-versatile")

if LLM_BACKEND.lower() == "groq" and not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY is not set — Groq calls will fail and fall back to rule-based insights.")


def heat_index(temp_c: float, humidity: float) -> float:
    try:
        t  = temp_c * 9 / 5 + 32
        rh = humidity
        hi = (-42.379
              + 2.04901523  * t + 10.14333127 * rh
              - 0.22475541  * t * rh - 0.00683783 * t * t
              - 0.05481717  * rh * rh + 0.00122874 * t * t * rh
              + 0.00085282  * t * rh * rh
              - 0.00000199  * t * t * rh * rh)
        return (hi - 32) * 5 / 9
    except Exception as e:
        logger.warning(f"heat_index failed (temp={temp_c}, hum={humidity}): {e}")
        return temp_c


def co2_risk_level(co2_ppm: float) -> str:
    if co2_ppm < 450:  return "normal outdoor levels"
    if co2_ppm < 600:  return "slightly elevated"
    if co2_ppm < 800:  return "moderately elevated"
    if co2_ppm < 1000: return "high — may affect comfort"
    if co2_ppm < 1500: return "very high — poor air quality"
    return "critically high — immediate action needed"


def comfort_label(hi: float) -> str:
    if hi < 27: return "comfortable"
    if hi < 32: return "warm"
    if hi < 41: return "hot — caution advised"
    if hi < 54: return "very hot — dangerous"
    return "extreme danger"


async def _call_groq(prompt: str) -> str:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set.")
    import asyncio
    import httpx
    async with httpx.AsyncClient(timeout=30.0) as client:
        for attempt in range(3):
            try:
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                    json={
                        "model":       GROQ_MODEL,
                        "messages":    [{"role": "user", "content": prompt}],
                        "temperature": 0.3,
                        "max_tokens":  300,
                    },
                )
                if response.status_code == 429:
                    if attempt == 2:
                        raise RuntimeError("Groq rate limit exceeded after 3 retries.")
                    wait = 2 ** attempt
                    logger.warning(f"Groq rate limited — retrying in {wait}s (attempt {attempt + 1}/3)")
                    await asyncio.sleep(wait)
                    continue
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                if attempt == 2:
                    raise
                logger.warning(f"Groq attempt {attempt + 1} failed: {e}")
    raise RuntimeError("Groq failed after 3 attempts.")


async def _call_ollama(prompt: str) -> str:
    import httpx
    async with httpx.AsyncClient(timeout=60.0) as client:
        for attempt in range(3):
            try:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model":   OLLAMA_MODEL,
                        "prompt":  prompt,
                        "stream":  False,
                        "options": {"temperature": 0.3, "num_predict": 300},
                    },
                )
                response.raise_for_status()
                return response.json().get("response", "").strip()
            except Exception as e:
                if attempt == 2:
                    raise
                logger.warning(f"Ollama attempt {attempt + 1} failed: {e}")
    raise RuntimeError("Ollama failed after 3 attempts.")


async def _call_llm(prompt: str) -> str:
    if LLM_BACKEND.lower() == "groq":
        return await _call_groq(prompt)
    return await _call_ollama(prompt)


def _parse_status_insight(response: str) -> tuple[str, str]:
    lines        = response.strip().splitlines()
    safe_status  = "CAUTION"
    insight_text = response.strip()

    for line in lines:
        line = line.strip()
        if line.upper().startswith("STATUS:"):
            raw = line.split(":", 1)[1].strip().upper()
            if "UNSAFE" in raw:
                safe_status = "UNSAFE"
            elif "SAFE" in raw and "CAUTION" not in raw:
                safe_status = "SAFE"
            else:
                safe_status = "CAUTION"
        elif line.upper().startswith("INSIGHT:"):
            insight_text = line.split(":", 1)[1].strip()

    return safe_status, insight_text


def _build_hourly_predicted_prompt(
    establishment_name: str,
    establishment_type: str,
    barangay_name:      str,
    hour:               int,
    prediction_date:    str,
    co2_ppm:            float,
    temperature_c:      float,
    humidity_percent:   float,
) -> str:
    hi       = heat_index(temperature_c, humidity_percent)
    time_str = f"{hour:02d}:00"
    return f"""You are an environmental health AI for Naga City, Philippines.

Based on PREDICTED sensor values for a future hour, determine if this location
will be safe to visit at that time.

ESTABLISHMENT:
  Name : {establishment_name}
  Type : {establishment_type}
  Area : {barangay_name}, Naga City

PREDICTED VALUES FOR {prediction_date} at {time_str}:
  CO2         : {co2_ppm:.0f} ppm ({co2_risk_level(co2_ppm)})
  Temperature : {temperature_c:.1f}°C
  Humidity    : {humidity_percent:.0f}%
  Heat Index  : {hi:.1f}°C (feels like — {comfort_label(hi)})

SAFETY THRESHOLDS:
  CO2        : <800 = safe | 800–1000 = caution | >1000 = unsafe
  Heat Index : <32°C = safe | 32–41°C = caution | >41°C = unsafe
  Humidity   : <85% = fine | 85–95% = caution | >95% = uncomfortable

YOUR TASK:
1. Decide: SAFE, CAUTION, or UNSAFE
2. Write 2 sentences explaining why, mentioning:
   - {establishment_name} by name
   - The specific predicted values that drove your decision
   - A practical tip for visitors at {time_str}

RESPOND IN THIS EXACT FORMAT:
STATUS: [SAFE or CAUTION or UNSAFE]
INSIGHT: [your 2 sentence explanation]
"""


def _fallback_hourly_insight(
    establishment_name: str,
    co2_ppm:            float,
    temperature_c:      float,
    humidity_percent:   float,
    hour:               int,
) -> tuple[str, str]:
    hi          = heat_index(temperature_c, humidity_percent)
    issues      = []
    safe_status = "SAFE"

    if co2_ppm > 1000:
        safe_status = "UNSAFE"
        issues.append(f"CO2 predicted at {co2_ppm:.0f} ppm")
    elif co2_ppm > 800:
        safe_status = "CAUTION"
        issues.append(f"CO2 predicted at {co2_ppm:.0f} ppm")

    if hi > 41:
        safe_status = "UNSAFE"
        issues.append(f"heat index predicted at {hi:.1f}°C")
    elif hi > 32:
        if safe_status != "UNSAFE":
            safe_status = "CAUTION"
        issues.append(f"heat index predicted at {hi:.1f}°C")

    if not issues:
        insight = (
            f"{establishment_name} at {hour:02d}:00 is predicted to have safe conditions "
            f"with CO2 at {co2_ppm:.0f} ppm and temperature at {temperature_c:.1f}°C. Safe to visit."
        )
    else:
        rec     = "Avoid if possible." if safe_status == "UNSAFE" else "Proceed with caution."
        insight = f"{establishment_name} at {hour:02d}:00 is predicted to have concerns: {' and '.join(issues)}. {rec}"

    return safe_status, insight


async def generate_hourly_safety_insight(
    establishment_name: str,
    establishment_type: str,
    barangay_name:      str,
    hour:               int,
    prediction_date:    str,
    co2_ppm:            float,
    temperature_c:      float,
    humidity_percent:   float,
) -> tuple[str, str]:
    try:
        prompt                    = _build_hourly_predicted_prompt(
            establishment_name, establishment_type, barangay_name,
            hour, prediction_date, co2_ppm, temperature_c, humidity_percent,
        )
        response                  = await _call_llm(prompt)
        safe_status, insight_text = _parse_status_insight(response)
        logger.info(f"Hourly insight [{establishment_name} {hour:02d}:00]: {safe_status} via {LLM_BACKEND}")
        return safe_status, insight_text
    except Exception as e:
        logger.error(f"Hourly insight failed [{establishment_name} {hour:02d}:00]: {e}")
        return _fallback_hourly_insight(establishment_name, co2_ppm, temperature_c, humidity_percent, hour)


def _build_safety_prompt(
    establishment_name: str,
    establishment_type: str,
    barangay_name:      str,
    co2_ppm:            float,
    temperature_c:      float,
    humidity_percent:   float,
    heat_index_c:       float,
) -> str:
    return f"""You are an environmental health AI for Naga City, Philippines.

LIVE sensor readings from {establishment_name}:

ESTABLISHMENT:
  Name : {establishment_name}
  Type : {establishment_type}
  Area : {barangay_name}, Naga City

CURRENT READINGS (right now):
  CO2         : {co2_ppm:.0f} ppm ({co2_risk_level(co2_ppm)})
  Temperature : {temperature_c:.1f}°C
  Humidity    : {humidity_percent:.0f}%
  Heat Index  : {heat_index_c:.1f}°C (feels like — {comfort_label(heat_index_c)})

THRESHOLDS:
  CO2        : <800 = safe | 800–1000 = caution | >1000 = unsafe
  Heat Index : <32°C = safe | 32–41°C = caution | >41°C = unsafe

TASK:
1. Decide: SAFE, CAUTION, or UNSAFE
2. Write 2 sentences mentioning {establishment_name} by name,
   the specific readings, and a practical tip for visitors.

FORMAT:
STATUS: [SAFE or CAUTION or UNSAFE]
INSIGHT: [2 sentence explanation]
"""


def _fallback_safety_insight(
    establishment_name: str,
    establishment_type: str,
    co2_ppm:            float,
    temperature_c:      float,
    humidity_percent:   float,
    heat_index_c:       float,
) -> tuple[str, str]:
    issues      = []
    safe_status = "SAFE"

    if co2_ppm > 1000:
        safe_status = "UNSAFE"
        issues.append(f"CO2 at {co2_ppm:.0f} ppm")
    elif co2_ppm > 800:
        safe_status = "CAUTION"
        issues.append(f"CO2 at {co2_ppm:.0f} ppm")

    if heat_index_c > 41:
        safe_status = "UNSAFE"
        issues.append(f"heat index at {heat_index_c:.1f}°C")
    elif heat_index_c > 32:
        if safe_status != "UNSAFE":
            safe_status = "CAUTION"
        issues.append(f"heat index at {heat_index_c:.1f}°C")

    if not issues:
        insight = (
            f"{establishment_name} currently has good air quality with CO2 at "
            f"{co2_ppm:.0f} ppm and temperature at {temperature_c:.1f}°C. Safe to visit."
        )
    else:
        rec     = "Avoid if possible." if safe_status == "UNSAFE" else "Proceed with caution."
        insight = f"{establishment_name} has concerns: {' and '.join(issues)}. {rec}"

    return safe_status, insight


async def generate_safety_insight(
    establishment_name: str,
    establishment_type: str,
    barangay_name:      str,
    co2_ppm:            float,
    temperature_c:      float,
    humidity_percent:   float,
    heat_index_c:       float,
) -> tuple[str, str]:
    try:
        prompt                    = _build_safety_prompt(
            establishment_name, establishment_type, barangay_name,
            co2_ppm, temperature_c, humidity_percent, heat_index_c,
        )
        response                  = await _call_llm(prompt)
        safe_status, insight_text = _parse_status_insight(response)
        logger.info(f"Safety insight [{establishment_name}]: {safe_status}")
        return safe_status, insight_text
    except Exception as e:
        logger.error(f"Safety insight failed [{establishment_name}]: {e}")
        return _fallback_safety_insight(
            establishment_name, establishment_type,
            co2_ppm, temperature_c, humidity_percent, heat_index_c,
        )


async def generate_insight(
    sensor:         dict,
    current:        dict,
    predicted:      dict,
    peak_temp_hour: int = 13,
    peak_co2_hour:  int = 12,
    hours_ahead:    int = 24,
) -> str:
    sensor_id = sensor.get("sensor_id", "NODE-01")
    barangay  = sensor.get("barangay",  "Naga").title()
    co2       = float(predicted.get("co2_ppm",          415))
    temp      = float(predicted.get("temperature_c",    28))
    hum       = float(predicted.get("humidity_percent", 75))
    hi        = heat_index(temp, hum)

    prompt = f"""You are a climate monitoring AI for Naga City, Philippines.

Write ONE sentence summarizing today's forecast for {sensor_id} in {barangay}:
- CO2: {co2:.0f} ppm, peaks at {peak_co2_hour:02d}:00
- Temperature: {temp:.1f}°C, peaks at {peak_temp_hour:02d}:00 (feels like {hi:.1f}°C)
- Humidity: {hum:.0f}%

Write one clear sentence with the key values and one practical recommendation.
"""
    try:
        return await _call_llm(prompt)
    except Exception as e:
        logger.error(f"Daily insight failed: {e}")
        return (
            f"{sensor_id} in {barangay}: CO2 forecast {co2:.0f} ppm, "
            f"temperature peaks {temp:.1f}°C at {peak_temp_hour:02d}:00 "
            f"(feels like {hi:.1f}°C), humidity {hum:.0f}%."
        )