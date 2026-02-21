import asyncio
import logging
import os

import httpx
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
        hi = (
            -42.379
            + 2.04901523  * t
            + 10.14333127 * rh
            - 0.22475541  * t  * rh
            - 0.00683783  * t  * t
            - 0.05481717  * rh * rh
            + 0.00122874  * t  * t  * rh
            + 0.00085282  * t  * rh * rh
            - 0.00000199  * t  * t  * rh * rh
        )
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
    async with httpx.AsyncClient(timeout=30.0) as client:
        for attempt in range(3):
            try:
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type":  "application/json",
                    },
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
    return await (_call_groq(prompt) if LLM_BACKEND.lower() == "groq" else _call_ollama(prompt))


def _parse_status_insight(response: str) -> tuple[str, str]:
    safe_status  = "CAUTION"
    insight_text = response.strip()

    for line in response.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("STATUS:"):
            raw = line.split(":", 1)[1].strip().upper()
            if "UNSAFE" in raw:
                safe_status = "UNSAFE"
            elif "SAFE" in raw and "CAUTION" not in raw:
                safe_status = "SAFE"
        elif line.upper().startswith("INSIGHT:"):
            insight_text = line.split(":", 1)[1].strip()

    return safe_status, insight_text


def _resolve_status(co2_ppm: float, hi: float) -> str:
    if co2_ppm > 1000 or hi > 41:
        return "UNSAFE"
    if co2_ppm > 800 or hi > 32:
        return "CAUTION"
    return "SAFE"


def _fallback_insight_text(
    name:       str,
    co2_ppm:    float,
    temp_c:     float,
    hi:         float,
    status:     str,
    label:      str = "",
) -> str:
    issues = []
    if co2_ppm > 1000:
        issues.append("the air is very stuffy")
    elif co2_ppm > 800:
        issues.append("the air is getting a little stuffy")
    if hi > 41:
        issues.append("it feels dangerously hot")
    elif hi > 32:
        issues.append("it feels quite warm")

    prefix = f"{name}{(' ' + label) if label else ''}"
    if not issues:
        return f"{prefix} has fresh air and comfortable temperatures. Safe to visit — enjoy your time there."
    rec = "It is best to avoid this area if possible." if status == "UNSAFE" else "Please take extra care during your visit."
    return f"{prefix} has some concerns — {' and '.join(issues)}. {rec} Stay hydrated and take breaks as needed."


def _build_safety_prompt(
    name:     str,
    est_type: str,
    barangay: str,
    co2_ppm:  float,
    temp_c:   float,
    humidity: float,
    hi:       float,
) -> str:
    return f"""You are a helpful health advisor for residents of Naga City, Philippines.

LOCATION: {name} ({est_type}), {barangay}, Naga City

CURRENT CONDITIONS:
- CO2 level       : {co2_ppm:.0f} ppm
- Temperature     : {temp_c:.1f}°C
- Feels-like temp : {hi:.1f}°C
- Humidity        : {humidity:.0f}%

THRESHOLDS:
- CO2        : <800 = safe | 800–1000 = caution | >1000 = unsafe
- Heat Index : <32°C = safe | 32–41°C = caution | >41°C = unsafe
- Humidity   : <85% = fine | 85–95% = caution | >95% = uncomfortable

YOUR TASK:
1. Decide the safety status: SAFE, CAUTION, or UNSAFE
2. Write 2–3 sentences in plain English that anyone can understand — not too
   technical, not too simple. Mention {name} by name, describe how the air
   and heat feel right now, who should be careful, and give one practical health tip.

RESPOND IN THIS EXACT FORMAT:
STATUS: [SAFE or CAUTION or UNSAFE]
INSIGHT: [2–3 sentences in plain, easy-to-understand English]
"""


def _build_hourly_prompt(
    name:            str,
    est_type:        str,
    barangay:        str,
    hour:            int,
    prediction_date: str,
    co2_ppm:         float,
    temp_c:          float,
    humidity:        float,
    hi:              float,
) -> str:
    time_str = f"{hour:02d}:00"
    return f"""You are a helpful health advisor for residents of Naga City, Philippines.

LOCATION: {name} ({est_type}), {barangay}, Naga City
FORECAST FOR: {prediction_date} at {time_str}

PREDICTED CONDITIONS AT {time_str}:
- CO2 level       : {co2_ppm:.0f} ppm
- Temperature     : {temp_c:.1f}°C
- Feels-like temp : {hi:.1f}°C
- Humidity        : {humidity:.0f}%

THRESHOLDS:
- CO2        : <800 = safe | 800–1000 = caution | >1000 = unsafe
- Heat Index : <32°C = safe | 32–41°C = caution | >41°C = unsafe
- Humidity   : <85% = fine | 85–95% = caution | >95% = uncomfortable

YOUR TASK:
1. Decide the safety status: SAFE, CAUTION, or UNSAFE
2. Write 2–3 sentences in plain English that anyone can understand — not too
   technical, not too simple. Mention {name} by name, describe how the air
   and heat are expected to feel at {time_str}, who should be careful, and
   give one practical health tip.

RESPOND IN THIS EXACT FORMAT:
STATUS: [SAFE or CAUTION or UNSAFE]
INSIGHT: [2–3 sentences in plain, easy-to-understand English]
"""


async def _generate(prompt: str, log_tag: str) -> tuple[str, str] | str:
    response = await _call_llm(prompt)
    logger.info(f"[{log_tag}] LLM response received via {LLM_BACKEND}")
    return response


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
        prompt               = _build_safety_prompt(
            establishment_name, establishment_type, barangay_name,
            co2_ppm, temperature_c, humidity_percent, heat_index_c,
        )
        response             = await _call_llm(prompt)
        status, insight      = _parse_status_insight(response)
        logger.info(f"Safety insight [{establishment_name}]: {status}")
        return status, insight
    except Exception as e:
        logger.error(f"Safety insight failed [{establishment_name}]: {e}")
        status  = _resolve_status(co2_ppm, heat_index_c)
        insight = _fallback_insight_text(establishment_name, co2_ppm, temperature_c, heat_index_c, status)
        return status, insight


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
    hi = heat_index(temperature_c, humidity_percent)
    try:
        prompt          = _build_hourly_prompt(
            establishment_name, establishment_type, barangay_name,
            hour, prediction_date, co2_ppm, temperature_c, humidity_percent, hi,
        )
        response        = await _call_llm(prompt)
        status, insight = _parse_status_insight(response)
        logger.info(f"Hourly insight [{establishment_name} {hour:02d}:00]: {status} via {LLM_BACKEND}")
        return status, insight
    except Exception as e:
        logger.error(f"Hourly insight failed [{establishment_name} {hour:02d}:00]: {e}")
        status  = _resolve_status(co2_ppm, hi)
        insight = _fallback_insight_text(establishment_name, co2_ppm, temperature_c, hi, status, f"at {hour:02d}:00")
        return status, insight


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

    prompt = f"""You are a helpful health advisor for residents of Naga City, Philippines.

LOCATION: {sensor_id}, {barangay}, Naga City

TODAY'S FORECAST:
- CO2 level       : {co2:.0f} ppm (expected to be worst around {peak_co2_hour:02d}:00)
- Temperature     : {temp:.1f}°C (expected to peak around {peak_temp_hour:02d}:00)
- Feels-like temp : {hi:.1f}°C
- Humidity        : {hum:.0f}%

YOUR TASK:
Write 2–3 sentences in plain English that anyone can understand — not too
technical, not too simple. Summarize how the air and heat will feel today,
when it will be at its worst, and give one practical health tip for residents.

RESPOND IN THIS EXACT FORMAT:
INSIGHT: [2–3 sentences in plain, easy-to-understand English]
"""
    try:
        return await _call_llm(prompt)
    except Exception as e:
        logger.error(f"Daily insight failed: {e}")
        return (
            f"INSIGHT: Today in {barangay}, the air may feel a little stuffy around "
            f"{peak_co2_hour:02d}:00 and the heat will peak at {temp:.1f}°C "
            f"(feels like {hi:.1f}°C) around {peak_temp_hour:02d}:00. "
            f"Stay hydrated and avoid staying too long outdoors during the hottest part of the day."
        )