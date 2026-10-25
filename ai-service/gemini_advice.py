"""
AI Service — Gemini-powered health advice generation.
------------------------------------------------------
Extracted as a standalone module so the Gemini integration
lives in ai-service/ per the project structure requirements.

Usage from backend:
    from importlib.machinery import SourceFileLoader
    # or simply:  from ai_service.gemini_advice import get_health_advice
"""

import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Gemini client (initialised once at import time)
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


def is_available() -> bool:
    """Return True if Gemini client is configured."""
    return _client is not None


async def get_health_advice(
    aqi: float,
    category: str,
    pollutants: dict,
    user_info: dict | None = None,
) -> str | None:
    """
    Generate personalised health advice using Gemini.

    Parameters
    ----------
    aqi : float
        Predicted AQI value.
    category : str
        AQI category label (e.g. "Good", "Unhealthy").
    pollutants : dict
        Keys: pm2_5, pm10, co, no2, so2, o3, temperature, humidity.
    user_info : dict, optional
        Keys: age (int|None), health_condition (str), activity (str).

    Returns
    -------
    str or None
        Generated advice text, or None if Gemini is unavailable / errors.
    """
    if not _client:
        return None

    # Build user details string
    details = []
    if user_info:
        if user_info.get("age") is not None:
            details.append(f"Age: {user_info['age']}")
        hc = user_info.get("health_condition", "none")
        if hc and hc != "none":
            details.append(f"Health condition: {hc}")
        act = user_info.get("activity", "moderate")
        if act:
            details.append(f"Activity level: {act}")
    user_str = ", ".join(details) if details else "No specific details"

    prompt = f"""You are a health advisory AI. Based on the following air quality information,
provide brief, actionable health advice in 3-5 bullet points. Keep it short and simple.

AQI Value: {aqi:.0f}
AQI Category: {category}
Pollutant Levels:
  - PM2.5: {pollutants.get('pm2_5', 'N/A')} µg/m³
  - PM10: {pollutants.get('pm10', 'N/A')} µg/m³
  - CO: {pollutants.get('co', 'N/A')} mg/m³
  - NO₂: {pollutants.get('no2', 'N/A')} µg/m³
  - SO₂: {pollutants.get('so2', 'N/A')} µg/m³
  - O₃: {pollutants.get('o3', 'N/A')} µg/m³
  - Temperature: {pollutants.get('temperature', 'N/A')}°C
  - Humidity: {pollutants.get('humidity', 'N/A')}%

User Details: {user_str}

Provide personalised, practical health recommendations. Be concise."""

    try:
        response = _client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
        )
        return response.text or None
    except Exception as e:
        print(f"Gemini API error: {e}")
        return None
