"""Agricultural Weather Service Module.

Provides live weather observations, 5-day forecasts, and agricultural weather
advisories using the open Open-Meteo Meteorological API (No API key required).
Also supports OpenWeatherMap if an API key is configured.
"""

import os
import requests
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("weather_service")

# WMO Weather interpretation codes
WMO_DESCRIPTIONS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow fall",
    75: "Heavy snow fall",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail"
}


def geocode_location(location_name: str, timeout: int = 5) -> Optional[Dict[str, Any]]:
    """Resolves location name into latitude and longitude coordinates."""
    try:
        url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {
            "name": location_name.strip(),
            "count": 1,
            "language": "en",
            "format": "json"
        }
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            if "results" in data and len(data["results"]) > 0:
                res = data["results"][0]
                return {
                    "name": res.get("name"),
                    "latitude": res.get("latitude"),
                    "longitude": res.get("longitude"),
                    "country": res.get("country", ""),
                    "admin1": res.get("admin1", "")  # State/Province
                }
    except Exception as e:
        logger.warning(f"Geocoding error for '{location_name}': {e}")
    return None


def fetch_open_meteo_weather(lat: float, lon: float, location_label: str) -> Optional[Dict[str, Any]]:
    """Fetches real current weather and 5-day daily forecast from Open-Meteo."""
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,weather_code",
            "timezone": "auto",
            "forecast_days": 5
        }
        resp = requests.get(url, params=params, timeout=6)
        if resp.status_code == 200:
            data = resp.json()
            curr = data.get("current", {})
            daily = data.get("daily", {})

            w_code = curr.get("weather_code", 0)
            condition = WMO_DESCRIPTIONS.get(w_code, "Variable conditions")

            current_temp = curr.get("temperature_2m")
            humidity = curr.get("relative_humidity_2m")
            wind_speed = curr.get("wind_speed_10m")
            precip = curr.get("precipitation")

            # Daily forecast summaries
            daily_forecasts = []
            dates = daily.get("time", [])
            max_temps = daily.get("temperature_2m_max", [])
            min_temps = daily.get("temperature_2m_min", [])
            rain_probs = daily.get("precipitation_probability_max", [])
            rain_sums = daily.get("precipitation_sum", [])
            d_codes = daily.get("weather_code", [])

            for i in range(len(dates)):
                d_cond = WMO_DESCRIPTIONS.get(d_codes[i] if i < len(d_codes) else 0, "Fair")
                daily_forecasts.append({
                    "date": dates[i],
                    "max_temp": max_temps[i] if i < len(max_temps) else None,
                    "min_temp": min_temps[i] if i < len(min_temps) else None,
                    "rain_prob": rain_probs[i] if i < len(rain_probs) else 0,
                    "rain_sum_mm": rain_sums[i] if i < len(rain_sums) else 0,
                    "condition": d_cond
                })

            # Formulate Agricultural Weather Advisory
            advisory_points = []
            max_upcoming_rain_prob = max(rain_probs) if rain_probs else 0

            if max_upcoming_rain_prob > 50:
                advisory_points.append("⚠️ **High Rain Probability**: Postpone foliar spraying of pesticides and fertilizers; ensure drainage channels in low-lying crop fields are clear.")
            elif max_upcoming_rain_prob > 25:
                advisory_points.append("🌦️ **Moderate Rain Risk**: Monitor skies before undertaking major harvesting or outdoor grain drying activities.")
            else:
                advisory_points.append("☀️ **Dry Weather Expected**: Favorable conditions for land preparation, weeding, and planned spraying operations.")

            if current_temp and current_temp > 37:
                advisory_points.append("🌡️ **High Temperature Advisory**: Provide light and frequent irrigation to vegetable crops to prevent heat stress and blossom drop.")
            elif current_temp and current_temp < 5:
                advisory_points.append("❄️ **Frost Risk Warning**: Sensitive vegetables and mustard may face cold injury; maintain soil moisture and create smoke mulches in late evening.")

            if wind_speed and wind_speed > 16:
                advisory_points.append("💨 **High Wind Advisory**: Avoid chemical spraying to prevent drift losses and uneven distribution.")

            return {
                "location": location_label,
                "current_temperature": current_temp,
                "humidity": humidity,
                "wind_speed_kmh": wind_speed,
                "current_condition": condition,
                "precipitation_mm": precip,
                "daily_forecast": daily_forecasts,
                "agricultural_advisory": advisory_points,
                "source": "Open-Meteo Meteorological Service",
                "source_url": "https://open-meteo.com/",
                "status": "LIVE_SUCCESS"
            }
    except Exception as e:
        logger.error(f"Weather forecast retrieval error: {e}")
    return None


def get_live_weather(location_name: Optional[str] = None) -> Dict[str, Any]:
    """Retrieves live weather data and agricultural advisories for a specified location.

    If location is omitted or cannot be resolved, returns an informative prompt without guessing.
    """
    if not location_name or not location_name.strip():
        return {
            "status": "LOCATION_REQUIRED",
            "message": "Please specify a location (e.g., 'weather in Coimbatore', 'weather in Ludhiana', or select in the weather panel).",
            "source_url": "https://imdagrimet.gov.in/"
        }

    loc_info = geocode_location(location_name.strip())
    if not loc_info:
        return {
            "status": "LOCATION_NOT_FOUND",
            "message": f"Could not pinpoint coordinates for '{location_name}'. Please verify the district or town name.",
            "source_url": "https://imdagrimet.gov.in/"
        }

    label = f"{loc_info['name']}, {loc_info['admin1']}, {loc_info['country']}".strip(", ")
    weather_data = fetch_open_meteo_weather(loc_info["latitude"], loc_info["longitude"], label)

    if not weather_data:
        return {
            "status": "SERVICE_UNAVAILABLE",
            "message": "Live weather service is currently unreachable. Please consult the India Meteorological Department Agrimet portal.",
            "source_url": "https://imdagrimet.gov.in/"
        }

    return weather_data

