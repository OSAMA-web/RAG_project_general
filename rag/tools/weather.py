"""Live weather via Open-Meteo (https://open-meteo.com) — completely free, no API key required."""

import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODE_DESCRIPTIONS = {
    0: "clear sky", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow",
    80: "rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}


def get_weather(location_name: str) -> str:
    try:
        geo_response = requests.get(
            GEOCODE_URL, params={"name": location_name, "count": 1}, timeout=8
        )
        geo_response.raise_for_status()
        geo_results = geo_response.json().get("results") or []
    except Exception as e:
        return f'I couldn\'t look up the location "{location_name}" ({e}). Please try again shortly.'

    if not geo_results:
        return f'I couldn\'t find a location matching "{location_name}".'

    place = geo_results[0]
    lat, lon = place["latitude"], place["longitude"]
    resolved_name = place.get("name", location_name)

    try:
        weather_response = requests.get(
            FORECAST_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,precipitation,weather_code",
                "forecast_days": 1,
            },
            timeout=8,
        )
        weather_response.raise_for_status()
        current = weather_response.json().get("current", {})
    except Exception as e:
        return f"I couldn't fetch the weather for {resolved_name} ({e}). Please try again shortly."

    temp = current.get("temperature_2m")
    precipitation = current.get("precipitation", 0) or 0
    code = current.get("weather_code")
    condition = WEATHER_CODE_DESCRIPTIONS.get(code, "unknown conditions")

    # Neutral facts only — the domain framing ("could this delay flights?") comes from
    # the active profile's prompt when the LLM phrases the answer, not from the tool.
    lines = [f"Current weather in {resolved_name}: {condition}, {temp}°C."]
    if precipitation > 0:
        lines.append(f"Precipitation right now: {precipitation} mm.")
    else:
        lines.append("No precipitation currently reported.")

    return "\n".join(lines)
