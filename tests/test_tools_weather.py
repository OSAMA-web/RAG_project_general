from unittest.mock import MagicMock, patch

from rag.tools import weather


@patch("rag.tools.weather.requests.get")
def test_get_weather_returns_formatted_summary(mock_get):
    geo_response = MagicMock()
    geo_response.raise_for_status = MagicMock()
    geo_response.json.return_value = {
        "results": [{"name": "Delhi", "latitude": 28.6, "longitude": 77.2}]
    }

    weather_response = MagicMock()
    weather_response.raise_for_status = MagicMock()
    weather_response.json.return_value = {
        "current": {"temperature_2m": 22.5, "precipitation": 0, "weather_code": 1}
    }

    mock_get.side_effect = [geo_response, weather_response]

    result = weather.get_weather("Delhi")
    assert "Delhi" in result
    assert "22.5" in result
    assert "No precipitation" in result


@patch("rag.tools.weather.requests.get")
def test_get_weather_reports_precipitation_neutrally(mock_get):
    geo_response = MagicMock()
    geo_response.raise_for_status = MagicMock()
    geo_response.json.return_value = {
        "results": [{"name": "Mumbai", "latitude": 19.0, "longitude": 72.8}]
    }

    weather_response = MagicMock()
    weather_response.raise_for_status = MagicMock()
    weather_response.json.return_value = {
        "current": {"temperature_2m": 26.0, "precipitation": 4.2, "weather_code": 61}
    }

    mock_get.side_effect = [geo_response, weather_response]

    result = weather.get_weather("Mumbai")
    assert "Precipitation right now: 4.2 mm" in result
    assert "flights" not in result  # domain framing belongs to the profile's prompt, not the tool


@patch("rag.tools.weather.requests.get")
def test_get_weather_handles_unknown_location(mock_get):
    geo_response = MagicMock()
    geo_response.raise_for_status = MagicMock()
    geo_response.json.return_value = {"results": []}
    mock_get.return_value = geo_response

    result = weather.get_weather("Nowhereville")
    assert "couldn't find" in result.lower()


@patch("rag.tools.weather.requests.get", side_effect=Exception("timeout"))
def test_get_weather_handles_network_failure(mock_get):
    result = weather.get_weather("Delhi")
    assert "couldn't look up" in result.lower()
