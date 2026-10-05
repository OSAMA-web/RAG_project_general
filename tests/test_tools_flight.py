from unittest.mock import MagicMock, patch

from rag.tools import flight


def test_get_flight_status_without_api_key_returns_helpful_message(monkeypatch):
    monkeypatch.delenv("AVIATIONSTACK_API_KEY", raising=False)
    result = flight.get_flight_status("AI302")
    assert "API key" in result


@patch("rag.tools.flight.requests.get")
def test_get_flight_status_parses_successful_response(mock_get, monkeypatch):
    monkeypatch.setenv("AVIATIONSTACK_API_KEY", "fake-key")
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {
                "flight_status": "active",
                "departure": {"airport": "Delhi", "scheduled": "2026-01-01T10:00:00"},
                "arrival": {
                    "airport": "Mumbai",
                    "scheduled": "2026-01-01T12:00:00",
                    "estimated": "2026-01-01T12:15:00",
                    "delay": 15,
                },
            }
        ]
    }
    mock_get.return_value = mock_response

    result = flight.get_flight_status("AI302")
    assert "active" in result
    assert "Delhi" in result and "Mumbai" in result
    assert "15 minutes" in result


@patch("rag.tools.flight.requests.get")
def test_get_flight_status_handles_no_data_found(mock_get, monkeypatch):
    monkeypatch.setenv("AVIATIONSTACK_API_KEY", "fake-key")
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"data": []}
    mock_get.return_value = mock_response

    result = flight.get_flight_status("AI999")
    assert "couldn't find" in result.lower()


@patch("rag.tools.flight.requests.get", side_effect=Exception("timeout"))
def test_get_flight_status_handles_network_failure(mock_get, monkeypatch):
    monkeypatch.setenv("AVIATIONSTACK_API_KEY", "fake-key")
    result = flight.get_flight_status("AI302")
    assert "couldn't reach" in result.lower()
