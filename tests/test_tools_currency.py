from unittest.mock import MagicMock, patch

from rag.tools import currency


def test_convert_currency_same_currency_short_circuits_without_network_call():
    result = currency.convert_currency(100, "USD", "USD")
    assert "already in" in result


@patch("rag.tools.currency.requests.get")
def test_convert_currency_returns_converted_amount(mock_get):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"rates": {"USD": 600.5}}
    mock_get.return_value = mock_response

    result = currency.convert_currency(50000, "INR", "USD")
    assert "50,000.00 INR" in result
    assert "600.50 USD" in result


@patch("rag.tools.currency.requests.get")
def test_convert_currency_handles_unsupported_pair(mock_get):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"rates": {}}
    mock_get.return_value = mock_response

    result = currency.convert_currency(100, "USD", "XYZ")
    assert "couldn't convert" in result.lower()


@patch("rag.tools.currency.requests.get", side_effect=Exception("timeout"))
def test_convert_currency_handles_network_failure(mock_get):
    result = currency.convert_currency(100, "USD", "INR")
    assert "couldn't fetch" in result.lower()
