"""Live currency conversion via Frankfurter (https://frankfurter.app) — free, no API key required."""

import requests

API_URL = "https://api.frankfurter.app/latest"


def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    if from_currency == to_currency:
        return f"{amount:,.2f} {from_currency} is already in {to_currency}."

    try:
        response = requests.get(
            API_URL,
            params={"amount": amount, "from": from_currency, "to": to_currency},
            timeout=8,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as e:
        return f"I couldn't fetch a live exchange rate right now ({e}). Please try again shortly."

    rates = payload.get("rates", {})
    converted = rates.get(to_currency)
    if converted is None:
        return f"I couldn't convert {from_currency} to {to_currency} — that currency pair isn't supported."

    return f"{amount:,.2f} {from_currency} ≈ {converted:,.2f} {to_currency} (live exchange rate)."
