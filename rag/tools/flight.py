"""Live flight status for any airline via AviationStack (https://aviationstack.com).

Requires a free API key set as the AVIATIONSTACK_API_KEY environment
variable — sign up free at aviationstack.com (no credit card, 100
requests/month on the free tier). Without a key configured, this returns a
clear explanation instead of crashing, so the rest of the app keeps working.
"""

import os

import requests

API_URL = "http://api.aviationstack.com/v1/flights"


def get_flight_status(flight_number: str) -> str:
    api_key = os.environ.get("AVIATIONSTACK_API_KEY")
    if not api_key:
        return (
            f"I can't check live status for {flight_number} because no flight-data API key is "
            "configured. Set the AVIATIONSTACK_API_KEY environment variable with a free key from "
            "aviationstack.com to enable this."
        )

    try:
        response = requests.get(
            API_URL,
            params={"access_key": api_key, "flight_iata": flight_number},
            timeout=8,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as e:
        return (
            f"I couldn't reach the flight-status service for {flight_number} ({e}). "
            "Please try again shortly."
        )

    flights = payload.get("data") or []
    if not flights:
        return f"I couldn't find live data for flight {flight_number}. Double-check the flight number and try again."

    flight = flights[0]
    status = flight.get("flight_status", "unknown")
    departure = flight.get("departure") or {}
    arrival = flight.get("arrival") or {}

    dep_airport = departure.get("airport", "an unknown airport")
    arr_airport = arrival.get("airport", "an unknown airport")
    scheduled_arrival = arrival.get("scheduled", "an unknown time")
    estimated_arrival = arrival.get("estimated") or scheduled_arrival
    delay_minutes = arrival.get("delay")

    lines = [
        f"Flight {flight_number} ({dep_airport} → {arr_airport}) is currently **{status}**.",
        f"Scheduled arrival: {scheduled_arrival}",
    ]
    if estimated_arrival and estimated_arrival != scheduled_arrival:
        lines.append(f"Estimated arrival: {estimated_arrival}")
    if delay_minutes:
        lines.append(f"Delay: {delay_minutes} minutes")

    return "\n".join(lines)
