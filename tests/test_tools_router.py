import pytest

from rag import config
from rag.tools.router import (
    classify_intent,
    extract_currency_conversion,
    extract_flight_number,
    extract_location,
)


@pytest.fixture(autouse=True)
def _all_tools_enabled(monkeypatch):
    """Routing is gated by the active profile's enabled_tools — pin all three on
    so these tests don't depend on which profile happens to be active."""
    monkeypatch.setattr(config, "ENABLED_TOOLS", ["flight", "weather", "currency"])


def test_classify_intent_flight_examples_from_roadmap():
    assert classify_intent("Where is AI302?") == "flight"
    assert classify_intent("Is AI101 delayed?") == "flight"
    assert classify_intent("What is today's arrival time?") == "flight"


def test_classify_intent_weather_examples_from_roadmap():
    assert classify_intent("Will rain affect my flight?") == "weather"


def test_classify_intent_currency_examples_from_roadmap():
    assert classify_intent("Convert ₹50,000 to USD.") == "currency"


def test_classify_intent_defaults_to_rag_for_document_questions():
    assert classify_intent("How many aircraft does Air India have?") == "rag"
    assert classify_intent("What cabin classes does Air India offer?") == "rag"


def test_extract_flight_number_variants():
    assert extract_flight_number("Where is AI302?") == "AI302"
    assert extract_flight_number("Is AI 101 delayed?") == "AI101"
    assert extract_flight_number("How many aircraft does Air India have?") is None


def test_extract_location_finds_known_cities_case_insensitively():
    assert extract_location("Will rain affect my flight to Delhi?") == "Delhi"
    assert extract_location("What's the weather in london?") == "London"
    assert extract_location("How many aircraft does Air India have?") is None


def test_extract_currency_conversion_parses_amount_and_both_currencies():
    assert extract_currency_conversion("Convert ₹50,000 to USD.") == (50000.0, "INR", "USD")
    assert extract_currency_conversion("How much is 100 USD in INR?") == (100.0, "USD", "INR")


def test_extract_currency_conversion_returns_none_without_two_currencies():
    assert extract_currency_conversion("How many aircraft does Air India have?") is None
    assert extract_currency_conversion("I have 50000 of something") is None


# ---------- Precision: ordinary document questions must NOT be hijacked ----------
# A fast-path false positive skips retrieval entirely, so each of these used to get
# a live-data answer instead of an answer from the documents.


@pytest.mark.parametrize(
    "question",
    [
        "What happened to Air India in 1985?",           # "in 1985" used to look like flight IN1985
        "Is it true that the retirement age is 60?",     # "is 60" used to look like flight IS60
        "How many routes did Air India add in 2023?",
        "What were the FY 2024 results?",
        "Where is the Q3 2024 revenue reported?",
        "How many passengers were on the flight in 2020?",
        "Which aircraft does AI302 use?",                # flight number, but no live-status cue
        "What training do new employees get in Delhi?",  # "rain" inside "training"
        "What is the storage temperature in Mumbai?",    # condition word without any "now" cue
        "What is the revenue forecast for Delhi?",
        "Does the policy cover weather-related delays in Mumbai?",
        "How much is the baggage allowance for first class to Europe for 2 years?",  # RS/FIRST, EURO/EUROPE
    ],
)
def test_document_questions_are_not_routed_to_tools(question):
    assert classify_intent(question) == "rag"


def test_flight_number_needs_uppercase_known_code_or_the_word_flight():
    assert extract_flight_number("What happened in 2020?") is None
    assert extract_flight_number("Is XY123 on time?") is None       # unknown airline code
    assert extract_flight_number("Is flight XY123 on time?") == "XY123"
    assert extract_flight_number("Is 6E 2345 delayed?") == "6E2345"
    assert classify_intent("Is flight XY123 on time?") == "flight"


def test_weather_condition_words_need_a_live_cue():
    assert classify_intent("Is it raining in Mumbai right now?") == "weather"
    assert classify_intent("What's the weather in Delhi?") == "weather"


def test_extract_location_matches_whole_words_only():
    assert extract_location("Is there a comparison of fares?") is None  # "paris" inside "comparison"
    assert extract_location("Our goal is growth") is None                # "goa" inside "goal"
    assert extract_location("Flights from New Delhi") == "New Delhi"


def test_currency_words_match_whole_words_and_amount_must_start_with_a_digit():
    assert extract_currency_conversion("First class to Europe for 2 years") is None
    assert extract_currency_conversion("Convert, please, 100 USD to INR") == (100.0, "USD", "INR")


def test_classify_intent_only_considers_enabled_tools():
    assert classify_intent("Where is AI302?", enabled_tools=[]) == "rag"
    assert classify_intent("Where is AI302?", enabled_tools=["weather"]) == "rag"
    assert classify_intent("What's the weather in Delhi?", enabled_tools=["weather"]) == "weather"
    assert classify_intent("Convert 100 USD to INR", enabled_tools=["flight"]) == "rag"
