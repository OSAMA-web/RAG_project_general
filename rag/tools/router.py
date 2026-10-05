"""Lightweight keyword/pattern-based intent routing (Phase 5).

Decides whether a question should go to the RAG pipeline or one of the live
data tools (flight status, weather, currency conversion). Intentionally
simple — regex/keyword rules rather than an LLM call — so routing is instant
and deterministic, with no extra Ollama round-trip before every answer.

Design rule: this is a FAST PATH, so it must be high-precision. A false positive
skips document retrieval entirely (the user gets a live-data answer to a
document question), while a false negative only costs one extra LLM call — the
agent fallback in rag/agent.py has the same tools and can still route it. So
every rule below errs on the side of NOT firing:

  - Keywords and city names match whole words only ("rain" is not in "training",
    "paris" is not in "comparison", "goa" is not in "goal").
  - A tool fires only when the question has BOTH the thing to look up (a flight
    number, a city, an amount + two currencies) AND a sign the user wants live
    data ("where is", "delayed", "right now", "convert", ...). "What happened to
    AI171?" is a history question for the documents, not a status check.
  - A flight number must look like one: an UPPERCASE airline code that is either
    known ("AI302") or directly follows the word "flight" ("flight XY123"). So
    "in 2020", "age is 60" and "FY 2024" are never mistaken for flights.
  - Currency names match whole words ("RS" is not in "first", "EURO" is not in "Europe").

classify_intent() only considers the tools the active profile enables.
"""

import re
from typing import Iterable, List, Optional, Tuple

from .. import config


def _phrase_pattern(phrases: Iterable[str]) -> re.Pattern:
    """One case-insensitive regex matching any of the phrases as whole words."""
    alternatives = "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE)


# ---------- Flight ----------

# IATA airline designators that are safe to recognise on their own ("Is AI302 delayed?").
# Any other code only counts after the word "flight" ("Is flight XY123 delayed?"),
# which keeps look-alikes such as "Q3 2024", "FY 2024" or "NH 48" out.
KNOWN_AIRLINE_CODES = {"AI", "IX", "6E", "SG", "QP", "EK", "QR", "EY", "SQ", "BA", "LH"}

_FLIGHT_TOKEN = r"([A-Z][A-Z0-9]|[0-9][A-Z])[\s-]?(\d{1,4})"   # case-SENSITIVE on purpose
FLIGHT_NUMBER_PATTERN = re.compile(rf"\b{_FLIGHT_TOKEN}\b")
EXPLICIT_FLIGHT_PATTERN = re.compile(rf"\b(?i:flight)\s+(?:#\s*)?{_FLIGHT_TOKEN}\b")

# Phrases that on their own mean "the user is asking about a flight's live status".
FLIGHT_KEYWORDS = _phrase_pattern([
    "flight status", "where is my flight", "is my flight", "arrival time", "departure time",
    "flight tracker", "is flight",
])
# Words that turn a bare flight number into a live-status question.
FLIGHT_STATUS_CUES = _phrase_pattern([
    "status", "where is", "where's", "delayed", "delay", "on time", "late", "arrival", "arriving",
    "arrive", "departure", "departing", "depart", "landed", "landing", "took off", "gate",
    "track", "tracking", "cancelled", "canceled", "diverted", "eta",
])

# ---------- Weather ----------

# Phrases that clearly ask for current weather on their own.
WEATHER_PHRASES = _phrase_pattern([
    "weather in", "weather at", "weather for", "weather like", "weather today", "weather now",
    "weather forecast", "what's the weather", "how's the weather",
])
# Condition words are ambiguous ("storage temperature", "rain damage in 2005") — they
# only signal weather together with a sign the user means now / soon.
WEATHER_CONDITION_WORDS = _phrase_pattern([
    "rain", "raining", "rainy", "snow", "snowing", "storm", "stormy", "thunderstorm",
    "temperature", "humid", "humidity",
])
LIVE_CUES = _phrase_pattern([
    "now", "right now", "today", "tonight", "tomorrow", "currently", "this morning",
    "this evening", "this week", "will", "is it", "going to",
])

# A modest set of major Indian + international cities. Not exhaustive — the agent
# fallback can still call the weather tool for any city the LLM recognises.
KNOWN_CITIES = [
    "new delhi", "delhi", "mumbai", "bangalore", "bengaluru", "chennai", "kolkata",
    "hyderabad", "ahmedabad", "pune", "goa", "kochi", "jaipur", "lucknow", "amritsar",
    "jamnagar", "vadodara", "jammu",
    "london", "new york", "dubai", "singapore", "toronto", "paris", "frankfurt",
    "sydney", "melbourne", "tokyo", "tel aviv", "san francisco", "chicago",
    "washington", "vancouver", "hong kong", "bangkok",
]
CITY_PATTERN = _phrase_pattern(KNOWN_CITIES)

# ---------- Currency ----------

CURRENCY_ALIASES = {
    "₹": "INR", "RS": "INR", "RUPEES": "INR", "RUPEE": "INR", "INR": "INR",
    "$": "USD", "USD": "USD", "DOLLARS": "USD", "DOLLAR": "USD",
    "€": "EUR", "EUR": "EUR", "EUROS": "EUR", "EURO": "EUR",
    "£": "GBP", "GBP": "GBP", "POUNDS": "GBP", "POUND": "GBP",
}
# Symbols can sit right next to digits ("₹500"); words must stand alone.
_CURRENCY_ALIAS_PATTERNS = [
    (re.compile(re.escape(alias) if not alias.isalpha() else rf"(?<![A-Z]){alias}(?![A-Z])"), code)
    for alias, code in CURRENCY_ALIASES.items()
]
CURRENCY_INTENT_KEYWORDS = _phrase_pattern([
    "convert", "exchange", "how much is", "in usd", "in inr", "in eur", "in gbp",
    "to usd", "to inr", "to eur", "to gbp",
])
AMOUNT_PATTERN = re.compile(r"\d[\d,]*(?:\.\d+)?")  # must START with a digit (a lone "," isn't an amount)


# ---------- Extractors ----------


def extract_flight_number(question: str) -> Optional[str]:
    """Pulls a flight number (e.g. "AI302") out of the question, if one is present."""
    for match in FLIGHT_NUMBER_PATTERN.finditer(question):
        if match.group(1) in KNOWN_AIRLINE_CODES:
            return f"{match.group(1)}{match.group(2)}"
    match = EXPLICIT_FLIGHT_PATTERN.search(question)
    if match:
        return f"{match.group(1)}{match.group(2)}"
    return None


def extract_location(question: str) -> Optional[str]:
    """Finds the first known city mentioned in the question (whole words only), if any."""
    match = CITY_PATTERN.search(question)
    return match.group(0).title() if match else None


def extract_currency_conversion(question: str) -> Optional[Tuple[float, str, str]]:
    """Parses an amount + source/target currency from the question, if present.

    Returns (amount, from_code, to_code) in the order the currencies were
    mentioned, e.g. "Convert ₹50,000 to USD" -> (50000.0, "INR", "USD").
    """
    q = question.upper()
    amount_match = AMOUNT_PATTERN.search(q)
    if not amount_match:
        return None
    try:
        amount = float(amount_match.group().replace(",", ""))
    except ValueError:
        return None

    found = []
    for pattern, code in _CURRENCY_ALIAS_PATTERNS:
        match = pattern.search(q)
        if match:
            found.append((match.start(), code))
    found.sort()

    codes_in_order: List[str] = []
    for _, code in found:
        if code not in codes_in_order:
            codes_in_order.append(code)

    if len(codes_in_order) < 2:
        return None
    return amount, codes_in_order[0], codes_in_order[1]


# ---------- Intent ----------


def _is_flight_question(question: str) -> bool:
    if FLIGHT_KEYWORDS.search(question):
        return True
    return bool(extract_flight_number(question)) and bool(FLIGHT_STATUS_CUES.search(question))


def _is_weather_question(question: str) -> bool:
    if WEATHER_PHRASES.search(question):
        return True
    return bool(WEATHER_CONDITION_WORDS.search(question)) and bool(LIVE_CUES.search(question))


def _is_currency_question(question: str) -> bool:
    return bool(extract_currency_conversion(question)) and bool(CURRENCY_INTENT_KEYWORDS.search(question))


def classify_intent(question: str, enabled_tools: Optional[Iterable[str]] = None) -> str:
    """Returns one of: 'flight', 'weather', 'currency', 'rag'.

    Only tools in `enabled_tools` (default: the active profile's) are considered,
    so a profile without, say, the flight tool can never be routed to it.
    """
    enabled = set(config.ENABLED_TOOLS if enabled_tools is None else enabled_tools)

    if "currency" in enabled and _is_currency_question(question):
        return "currency"
    if "flight" in enabled and _is_flight_question(question):
        return "flight"
    if "weather" in enabled and _is_weather_question(question):
        return "weather"
    return "rag"
