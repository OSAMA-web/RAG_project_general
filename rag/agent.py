"""AI Agent (Phase 6): implements the roadmap's decision flow —

    Question -> Intent Classification -> (RAG | Tool Calling) -> Ollama -> Final Response

Two-tier design, kept for efficiency:
  1. FAST PATH — Phase 5's regex/keyword classifier (rag/tools/router.py) resolves
     the clear-cut cases (explicit flight numbers, currency symbols, weather words)
     with zero extra LLM calls, and critically SKIPS running the Phase 3 hybrid
     RAG retrieval pipeline entirely for these, since the answer was never going
     to use retrieved documents anyway.
  2. AGENT FALLBACK — anything the fast path can't confidently resolve still runs
     full RAG retrieval as before, but the LLM is given real tool-calling ability
     (LangChain's bind_tools) as a safety net, so a cleverly-phrased live-data
     question the regex missed can still get routed correctly.

Either way, Ollama always produces the final wording — even a fast-path tool
hit gets phrased naturally by the LLM rather than returned as a raw templated
string, matching the diagram (both branches feed into "Ollama" before the
"Final Response").

Domain-agnostic: who the assistant is and what it's about come from the active
profile (config.APP_NAME / config.DOMAIN_DESCRIPTION), and only the tools the
profile lists in "enabled_tools" are bound to the LLM or tried on the fast path.

Model support: tool calling needs an Ollama model that supports it (Llama 3.1+,
Qwen2.5+/Qwen3, ...). For a model without it (e.g. llama3, phi3), Ollama rejects
every request that includes tools with "<model> does not support tools". The
agent catches exactly that error once, logs a warning, and from then on answers
from the retrieved context with the plain LLM. The fast path is unaffected,
since it calls the tools itself and only uses the LLM to phrase the result.
"""

import logging
from typing import Iterable, Iterator, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from . import config
from .generation import llm
from .tools.currency import convert_currency
from .tools.flight import get_flight_status
from .tools.router import (
    classify_intent,
    extract_currency_conversion,
    extract_flight_number,
    extract_location,
)
from .tools.weather import get_weather

logger = logging.getLogger("rag_assistant")


# ---------- Tools the agent fallback can call ----------

@tool
def flight_status_tool(flight_number: str) -> str:
    """Get the live status of a commercial flight (on-time, delayed, arrival time).
    Use this whenever the user asks about a specific flight's current status or location.

    Args:
        flight_number: IATA flight number, e.g. "AI302" or "EK500".
    """
    return get_flight_status(flight_number)


@tool
def weather_tool(location: str) -> str:
    """Get the current live weather for a city, including any precipitation right now.

    Args:
        location: City name, e.g. "Delhi".
    """
    return get_weather(location)


@tool
def currency_conversion_tool(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount between currencies using a live exchange rate.

    Args:
        amount: Numeric amount to convert.
        from_currency: 3-letter source currency code, e.g. "INR".
        to_currency: 3-letter target currency code, e.g. "USD".
    """
    return convert_currency(amount, from_currency, to_currency)


# Profile-facing name (the "enabled_tools" entries in profile.json) -> LangChain tool.
TOOL_REGISTRY = {
    "flight": flight_status_tool,
    "weather": weather_tool,
    "currency": currency_conversion_tool,
}
# How each tool is described to the LLM in the system prompt.
TOOL_PURPOSES = {
    "flight": "live flight status",
    "weather": "current weather",
    "currency": "live currency conversion",
}


def build_agent_tools(enabled_tools: Iterable[str]) -> List:
    """The LangChain tools to bind for a profile, in the order the profile lists them."""
    names = list(enabled_tools)
    unknown = [n for n in names if n not in TOOL_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown tool(s) {unknown}. Available: {sorted(TOOL_REGISTRY)}")
    return [TOOL_REGISTRY[n] for n in names]


AGENT_TOOLS = build_agent_tools(config.ENABLED_TOOLS)
TOOL_BY_NAME = {t.name: t for t in AGENT_TOOLS}
# No tools enabled -> plain LLM: the "agent" simply answers from the retrieved context.
agent_llm = llm.bind_tools(AGENT_TOOLS) if AGENT_TOOLS else llm

# Flips to True the first time Ollama says the model "does not support tools".
_tool_calling_disabled = False

FALLBACK_SYSTEM_PROMPT = """You are {app_name}, an assistant that answers questions about {domain}.
{grounding}
Use the conversation history only to understand what the user is referring to (e.g. "it", "that one", \
"those numbers") — it is not a source of facts.

Conversation history:
{history}

Context:
{context}
"""

GROUNDING_WITH_TOOLS = (
    "Answer using the context below. If the question genuinely needs real-time data that isn't in "
    "the context ({purposes}), call the matching tool and answer from its result. If neither the "
    "context nor a tool gives you the answer, say you don't have that information — do not guess."
)
GROUNDING_CONTEXT_ONLY = (
    "Answer using ONLY the context below. If it doesn't contain the answer, say you don't have "
    "that information — do not guess."
)

PHRASING_PROMPT = """You are {app_name}, an assistant that answers questions about {domain}. You already \
looked up this live data for the user's question — phrase it as a natural, conversational answer. Don't just \
repeat it verbatim; respond as if you're answering the question directly.

Conversation history:
{history}

Live data retrieved: {tool_result}

User's question: {question}
"""


def _fallback_system_prompt(context: str, history: str, use_tools: bool = True) -> str:
    enabled = [n for n in config.ENABLED_TOOLS if n in TOOL_PURPOSES] if use_tools else []
    if enabled:
        grounding = GROUNDING_WITH_TOOLS.format(purposes=", ".join(TOOL_PURPOSES[n] for n in enabled))
    else:
        grounding = GROUNDING_CONTEXT_ONLY
    return FALLBACK_SYSTEM_PROMPT.format(
        app_name=config.APP_NAME,
        domain=config.DOMAIN_DESCRIPTION,
        grounding=grounding,
        history=history or "(none yet)",
        context=context,
    )


def _phrasing_prompt(question: str, tool_result: str, history: str) -> str:
    return PHRASING_PROMPT.format(
        app_name=config.APP_NAME,
        domain=config.DOMAIN_DESCRIPTION,
        history=history or "(none yet)",
        tool_result=tool_result,
        question=question,
    )


def fast_path_tool_answer(question: str, enabled_tools: Optional[Iterable[str]] = None) -> Optional[str]:
    """Phase 5's cheap regex classifier — resolves the clear-cut cases without
    any LLM call. Returns the raw tool result, or None if nothing matched
    confidently (including when an intent was detected but a needed argument,
    like a flight number, couldn't be extracted — that's left to the agent
    fallback, which can ask a clarifying question). Only the tools the profile
    enables (or `enabled_tools`, if given) are ever tried.
    """
    intent = classify_intent(question, enabled_tools)

    if intent == "flight":
        flight_number = extract_flight_number(question)
        return get_flight_status(flight_number) if flight_number else None

    if intent == "weather":
        location = extract_location(question)
        return get_weather(location) if location else None

    if intent == "currency":
        parsed = extract_currency_conversion(question)
        if parsed:
            amount, from_code, to_code = parsed
            return convert_currency(amount, from_code, to_code)
        return None

    return None


def phrase_fast_path_answer(question: str, tool_result: str, history: str) -> str:
    prompt = _phrasing_prompt(question, tool_result, history)
    return llm.invoke(prompt).content


def stream_phrase_fast_path_answer(question: str, tool_result: str, history: str) -> Iterator[str]:
    prompt = _phrasing_prompt(question, tool_result, history)
    for chunk in llm.stream(prompt):
        if chunk.content:
            yield chunk.content


def _execute_tool_calls(messages: List, ai_message) -> None:
    """Runs every tool the agent requested and appends each result as a ToolMessage."""
    messages.append(ai_message)
    for call in ai_message.tool_calls:
        tool_obj = TOOL_BY_NAME.get(call["name"])
        if tool_obj is None:
            result = f"Unknown tool requested: {call['name']}"
        else:
            try:
                result = tool_obj.invoke(call["args"])
            except Exception as e:
                result = f"Tool execution failed: {e}"
        messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))


def _is_tools_unsupported_error(error: Exception) -> bool:
    return "does not support tools" in str(error).lower()


def _first_agent_call(question: str, context: str, history: str):
    """The agent's first LLM call. Returns (messages, ai_message).

    Normally uses the tool-enabled LLM. If Ollama rejects the request because the
    model has no tool support, remembers that and answers with the plain LLM and a
    context-only prompt instead — now and on every later call. Any other error is
    re-raised unchanged.
    """
    global _tool_calling_disabled
    if not _tool_calling_disabled:
        messages: List = [
            SystemMessage(content=_fallback_system_prompt(context, history)),
            HumanMessage(content=question),
        ]
        try:
            return messages, agent_llm.invoke(messages)
        except Exception as e:
            if not AGENT_TOOLS or not _is_tools_unsupported_error(e):
                raise
            _tool_calling_disabled = True
            logger.warning(
                f"Ollama model '{config.OLLAMA_MODEL}' does not support tool calling, so the agent "
                "will answer from the documents only. For tool calling, set OLLAMA_MODEL to a "
                "tool-capable model such as llama3.1 or qwen3."
            )

    messages = [
        SystemMessage(content=_fallback_system_prompt(context, history, use_tools=False)),
        HumanMessage(content=question),
    ]
    return messages, llm.invoke(messages)


def run_agent_with_context(question: str, context: str, history: str) -> str:
    """Agent fallback: gives the LLM real tool-calling ability alongside RAG
    context, for questions the fast path didn't confidently resolve."""
    messages, ai_message = _first_agent_call(question, context, history)
    if not getattr(ai_message, "tool_calls", None):
        return ai_message.content

    _execute_tool_calls(messages, ai_message)
    final_message = agent_llm.invoke(messages)
    return final_message.content


def stream_agent_with_context(question: str, context: str, history: str) -> Iterator[str]:
    """Streaming counterpart to run_agent_with_context.

    The tool-vs-direct-answer decision itself is never streamed (there's nothing
    meaningful to show mid-decision) — only the final answer is. If no tool was
    needed, the decision call already produced the complete answer, so it's
    yielded as-is rather than wastefully regenerating it via a second call.
    """
    messages, ai_message = _first_agent_call(question, context, history)

    if not getattr(ai_message, "tool_calls", None):
        if ai_message.content:
            yield ai_message.content
        return

    _execute_tool_calls(messages, ai_message)
    for chunk in agent_llm.stream(messages):
        if chunk.content:
            yield chunk.content
