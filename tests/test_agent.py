from unittest.mock import MagicMock, patch

import pytest

from rag import agent, config
from rag.tools import AVAILABLE_TOOLS


@pytest.fixture(autouse=True)
def _all_tools_enabled(monkeypatch):
    """Pin every tool on so these tests don't depend on the active profile, and start
    every test with tool calling allowed (one test may switch it off)."""
    monkeypatch.setattr(config, "ENABLED_TOOLS", ["flight", "weather", "currency"])
    monkeypatch.setattr(agent, "_tool_calling_disabled", False)


# ---------- Fast path (regex-resolved, no LLM call needed for detection) ----------


@patch("rag.agent.get_flight_status", return_value="Flight AI302 is on time.")
def test_fast_path_resolves_flight_question_with_number(mock_flight):
    result = agent.fast_path_tool_answer("Where is AI302?")
    assert result == "Flight AI302 is on time."


def test_fast_path_returns_none_when_flight_number_missing():
    assert agent.fast_path_tool_answer("Is my flight delayed?") is None


@patch("rag.agent.get_weather", return_value="Clear skies in Delhi, 24.0°C.")
def test_fast_path_resolves_weather_question(mock_weather):
    result = agent.fast_path_tool_answer("What's the weather in Delhi?")
    assert result == "Clear skies in Delhi, 24.0°C."


@patch("rag.agent.convert_currency", return_value="50,000.00 INR ≈ 600.50 USD (live exchange rate).")
def test_fast_path_resolves_currency_question(mock_convert):
    result = agent.fast_path_tool_answer("Convert 50000 INR to USD")
    assert "600.50 USD" in result


def test_fast_path_returns_none_for_document_questions():
    assert agent.fast_path_tool_answer("How many aircraft does Air India have?") is None


# ---------- Phrasing a fast-path tool result through the LLM ----------


@patch("rag.agent.llm")
def test_phrase_fast_path_answer_calls_llm_with_tool_result(mock_llm):
    mock_llm.invoke.return_value = MagicMock(content="Your flight AI302 is on time, all good!")
    result = agent.phrase_fast_path_answer("Where is AI302?", "Flight AI302 is on time.", "")
    assert result == "Your flight AI302 is on time, all good!"
    mock_llm.invoke.assert_called_once()


@patch("rag.agent.llm")
def test_stream_phrase_fast_path_answer_yields_chunks(mock_llm):
    mock_llm.stream.return_value = iter([MagicMock(content="Yes, "), MagicMock(content="on time!")])
    chunks = list(agent.stream_phrase_fast_path_answer("Where is AI302?", "on time", ""))
    assert chunks == ["Yes, ", "on time!"]


# ---------- Agent fallback (real LangChain tool-calling) ----------


@patch("rag.agent.agent_llm")
def test_run_agent_with_context_returns_direct_answer_when_no_tool_called(mock_agent_llm):
    fake_response = MagicMock()
    fake_response.tool_calls = []
    fake_response.content = "Air India has 138 aircraft."
    mock_agent_llm.invoke.return_value = fake_response

    result = agent.run_agent_with_context("How many aircraft?", "context about fleet", "")

    assert result == "Air India has 138 aircraft."
    mock_agent_llm.invoke.assert_called_once()


@patch("rag.agent.agent_llm")
@patch("rag.agent.TOOL_BY_NAME")
def test_run_agent_with_context_executes_tool_call_then_returns_final_answer(
    mock_tool_by_name, mock_agent_llm
):
    tool_call_message = MagicMock()
    tool_call_message.tool_calls = [
        {"name": "flight_status_tool", "args": {"flight_number": "AI302"}, "id": "call_1"}
    ]
    final_message = MagicMock()
    final_message.content = "Your flight AI302 is currently on time."
    mock_agent_llm.invoke.side_effect = [tool_call_message, final_message]

    fake_tool = MagicMock()
    fake_tool.invoke.return_value = "Flight AI302 is on time."
    mock_tool_by_name.get.return_value = fake_tool

    result = agent.run_agent_with_context("Where is AI302?", "", "")

    assert result == "Your flight AI302 is currently on time."
    assert mock_agent_llm.invoke.call_count == 2
    fake_tool.invoke.assert_called_once_with({"flight_number": "AI302"})


@patch("rag.agent.agent_llm")
def test_stream_agent_with_context_yields_content_directly_without_regenerating(mock_agent_llm):
    fake_response = MagicMock()
    fake_response.tool_calls = []
    fake_response.content = "Air India has 138 aircraft."
    mock_agent_llm.invoke.return_value = fake_response

    chunks = list(agent.stream_agent_with_context("How many aircraft?", "context", ""))

    assert chunks == ["Air India has 138 aircraft."]
    mock_agent_llm.stream.assert_not_called()  # avoids a wasted second generation


@patch("rag.agent.agent_llm")
@patch("rag.agent.TOOL_BY_NAME")
def test_stream_agent_with_context_executes_tool_then_streams_final_answer(
    mock_tool_by_name, mock_agent_llm
):
    tool_call_message = MagicMock()
    tool_call_message.tool_calls = [
        {"name": "weather_tool", "args": {"location": "Delhi"}, "id": "call_1"}
    ]
    mock_agent_llm.invoke.return_value = tool_call_message
    mock_agent_llm.stream.return_value = iter([MagicMock(content="It's "), MagicMock(content="sunny in Delhi.")])

    fake_tool = MagicMock()
    fake_tool.invoke.return_value = "Clear skies in Delhi, 24.0°C."
    mock_tool_by_name.get.return_value = fake_tool

    chunks = list(agent.stream_agent_with_context("What's the weather in Delhi?", "", ""))

    assert chunks == ["It's ", "sunny in Delhi."]
    fake_tool.invoke.assert_called_once_with({"location": "Delhi"})


# ---------- Profile-driven behaviour (generalization) ----------


@patch("rag.agent.get_flight_status")
def test_fast_path_never_uses_a_tool_the_profile_disables(mock_flight, monkeypatch):
    monkeypatch.setattr(config, "ENABLED_TOOLS", ["weather", "currency"])
    assert agent.fast_path_tool_answer("Where is AI302?") is None
    mock_flight.assert_not_called()


def test_tool_registry_covers_every_available_tool():
    # Guards against profile.json names and the agent's registry drifting apart
    assert set(agent.TOOL_REGISTRY) == set(AVAILABLE_TOOLS) == set(agent.TOOL_PURPOSES)


def test_build_agent_tools_returns_only_the_enabled_tools():
    assert agent.build_agent_tools(["weather"]) == [agent.weather_tool]
    assert agent.build_agent_tools([]) == []
    with pytest.raises(ValueError):
        agent.build_agent_tools(["teleport"])


@patch("rag.agent.agent_llm")
def test_system_prompt_uses_the_profile_identity(mock_agent_llm, monkeypatch):
    monkeypatch.setattr(config, "APP_NAME", "HR Helper")
    monkeypatch.setattr(config, "DOMAIN_DESCRIPTION", "Acme Corp's HR policies")
    mock_agent_llm.invoke.return_value = MagicMock(tool_calls=[], content="ok")

    agent.run_agent_with_context("How many leave days?", "some context", "")

    system_message = mock_agent_llm.invoke.call_args.args[0][0]
    assert "HR Helper" in system_message.content
    assert "Acme Corp's HR policies" in system_message.content
    assert "Air India" not in system_message.content
    assert "do not guess" in system_message.content


@patch("rag.agent.agent_llm")
def test_system_prompt_mentions_tools_only_when_some_are_enabled(mock_agent_llm, monkeypatch):
    mock_agent_llm.invoke.return_value = MagicMock(tool_calls=[], content="ok")

    agent.run_agent_with_context("Q?", "ctx", "")
    with_tools = mock_agent_llm.invoke.call_args.args[0][0].content
    assert "live flight status" in with_tools

    monkeypatch.setattr(config, "ENABLED_TOOLS", [])
    agent.run_agent_with_context("Q?", "ctx", "")
    without_tools = mock_agent_llm.invoke.call_args.args[0][0].content
    assert "tool" not in without_tools
    assert "ONLY the context" in without_tools


@patch("rag.agent.llm")
def test_phrasing_prompt_uses_the_profile_identity(mock_llm, monkeypatch):
    monkeypatch.setattr(config, "APP_NAME", "Travel Desk")
    mock_llm.invoke.return_value = MagicMock(content="It's sunny.")

    agent.phrase_fast_path_answer("Weather in Delhi?", "Clear skies, 24°C.", "")

    prompt = mock_llm.invoke.call_args.args[0]
    assert "Travel Desk" in prompt and "Clear skies, 24°C." in prompt


# ---------- Ollama models without tool support (e.g. llama3, phi3) ----------

TOOLS_UNSUPPORTED = Exception(
    "registry.ollama.ai/library/llama3:latest does not support tools (status code: 400)"
)


@patch("rag.agent.llm")
@patch("rag.agent.agent_llm")
def test_agent_answers_from_documents_when_the_model_rejects_tools(mock_agent_llm, mock_llm, monkeypatch):
    monkeypatch.setattr(agent, "AGENT_TOOLS", [agent.weather_tool])
    mock_agent_llm.invoke.side_effect = TOOLS_UNSUPPORTED
    mock_llm.invoke.return_value = MagicMock(tool_calls=[], content="From the documents: 12 aircraft.")

    answer = agent.run_agent_with_context("How many aircraft?", "ctx", "")

    assert answer == "From the documents: 12 aircraft."
    fallback_prompt = mock_llm.invoke.call_args.args[0][0].content
    assert "tool" not in fallback_prompt and "ONLY the context" in fallback_prompt
    assert agent._tool_calling_disabled is True

    # Remembered: the next question goes straight to the plain LLM.
    agent.run_agent_with_context("And their average age?", "ctx", "")
    assert mock_agent_llm.invoke.call_count == 1
    assert mock_llm.invoke.call_count == 2


@patch("rag.agent.llm")
@patch("rag.agent.agent_llm")
def test_streaming_agent_uses_the_same_fallback(mock_agent_llm, mock_llm, monkeypatch):
    monkeypatch.setattr(agent, "AGENT_TOOLS", [agent.weather_tool])
    mock_agent_llm.invoke.side_effect = TOOLS_UNSUPPORTED
    mock_llm.invoke.return_value = MagicMock(tool_calls=[], content="From the documents.")

    assert list(agent.stream_agent_with_context("Q?", "ctx", "")) == ["From the documents."]


@patch("rag.agent.llm")
@patch("rag.agent.agent_llm")
def test_other_llm_errors_are_not_swallowed(mock_agent_llm, mock_llm, monkeypatch):
    monkeypatch.setattr(agent, "AGENT_TOOLS", [agent.weather_tool])
    mock_agent_llm.invoke.side_effect = ConnectionError("Ollama is not running")

    with pytest.raises(ConnectionError):
        agent.run_agent_with_context("Q?", "ctx", "")
    mock_llm.invoke.assert_not_called()
    assert agent._tool_calling_disabled is False
