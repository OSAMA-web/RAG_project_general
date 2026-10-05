from unittest.mock import patch

from rag.pipeline import get_response_with_sources, stream_response_with_sources


@patch("rag.pipeline.phrase_fast_path_answer", return_value="Your flight AI302 is on time.")
@patch("rag.pipeline.fast_path_tool_answer", return_value="Flight AI302 is on time.")
def test_get_response_uses_fast_path_and_skips_retrieval_entirely(mock_fast_path, mock_phrase):
    with patch("rag.pipeline.retrieve") as mock_retrieve:
        answer, sources = get_response_with_sources("Where is AI302?")

        assert answer == "Your flight AI302 is on time."
        assert "Live data" in sources[0]
        mock_retrieve.assert_not_called()  # fast path must never run RAG retrieval


@patch("rag.pipeline.fast_path_tool_answer", return_value=None)
@patch("rag.pipeline.run_agent_with_context", return_value="Air India has 138 aircraft.")
@patch("rag.pipeline.retrieve", return_value=[])
def test_get_response_falls_back_to_agent_with_retrieval_when_fast_path_misses(
    mock_retrieve, mock_run_agent, mock_fast_path
):
    answer, sources = get_response_with_sources("How many aircraft does Air India have?")

    assert answer == "Air India has 138 aircraft."
    mock_retrieve.assert_called_once()
    mock_run_agent.assert_called_once()


@patch(
    "rag.pipeline.stream_phrase_fast_path_answer",
    return_value=iter(["Your ", "flight ", "is on time."]),
)
@patch("rag.pipeline.fast_path_tool_answer", return_value="Flight AI302 is on time.")
def test_stream_response_uses_fast_path_and_skips_retrieval_entirely(mock_fast_path, mock_stream_phrase):
    with patch("rag.pipeline.retrieve") as mock_retrieve:
        events = list(stream_response_with_sources("Where is AI302?"))
        types = [e["type"] for e in events]

        assert types == ["sources", "token", "token", "token"]
        assert "Live data" in events[0]["sources"][0]
        mock_retrieve.assert_not_called()


@patch("rag.pipeline.fast_path_tool_answer", return_value=None)
@patch("rag.pipeline.stream_agent_with_context", return_value=iter(["Some ", "RAG ", "answer."]))
@patch("rag.pipeline.retrieve", return_value=[])
def test_stream_response_falls_back_to_agent_streaming_when_fast_path_misses(
    mock_retrieve, mock_stream_agent, mock_fast_path
):
    events = list(stream_response_with_sources("How many aircraft does Air India have?"))
    types = [e["type"] for e in events]

    assert types == ["sources", "token", "token", "token"]
    mock_retrieve.assert_called_once()
