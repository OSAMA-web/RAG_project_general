"""Public entrypoint: question in, grounded answer out.

Phase 6: the fast-path tool check (flight/weather/currency) still skips RAG
retrieval entirely when it resolves cleanly — no point running hybrid search
for a question that's never going to use retrieved documents. Anything else
runs full retrieval AND gives the LLM real tool-calling ability as a fallback
(see rag/agent.py), so a live-data question the fast path missed can still be
routed correctly. Either way, Ollama always produces the final wording.
"""

from typing import Iterator, List, Optional, Tuple

from . import memory
from .agent import (
    fast_path_tool_answer,
    phrase_fast_path_answer,
    run_agent_with_context,
    stream_agent_with_context,
    stream_phrase_fast_path_answer,
)
from .retrieval import retrieve


def get_response(
    question: str, session_id: Optional[str] = None, knowledge_base: Optional[str] = None
) -> str:
    answer, _ = get_response_with_sources(question, session_id, knowledge_base)
    return answer


def get_response_with_sources(
    question: str,
    session_id: Optional[str] = None,
    knowledge_base: Optional[str] = None,
) -> Tuple[str, List[str]]:
    """Same as get_response, but also returns what grounded the answer — either
    retrieved document chunks, or a note on which live API was used.
    """
    history_text = memory.format_history(session_id) if session_id else ""

    fast_result = fast_path_tool_answer(question)
    if fast_result is not None:
        answer = phrase_fast_path_answer(question, fast_result, history_text)
        if session_id:
            memory.add_turn(session_id, question, answer)
        return answer, [f"Live data: {fast_result}"]

    docs = retrieve(question, knowledge_base=knowledge_base)
    context = "\n\n".join(doc.page_content for doc in docs)
    sources = [doc.page_content.strip() for doc in docs]

    answer = run_agent_with_context(question, context, history_text)

    if session_id:
        memory.add_turn(session_id, question, answer)

    return answer, sources


def stream_response_with_sources(
    question: str,
    session_id: Optional[str] = None,
    knowledge_base: Optional[str] = None,
) -> Iterator[dict]:
    """Streaming counterpart to get_response_with_sources — same routing logic,
    yielding {"type": "sources", ...} once and then {"type": "token", ...} repeatedly.
    """
    history_text = memory.format_history(session_id) if session_id else ""

    fast_result = fast_path_tool_answer(question)
    if fast_result is not None:
        yield {"type": "sources", "sources": [f"Live data: {fast_result}"]}
        full_answer = ""
        for chunk in stream_phrase_fast_path_answer(question, fast_result, history_text):
            full_answer += chunk
            yield {"type": "token", "text": chunk}
        if session_id:
            memory.add_turn(session_id, question, full_answer)
        return

    docs = retrieve(question, knowledge_base=knowledge_base)
    context = "\n\n".join(doc.page_content for doc in docs)
    sources = [doc.page_content.strip() for doc in docs]

    yield {"type": "sources", "sources": sources}

    full_answer = ""
    for chunk in stream_agent_with_context(question, context, history_text):
        full_answer += chunk
        yield {"type": "token", "text": chunk}

    if session_id:
        memory.add_turn(session_id, question, full_answer)
