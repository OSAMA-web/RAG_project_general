from rag import memory


def test_new_session_id_is_unique():
    a = memory.new_session_id()
    b = memory.new_session_id()
    assert a != b


def test_add_turn_and_get_history():
    session = memory.new_session_id()
    memory.add_turn(session, "Who founded Air India?", "JRD Tata.")
    history = memory.get_history(session)
    assert history == [("Who founded Air India?", "JRD Tata.")]


def test_get_history_keeps_the_full_unlimited_conversation():
    session = memory.new_session_id()
    total_turns = memory.PROMPT_WINDOW_TURNS + 10
    for i in range(total_turns):
        memory.add_turn(session, f"question {i}", f"answer {i}")

    full_history = memory.get_history(session)
    assert len(full_history) == total_turns
    assert full_history[0] == ("question 0", "answer 0")  # nothing was dropped


def test_get_recent_history_returns_only_the_prompt_window():
    session = memory.new_session_id()
    total_turns = memory.PROMPT_WINDOW_TURNS + 10
    for i in range(total_turns):
        memory.add_turn(session, f"question {i}", f"answer {i}")

    recent = memory.get_recent_history(session)
    assert len(recent) == memory.PROMPT_WINDOW_TURNS
    assert recent[-1] == (f"question {total_turns - 1}", f"answer {total_turns - 1}")


def test_clear_session_removes_history():
    session = memory.new_session_id()
    memory.add_turn(session, "q", "a")
    memory.clear_session(session)
    assert memory.get_history(session) == []


def test_format_history_empty_session_returns_empty_string():
    session = memory.new_session_id()
    assert memory.format_history(session) == ""


def test_format_history_formats_turns_as_text():
    session = memory.new_session_id()
    memory.add_turn(session, "Who founded Air India?", "JRD Tata.")
    formatted = memory.format_history(session)
    assert "User: Who founded Air India?" in formatted
    assert "Assistant: JRD Tata." in formatted


def test_format_history_excludes_turns_outside_the_prompt_window():
    session = memory.new_session_id()
    total_turns = memory.PROMPT_WINDOW_TURNS + 2
    for i in range(total_turns):
        memory.add_turn(session, f"question {i}", f"answer {i}")

    formatted = memory.format_history(session)
    assert "question 0" not in formatted  # outside the window, correctly excluded
    assert f"question {total_turns - 1}" in formatted  # most recent turn, included
