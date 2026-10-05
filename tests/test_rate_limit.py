from rag import rate_limit


def test_allows_requests_under_the_limit(monkeypatch):
    monkeypatch.setattr(rate_limit, "MAX_REQUESTS_PER_WINDOW", 3)
    rate_limit.reset("user-1")
    assert rate_limit.is_allowed("user-1") is True
    assert rate_limit.is_allowed("user-1") is True
    assert rate_limit.is_allowed("user-1") is True


def test_blocks_requests_over_the_limit(monkeypatch):
    monkeypatch.setattr(rate_limit, "MAX_REQUESTS_PER_WINDOW", 2)
    rate_limit.reset("user-2")
    assert rate_limit.is_allowed("user-2") is True
    assert rate_limit.is_allowed("user-2") is True
    assert rate_limit.is_allowed("user-2") is False  # third request in the window is blocked


def test_different_keys_have_independent_quotas(monkeypatch):
    monkeypatch.setattr(rate_limit, "MAX_REQUESTS_PER_WINDOW", 1)
    rate_limit.reset("user-a")
    rate_limit.reset("user-b")
    assert rate_limit.is_allowed("user-a") is True
    assert rate_limit.is_allowed("user-b") is True  # separate quota, unaffected by user-a


def test_reset_clears_a_keys_history(monkeypatch):
    monkeypatch.setattr(rate_limit, "MAX_REQUESTS_PER_WINDOW", 1)
    rate_limit.reset("user-c")
    assert rate_limit.is_allowed("user-c") is True
    assert rate_limit.is_allowed("user-c") is False
    rate_limit.reset("user-c")
    assert rate_limit.is_allowed("user-c") is True
