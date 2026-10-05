"""Live-data tools. None are on by default — each profile opts in via the
"enabled_tools" list in its profile.json (see rag/config.py)."""

# The names a profile may list under "enabled_tools". rag/agent.py maps each
# one to its LangChain tool; a test guards the two against drifting apart.
AVAILABLE_TOOLS = ("flight", "weather", "currency")
