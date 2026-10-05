"""Answer generation via a locally-served Ollama model."""

import os

from langchain_ollama import ChatOllama

from . import config

# Lets the same code run standalone (localhost) or in Docker Compose (service name "ollama")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

llm = ChatOllama(model=config.OLLAMA_MODEL, temperature=0, base_url=OLLAMA_BASE_URL)

PROMPT_TEMPLATE = """You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not contained in the context, say you don't have that information — do not guess.
Use the conversation history only to understand what the user is referring to (e.g. "it", "that route", "those numbers") — the context below is still your only source of facts.

Conversation history:
{history}

Context:
{context}

Question:
{question}
"""


def generate(question: str, context: str, history: str = "") -> str:
    prompt = PROMPT_TEMPLATE.format(history=history or "(none yet)", context=context, question=question)
    response = llm.invoke(prompt)
    return response.content


def stream_generate(question: str, context: str, history: str = ""):
    """Same prompt as generate(), but yields text chunks as they're produced
    instead of waiting for the full answer."""
    prompt = PROMPT_TEMPLATE.format(history=history or "(none yet)", context=context, question=question)
    for chunk in llm.stream(prompt):
        if chunk.content:
            yield chunk.content
