"""Retrieval pipeline (Phase 3): hybrid search, multi-query expansion, parent-document
context, sentence-level compression, and score-thresholded reranking.

Pipeline for a single call to retrieve():
  1. expand_queries()   -- LLM generates a few rephrasings of the question
  2. For each phrasing   -- run BOTH dense (embedding) search and BM25 (keyword)
                             search, restricted to the chosen knowledge base if any
  3. Dedupe everything found across all variants and both methods
  4. Rerank the merged pool with a cross-encoder, scored against the ORIGINAL
     question (not the rephrasings) — drop anything below a minimum relevance score
  5. For the final top-k child chunks, look up their parent chunk (larger context)
     and compress it down to the most relevant sentences before returning

Trade-offs made deliberately for this project's scale:
  - The BM25 keyword index is rebuilt from the current document set on every call,
    rather than cached and incrementally updated. Fine for hundreds/thousands of
    chunks; a real high-traffic system would persist and update it instead.
  - Context compression re-uses the same small cross-encoder for sentence scoring
    rather than an extra LLM call, since multi-query already adds several retrieval
    passes and a second LLM round-trip per source would add real latency.
"""

import re
from typing import List, Optional

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from . import config
from .generation import llm
from .ingestion import get_all_child_documents, get_parent_text, vector_store

_reranker = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(config.RERANK_MODEL)
    return _reranker


def expand_queries(question: str) -> List[str]:
    """Multi-query retrieval: asks the LLM for a few alternate phrasings of the
    question. Different wordings often surface different relevant chunks that a
    single query embedding would miss. Always includes the original question, and
    falls back to just that if the LLM call fails for any reason.
    """
    prompt = (
        f"Generate {config.MULTI_QUERY_VARIANTS} different ways to ask the following "
        "question, to help search a document database. Return ONLY the questions, "
        "one per line, no numbering, no extra text.\n\n"
        f"Question: {question}"
    )
    variants: List[str] = []
    try:
        response = llm.invoke(prompt)
        for line in response.content.splitlines():
            cleaned = line.strip("-•0123456789. \t")
            if cleaned:
                variants.append(cleaned)
        variants = variants[: config.MULTI_QUERY_VARIANTS]
    except Exception:
        variants = []  # keep retrieval working even if the LLM call fails

    unique_variants = [v for v in variants if v.lower() != question.lower()]
    return [question] + unique_variants


def _dense_search(query: str, knowledge_base: Optional[str]) -> List[Document]:
    search_filter = {"knowledge_base": knowledge_base} if knowledge_base else None
    return vector_store.similarity_search(query, k=config.RETRIEVE_K, filter=search_filter)


def _dedupe(documents: List[Document]) -> List[Document]:
    seen = set()
    unique: List[Document] = []
    for doc in documents:
        key = doc.page_content.strip()
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique


def _compress_to_top_sentences(text: str, question: str, reranker: CrossEncoder) -> str:
    """Context compression: keeps only the most relevant sentences of a (parent)
    chunk relative to the question, so the final prompt isn't diluted with
    tangential surrounding text. Sentences are re-ordered back to their original
    sequence after selection, so the result still reads naturally.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) <= config.COMPRESSED_SENTENCES_PER_SOURCE:
        return text

    pairs = [(question, s) for s in sentences]
    scores = reranker.predict(pairs)

    ranked_indices = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
    selected_indices = sorted(ranked_indices[: config.COMPRESSED_SENTENCES_PER_SOURCE])
    return " ".join(sentences[i] for i in selected_indices)


def retrieve(
    question: str, k: int = config.RERANK_K, knowledge_base: Optional[str] = None
) -> List[Document]:
    """Retrieves the top-k most relevant chunks for a question (see module docstring
    for the full pipeline). Returns Documents whose page_content is already the
    compressed parent-context text, ready to use directly for both the LLM prompt
    and the "sources" shown to the user.
    """
    queries = expand_queries(question)

    bm25_corpus = get_all_child_documents(knowledge_base=knowledge_base)
    bm25_retriever = None
    if bm25_corpus:
        bm25_retriever = BM25Retriever.from_documents(bm25_corpus)
        bm25_retriever.k = config.BM25_TOP_K

    candidates: List[Document] = []
    for q in queries:
        candidates.extend(_dense_search(q, knowledge_base))
        if bm25_retriever:
            candidates.extend(bm25_retriever.invoke(q))

    candidates = _dedupe(candidates)
    if not candidates:
        return []

    reranker = _get_reranker()
    pairs = [(question, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
    ranked = [(score, doc) for score, doc in ranked if score >= config.MIN_RERANK_SCORE]
    top = ranked[:k]

    results: List[Document] = []
    for _score, child_doc in top:
        parent_id = child_doc.metadata.get("parent_id")
        parent_text = get_parent_text(parent_id) if parent_id else None
        full_text = parent_text or child_doc.page_content

        compressed_text = _compress_to_top_sentences(full_text, question, reranker)
        results.append(Document(page_content=compressed_text, metadata=child_doc.metadata))

    return results
