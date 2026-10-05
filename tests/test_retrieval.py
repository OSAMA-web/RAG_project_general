from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from rag.retrieval import _compress_to_top_sentences, expand_queries, retrieve


# ---------- expand_queries (multi-query retrieval) ----------


@patch("rag.retrieval.llm")
def test_expand_queries_includes_original_and_parses_variants(mock_llm):
    mock_llm.invoke.return_value = MagicMock(
        content="How big is the Air India fleet?\nWhat is the size of Air India's fleet?"
    )
    variants = expand_queries("How many aircraft does Air India have?")
    assert variants[0] == "How many aircraft does Air India have?"
    assert "How big is the Air India fleet?" in variants
    assert len(variants) == 3


@patch("rag.retrieval.llm")
def test_expand_queries_falls_back_to_original_on_llm_failure(mock_llm):
    mock_llm.invoke.side_effect = RuntimeError("Ollama not running")
    variants = expand_queries("How many aircraft does Air India have?")
    assert variants == ["How many aircraft does Air India have?"]


# ---------- _compress_to_top_sentences (context compression) ----------


def test_compress_returns_short_text_unchanged_without_scoring():
    reranker = MagicMock()
    text = "One sentence. Two sentences."
    result = _compress_to_top_sentences(text, "irrelevant question", reranker)
    assert result == text
    reranker.predict.assert_not_called()


def test_compress_keeps_top_scoring_sentences_in_original_order():
    sentences = [f"Sentence {i}." for i in range(6)]
    text = " ".join(sentences)
    reranker = MagicMock()
    # Sentences 5, 1, 3, 0 score highest (in that order) — expect them back in
    # their ORIGINAL order (0, 1, 3, 5) after compression, not score order.
    reranker.predict.return_value = [0.1, 0.9, 0.05, 0.8, 0.0, 0.95]

    result = _compress_to_top_sentences(text, "question", reranker)

    assert result == "Sentence 0. Sentence 1. Sentence 3. Sentence 5."


# ---------- retrieve() end-to-end (hybrid search + rerank + parent lookup) ----------


@patch("rag.retrieval.get_parent_text", return_value=None)
@patch("rag.retrieval._get_reranker")
@patch("rag.retrieval.get_all_child_documents", return_value=[])
@patch("rag.retrieval.vector_store")
@patch("rag.retrieval.expand_queries", return_value=["Fleet size?"])
def test_retrieve_dense_only_when_no_bm25_corpus(
    mock_expand, mock_vector_store, mock_get_all_docs, mock_get_reranker, mock_get_parent
):
    doc = Document(page_content="Air India has 138 aircraft.", metadata={})
    mock_vector_store.similarity_search.return_value = [doc]

    fake_reranker = MagicMock()
    fake_reranker.predict.return_value = [5.0]
    mock_get_reranker.return_value = fake_reranker

    results = retrieve("Fleet size?", k=3)

    assert len(results) == 1
    assert "138 aircraft" in results[0].page_content


@patch("rag.retrieval.get_parent_text", return_value=None)
@patch("rag.retrieval._get_reranker")
@patch("rag.retrieval.BM25Retriever")
@patch("rag.retrieval.get_all_child_documents")
@patch("rag.retrieval.vector_store")
@patch("rag.retrieval.expand_queries", return_value=["Fleet size?"])
def test_retrieve_merges_dense_and_bm25_results(
    mock_expand,
    mock_vector_store,
    mock_get_all_docs,
    mock_bm25_cls,
    mock_get_reranker,
    mock_get_parent,
):
    dense_doc = Document(page_content="Dense result about fleet size.", metadata={})
    bm25_doc = Document(page_content="Keyword match about fleet size.", metadata={})

    mock_vector_store.similarity_search.return_value = [dense_doc]
    mock_get_all_docs.return_value = [dense_doc, bm25_doc]  # non-empty -> BM25 index gets built

    fake_bm25_instance = MagicMock()
    fake_bm25_instance.invoke.return_value = [bm25_doc]
    mock_bm25_cls.from_documents.return_value = fake_bm25_instance

    fake_reranker = MagicMock()
    fake_reranker.predict.return_value = [3.0, 4.0]
    mock_get_reranker.return_value = fake_reranker

    results = retrieve("Fleet size?", k=5)

    result_texts = {doc.page_content for doc in results}
    assert "Dense result about fleet size." in result_texts
    assert "Keyword match about fleet size." in result_texts


@patch("rag.retrieval.get_parent_text", return_value=None)
@patch("rag.retrieval._get_reranker")
@patch("rag.retrieval.get_all_child_documents", return_value=[])
@patch("rag.retrieval.vector_store")
@patch("rag.retrieval.expand_queries", return_value=["Fleet size?"])
def test_retrieve_drops_candidates_below_min_rerank_score(
    mock_expand, mock_vector_store, mock_get_all_docs, mock_get_reranker, mock_get_parent
):
    good_doc = Document(page_content="Relevant text about fleet size.", metadata={})
    bad_doc = Document(page_content="Irrelevant unrelated text.", metadata={})
    mock_vector_store.similarity_search.return_value = [good_doc, bad_doc]

    fake_reranker = MagicMock()
    fake_reranker.predict.return_value = [5.0, -3.0]  # bad_doc is below MIN_RERANK_SCORE (0.0)
    mock_get_reranker.return_value = fake_reranker

    results = retrieve("Fleet size?", k=5)

    assert len(results) == 1
    assert "Relevant text" in results[0].page_content


@patch("rag.retrieval.get_parent_text")
@patch("rag.retrieval._get_reranker")
@patch("rag.retrieval.get_all_child_documents", return_value=[])
@patch("rag.retrieval.vector_store")
@patch("rag.retrieval.expand_queries", return_value=["Fleet size?"])
def test_retrieve_expands_child_chunk_to_its_parent_text(
    mock_expand, mock_vector_store, mock_get_all_docs, mock_get_reranker, mock_get_parent
):
    child_doc = Document(page_content="short child chunk", metadata={"parent_id": "parent-1"})
    mock_vector_store.similarity_search.return_value = [child_doc]
    mock_get_parent.return_value = "This is the much longer parent chunk text with full context."

    fake_reranker = MagicMock()
    fake_reranker.predict.return_value = [5.0]
    mock_get_reranker.return_value = fake_reranker

    results = retrieve("Fleet size?", k=3)

    assert len(results) == 1
    assert "longer parent chunk" in results[0].page_content
    mock_get_parent.assert_called_once_with("parent-1")


@patch("rag.retrieval._get_reranker")
@patch("rag.retrieval.get_all_child_documents", return_value=[])
@patch("rag.retrieval.vector_store")
@patch("rag.retrieval.expand_queries", return_value=["Fleet size?"])
def test_retrieve_returns_empty_list_when_no_candidates(
    mock_expand, mock_vector_store, mock_get_all_docs, mock_get_reranker
):
    mock_vector_store.similarity_search.return_value = []
    results = retrieve("Fleet size?", k=3)
    assert results == []
    mock_get_reranker.assert_not_called()


@patch("rag.retrieval._get_reranker")
@patch("rag.retrieval.get_all_child_documents", return_value=[])
@patch("rag.retrieval.vector_store")
@patch("rag.retrieval.expand_queries", return_value=["Fleet size?"])
def test_retrieve_passes_knowledge_base_filter_to_dense_search_and_bm25_corpus(
    mock_expand, mock_vector_store, mock_get_all_docs, mock_get_reranker
):
    mock_vector_store.similarity_search.return_value = []

    retrieve("Fleet size?", knowledge_base="HR Policy")

    _, kwargs = mock_vector_store.similarity_search.call_args
    assert kwargs["filter"] == {"knowledge_base": "HR Policy"}
    mock_get_all_docs.assert_called_once_with(knowledge_base="HR Policy")
