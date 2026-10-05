from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import config, ingestion


def test_chunk_settings_match_config():
    assert config.CHUNK_SIZE == 1000
    assert config.CHUNK_OVERLAP == 200


def test_splitter_breaks_long_document_into_multiple_chunks():
    long_text = "Air India operates flights across India and internationally. " * 100
    doc = Document(page_content=long_text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents([doc])

    assert len(chunks) > 1
    assert all(len(c.page_content) <= config.CHUNK_SIZE for c in chunks)


def test_splitter_keeps_short_document_as_one_chunk():
    short_text = "Air India was founded by JRD Tata."
    doc = Document(page_content=short_text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents([doc])

    assert len(chunks) == 1


# ---------- Phase 3: parent/child chunking ----------


@pytest.fixture(autouse=True)
def _reset_parent_store():
    """_parent_store is a module-level dict — isolate tests from each other."""
    ingestion._parent_store.clear()
    yield
    ingestion._parent_store.clear()


@patch("rag.ingestion._save_parent_store")
def test_chunk_with_parent_child_links_each_child_to_a_saved_parent(mock_save):
    long_text = "Air India information sentence. " * 200  # long enough for multiple parent chunks
    doc = Document(page_content=long_text, metadata={"knowledge_base": "Air India"})

    children = ingestion._chunk_with_parent_child([doc])

    assert len(children) > 0
    for child in children:
        parent_id = child.metadata.get("parent_id")
        assert parent_id in ingestion._parent_store
        assert len(ingestion._parent_store[parent_id]) >= len(child.page_content)
    mock_save.assert_called_once()


@patch("rag.ingestion._save_parent_store")
def test_chunk_with_parent_child_drops_near_empty_chunks(mock_save):
    doc = Document(page_content="Hi", metadata={})  # shorter than MIN_CHUNK_LENGTH
    children = ingestion._chunk_with_parent_child([doc])
    assert children == []


def test_get_parent_text_returns_none_for_unknown_id():
    assert ingestion.get_parent_text("does-not-exist-xyz") is None


@patch("rag.ingestion.vector_store")
def test_get_all_child_documents_filters_by_knowledge_base(mock_vector_store):
    mock_vector_store.get.return_value = {
        "documents": ["doc about Air India", "doc about HR policy"],
        "metadatas": [{"knowledge_base": "Air India"}, {"knowledge_base": "HR Policy"}],
    }
    result = ingestion.get_all_child_documents(knowledge_base="HR Policy")
    assert len(result) == 1
    assert result[0].page_content == "doc about HR policy"


@patch("rag.ingestion.vector_store")
def test_get_all_child_documents_returns_everything_when_no_filter(mock_vector_store):
    mock_vector_store.get.return_value = {
        "documents": ["doc A", "doc B"],
        "metadatas": [{"knowledge_base": "Air India"}, {"knowledge_base": "HR Policy"}],
    }
    result = ingestion.get_all_child_documents()
    assert len(result) == 2
