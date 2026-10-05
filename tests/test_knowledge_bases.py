from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from rag import ingestion


@patch("rag.ingestion.vector_store")
def test_list_knowledge_bases_returns_distinct_sorted_names(mock_vector_store):
    mock_vector_store.get.return_value = {
        "metadatas": [
            {"knowledge_base": "Air India"},
            {"knowledge_base": "HR Policy"},
            {"knowledge_base": "Air India"},
            {},  # missing key entirely -> falls back to the default
            None,  # some Chroma versions may return None for empty metadata
        ]
    }
    result = ingestion.list_knowledge_bases()
    assert result == sorted({"Air India", "HR Policy", ingestion.config.DEFAULT_KNOWLEDGE_BASE})


@patch("rag.ingestion.vector_store")
def test_list_knowledge_bases_empty_store_returns_empty_list(mock_vector_store):
    mock_vector_store.get.return_value = {"metadatas": []}
    assert ingestion.list_knowledge_bases() == []


@patch("rag.ingestion.vector_store")
@patch("rag.ingestion._loader_for")
def test_add_uploaded_document_tags_chunks_with_given_knowledge_base(mock_loader_for, mock_vector_store):
    fake_loader = MagicMock()
    fake_loader.load.return_value = [Document(page_content="Some HR policy text. " * 50)]
    mock_loader_for.return_value = fake_loader

    ingestion.add_uploaded_document("fake/path.pdf", "policy.pdf", knowledge_base="HR Policy")

    added_docs = mock_vector_store.add_documents.call_args.kwargs["documents"]
    assert len(added_docs) > 0
    assert all(doc.metadata["knowledge_base"] == "HR Policy" for doc in added_docs)
    assert all(doc.metadata["source"] == "policy.pdf" for doc in added_docs)
    assert all(doc.metadata["origin"] == "upload" for doc in added_docs)  # never touched by the folder sync


@patch("rag.ingestion.vector_store")
@patch("rag.ingestion._loader_for")
def test_add_uploaded_document_defaults_to_uploaded_files_kb(mock_loader_for, mock_vector_store):
    fake_loader = MagicMock()
    fake_loader.load.return_value = [Document(page_content="Some notes long enough to survive chunking. " * 3)]
    mock_loader_for.return_value = fake_loader

    ingestion.add_uploaded_document("fake/path.txt", "notes.txt")

    added_docs = mock_vector_store.add_documents.call_args.kwargs["documents"]
    assert len(added_docs) > 0
    assert all(
        doc.metadata["knowledge_base"] == ingestion.config.DEFAULT_UPLOAD_KNOWLEDGE_BASE
        for doc in added_docs
    )


@patch("rag.ingestion.vector_store")
@patch("rag.ingestion._loader_for")
def test_add_uploaded_document_with_no_text_adds_nothing_instead_of_crashing(mock_loader_for, mock_vector_store):
    # e.g. a scanned PDF: no extractable text -> 0 chunks. Chroma rejects an empty insert.
    fake_loader = MagicMock()
    fake_loader.load.return_value = [Document(page_content="  ")]
    mock_loader_for.return_value = fake_loader

    assert ingestion.add_uploaded_document("fake/scan.pdf", "scan.pdf") == 0
    mock_vector_store.add_documents.assert_not_called()
