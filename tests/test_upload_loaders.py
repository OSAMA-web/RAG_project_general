import pytest

from rag import ingestion


def _touch(tmp_path, name):
    """LangChain's PDF/DOCX loaders verify the file exists when they're constructed,
    so each test needs a real (empty) file rather than a made-up path."""
    path = tmp_path / name
    path.write_bytes(b"")
    return str(path)


def test_loader_for_pdf_returns_pypdf_loader(tmp_path):
    loader = ingestion._loader_for(_touch(tmp_path, "something.pdf"))
    assert loader.__class__.__name__ == "PyPDFLoader"


def test_loader_for_docx_returns_docx2txt_loader(tmp_path):
    loader = ingestion._loader_for(_touch(tmp_path, "something.docx"))
    assert loader.__class__.__name__ == "Docx2txtLoader"


def test_loader_for_txt_returns_text_loader():
    loader = ingestion._loader_for("something.txt")
    assert loader.__class__.__name__ == "TextLoader"


def test_loader_for_markdown_returns_text_loader():
    loader = ingestion._loader_for("notes.md")
    assert loader.__class__.__name__ == "TextLoader"


def test_loader_for_is_case_insensitive(tmp_path):
    loader = ingestion._loader_for(_touch(tmp_path, "SOMETHING.PDF"))
    assert loader.__class__.__name__ == "PyPDFLoader"


def test_loader_for_unsupported_extension_raises():
    with pytest.raises(ValueError):
        ingestion._loader_for("something.xyz")


def test_supported_upload_extensions_matches_loader_dispatch():
    # Guards against the two lists silently drifting apart
    assert ingestion.SUPPORTED_UPLOAD_EXTENSIONS == {".pdf", ".docx", ".txt", ".md"}
