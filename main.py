"""CLI entrypoint. Syncs the active profile's docs folder into the index, then answers one sample question."""

from rag import config
from rag.ingestion import build_index_if_needed
from rag.pipeline import get_response

if __name__ == "__main__":
    print(f"Profile: {config.PROFILE_NAME} ({config.APP_NAME}) — docs folder: {config.DOCS_FOLDER}")
    chunk_count = build_index_if_needed()
    print(f"Vector store ready with {chunk_count} chunks.\n")

    if chunk_count == 0:
        print("No documents indexed yet — add files to the docs folder above, or upload them in the web UI.")
    else:
        sample_question = (config.SUGGESTED_QUESTIONS or ["What are the main topics covered in these documents?"])[0]
        print(f"Q: {sample_question}")
        print(f"A: {get_response(sample_question)}")
