import streamlit as st

from rag import config
from rag.ingestion import build_index_if_needed
from rag.pipeline import get_response

st.set_page_config(page_title=config.APP_NAME, layout="wide")

st.title(f"{config.APP_ICON} {config.APP_NAME}")
st.markdown("Ask a question — answers are grounded in the loaded documents.")

with st.spinner("Preparing knowledge base (first run only, may take a minute)..."):
    build_index_if_needed()

question = st.text_input("Enter your question:", placeholder=config.INPUT_PLACEHOLDER)

if st.button("Ask"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            try:
                answer = get_response(question)
                st.markdown("### 📄 Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"An error occurred: {e}")
