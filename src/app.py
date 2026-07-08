import os
import tempfile
import streamlit as st
from main import get_answer
from rag import ingest_pdf
import asyncio


def render_answer(answer: dict):
    st.markdown(f"**Topic:** {answer['topic']}")
    st.markdown("**Key Points:**")
    st.markdown(answer["key_points"])
    st.markdown("**Summary:**")
    st.markdown(answer["summary"])
    st.markdown("**Sources:**")
    st.markdown(answer["source"])
    source_type = answer.get("source_type")
    if source_type:
        label = "📚 Local knowledge (RAG)" if source_type == "rag" else "🌐 Web search"
        st.caption(label)


st.set_page_config(page_title="Research Agent", page_icon="🔎")
st.title("🔎 Research Agent")
st.caption("Ask a question and the agent will research it for you.")

# --- Sidebar: PDF ingestion into the local RAG knowledge base ---
with st.sidebar:
    st.header("📄 Add to knowledge base")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Ingest PDF"):
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                try:
                    # ingest_pdf expects a filepath on disk; the uploader
                    # gives us an in-memory file, so save it to a temp
                    # file first.
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf"
                    ) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    result = ingest_pdf(tmp_path)
                    os.remove(tmp_path)

                    if result["status"] == "success":
                        st.success(
                            f"Ingested '{uploaded_file.name}' "
                            f"({result['num_chunks']} chunks)."
                        )
                    else:
                        st.error(f"Ingestion failed: {result.get('error')}")
                except Exception as e:
                    st.error(f"Something went wrong while ingesting the PDF: {e}")

    st.caption(
        "Uploaded PDFs are added to the local knowledge base. "
        "Future questions will check this knowledge base first "
        "before falling back to web search."
    )

# --- Main chat area ---
if "history" not in st.session_state:
    st.session_state.history = []

# Render past messages
for role, content in st.session_state.history:
    with st.chat_message(role):
        if role == "assistant" and isinstance(content, dict):
            render_answer(content)
        else:
            st.markdown(content)

# Chat input
query = st.chat_input("Ask a research question...")
if query:
    st.session_state.history.append(("user", query))
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Researching..."):
            try:
                answer = asyncio.run(get_answer(query))
                render_answer(answer)
                st.session_state.history.append(("assistant", answer))
            except Exception as e:
                import traceback
                traceback.print_exc()  # full details go to the terminal
                error_msg = f"Something went wrong while processing that query: {e}"
                st.error(error_msg)
                st.session_state.history.append(("assistant", error_msg))