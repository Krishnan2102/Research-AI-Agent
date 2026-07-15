import os
import tempfile
import streamlit as st
from main import get_answer
from rag_mcp_client import call_rag_tool
import asyncio
from role_definition import ROLES, get_all_domains, is_unrestricted


def render_answer(answer: dict):
    st.markdown(f"**Topic:** {answer['topic']}")
    st.markdown("**Key Points:**")
    st.markdown(answer["key_points"])
    st.markdown("**Summary:**")
    st.markdown(answer["summary"])
    st.markdown("**Sources:**")
    st.markdown(answer["source"])
    source_type = answer.get("source_type")
    if source_type == "rag":
        st.caption("📚 Local knowledge (RAG)")
    elif source_type == "rejected":
        st.caption("🚫 Outside your role's domain")
    elif source_type:
        st.caption("🌐 Web search")


st.set_page_config(page_title="Research Agent", page_icon="🔎")
st.title("🔎 Research Agent")
st.caption("Ask a question and the agent will research it for you.")

selected_role = st.sidebar.selectbox(
    "Role (testing only)",
    options=list(ROLES.keys()),
    format_func=lambda r: ROLES[r]["label"],
)
st.session_state["user_role"] = selected_role


# --- Sidebar: PDF ingestion into the local RAG knowledge base ---
# Ingestion is admin-only: only admin should be able to add new documents
# to the shared knowledge base. Other roles just query it.
with st.sidebar:
    if is_unrestricted(st.session_state["user_role"]):
        st.header("📄 Add to knowledge base")
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

        available_domains = get_all_domains()
        selected_domain = st.selectbox(
            "Document domain",
            options=available_domains,
            help="Tags every chunk from this PDF so it's only retrievable "
                 "by roles allowed to see this domain.",
        )

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

                        result = asyncio.run(
                            call_rag_tool(
                                "ingest_pdf",
                                filepath=tmp_path,
                                domain=selected_domain,
                            )
                        )
                        os.remove(tmp_path)

                        if result["status"] == "success":
                            st.success(
                                f"Ingested '{uploaded_file.name}' into domain "
                                f"'{selected_domain}' ({result['num_chunks']} chunks)."
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
    else:
        st.caption("📄 Only admins can add documents to the knowledge base.")

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
                answer = asyncio.run(get_answer(query, role=st.session_state["user_role"]))
                render_answer(answer)
                st.session_state.history.append(("assistant", answer))
            except Exception as e:
                import traceback
                traceback.print_exc()  # full details go to the terminal
                error_msg = f"Something went wrong while processing that query: {e}"
                st.error(error_msg)
                st.session_state.history.append(("assistant", error_msg))