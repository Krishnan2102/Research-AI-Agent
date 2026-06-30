import streamlit as st
from main import get_answer


def render_answer(answer: dict):
    st.markdown(f"**Topic:** {answer['topic']}")
    st.markdown("**Key Points:**")
    st.markdown(answer["key_points"])
    st.markdown("**Summary:**")
    st.markdown(answer["summary"])
    st.markdown("**Sources:**")
    st.markdown(answer["source"])


st.set_page_config(page_title="Research Agent", page_icon="🔎")
st.title("🔎 Research Agent")
st.caption("Ask a question and the agent will research it for you.")

if "history" not in st.session_state:
    st.session_state.history = []  # list of (role, content) where content is str or dict

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
                answer = get_answer(query)
                render_answer(answer)
                st.session_state.history.append(("assistant", answer))
            except Exception as e:
                error_msg = f"Something went wrong while processing that query: {e}"
                st.error(error_msg)
                st.session_state.history.append(("assistant", error_msg))