import os
import streamlit as st
import requests

st.set_page_config(page_title="RAGQuery", layout="centered")

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.title("RAGQuery")
st.write("Upload a document or paste text, then ask questions about it.")

# --- API Key ---
api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")

# --- Input Tabs ---
tab_upload, tab_paste = st.tabs(["Upload File", "Paste Text"])

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["txt", "pdf", "md", "docx"],
        help="Supported formats: .txt, .pdf, .md, .docx"
    )
    if st.button("Load Document"):
        if not api_key.strip():
            st.warning("Please enter your OpenAI API key.")
        elif not uploaded_file:
            st.warning("Please upload a document.")
        else:
            with st.spinner("Processing document..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/load",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                        data={"api_key": api_key}
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.session_id = data["session_id"]
                        st.session_state.chat_history = []
                        st.success(f"Loaded '{data['filename']}' — {data['chunks']} chunks indexed.")
                    else:
                        try:
                            detail = response.json().get('detail', 'Failed to load document.')
                        except Exception:
                            detail = response.text or 'Failed to load document.'
                        st.error(f"Error: {detail}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to the backend. Make sure the server is running.")

with tab_paste:
    pasted_text = st.text_area(
        "Paste your text here",
        height=300,
        placeholder="Paste any UTF-8 text document here..."
    )
    if st.button("Load Text"):
        if not api_key.strip():
            st.warning("Please enter your OpenAI API key.")
        elif not pasted_text.strip():
            st.warning("Please paste some text.")
        else:
            with st.spinner("Processing text..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/load-text",
                        json={"text": pasted_text, "api_key": api_key}
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.session_id = data["session_id"]
                        st.session_state.chat_history = []
                        st.success(f"Text loaded — {data['chunks']} chunks indexed.")
                    else:
                        try:
                            detail = response.json().get('detail', 'Failed to load text.')
                        except Exception:
                            detail = response.text or 'Failed to load text.'
                        st.error(f"Error: {detail}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to the backend. Make sure the server is running.")

# --- Chat ---
if "session_id" in st.session_state:
    st.divider()
    question = st.text_input("Enter your question")

    if st.button("Ask"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/ask",
                        json={"session_id": st.session_state.session_id, "question": question}
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.chat_history.append({
                            "question": question,
                            "answer": data["answer"],
                            "evidence": data.get("evidence", []),
                            "scores": data.get("scores", {})
                        })
                    else:
                        try:
                            detail = response.json().get('detail', 'Something went wrong.')
                        except Exception:
                            detail = response.text or 'Something went wrong.'
                        st.error(f"Error: {detail}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to the backend. Make sure the server is running.")

    # --- Chat History ---
    if st.session_state.get("chat_history"):
        st.divider()
        st.subheader("Chat History")
        for entry in reversed(st.session_state.chat_history):
            st.markdown(f"**You:** {entry['question']}")
            st.markdown(f"**Answer:** {entry['answer']}")
            if entry["evidence"]:
                with st.expander("Evidence"):
                    for chunk in entry["evidence"]:
                        st.write("- " + chunk)
            scores = entry.get("scores", {})
            if scores and "error" not in scores:
                col1, col2 = st.columns(2)
                col1.metric("Faithfulness", scores.get("faithfulness", "N/A"))
                col2.metric("Answer Relevancy", scores.get("answer_relevancy", "N/A"))
            st.divider()
