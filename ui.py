import streamlit as st
import requests
from typing import Dict, Any

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Local RAG", layout="wide")

st.title("📚 Local RAG App")
st.caption("Upload PDFs → Ask questions → Get grounded answers with sources")

# ---------------------------
# Sidebar: Upload
# ---------------------------
st.sidebar.header("Upload PDF")

uploaded_file = st.sidebar.file_uploader("Choose a PDF", type=["pdf"])

if uploaded_file is not None:
    with st.sidebar:
        with st.spinner("Uploading and indexing..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                r = requests.post(f"{API_BASE}/upload", files=files, timeout=120)
                r.raise_for_status()
                st.success(f"Upload result: {r.json()}")
            except Exception as e:
                st.error(f"Upload failed: {e}")

# ---------------------------
# Main: Chat
# ---------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.text_input("Ask a question about your documents:")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                payload = {"question": question, "top_k": 5}
                r = requests.post(f"{API_BASE}/query", json=payload, timeout=120)
                r.raise_for_status()
                data: Dict[str, Any] = r.json()

                st.session_state.chat_history.append(
                    {
                        "question": question,
                        "answer": data.get("answer"),
                        "sources": data.get("sources", []),
                    }
                )
            except Exception as e:
                st.error(f"Query failed: {e}")

# ---------------------------
# Render Chat History
# ---------------------------
for entry in reversed(st.session_state.chat_history):
    st.markdown("---")
    st.markdown(f"### ❓ Question\n{entry['question']}")
    st.markdown(f"### 🤖 Answer\n{entry['answer']}")

    if entry["sources"]:
        with st.expander("📖 Sources"):
            for i, s in enumerate(entry["sources"], 1):
                st.markdown(
                    f"""
**[{i}] {s.get('file')} p.{s.get('page')}**  
Score: {s.get('score'):.3f}

> {s.get('snippet')}
"""
                )