import os
from src.config import TOP_K
import streamlit as st
import requests
from typing import Dict, Any

# --------------------- Config ---------------------
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# --------------------- Sidebar ---------------------
with st.sidebar:
    st.header("📄 Document Upload")
    st.caption("Upload PDFs to build your knowledge base")

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        with st.spinner("Uploading and indexing document..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                response = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=180)
                response.raise_for_status()
                
                result = response.json()
                st.success(f"✅ Successfully uploaded: **{uploaded_file.name}**")
                if "chunks_created" in result:
                    st.info(f"Created {result.get('chunks_created', 0)} chunks")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Upload failed: {e}")

    # st.divider()
    # st.markdown("### ⚙️ Settings")
    # top_k = st.slider("Number of chunks to retrieve (Top-K)", 
    #                   min_value=3, max_value=15, value=TOP_K, step=1)

# --------------------- Main Area ---------------------
st.title("📚 RAG Assistant")
st.caption("Ask questions about your uploaded PDFs • Powered by Groq + FAISS")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat Input
question = st.chat_input("Ask anything about your documents...")

if question:
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.spinner("🔍 Retrieving relevant chunks... Thinking..."):
        try:
            payload = {"question": question, "top_k": TOP_K}
            r = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=180)
            r.raise_for_status()
            data: Dict[str, Any] = r.json()

            answer = data.get("answer", "Sorry, I couldn't generate an answer.")
            sources = data.get("sources", [])

            # Add assistant message
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

        except Exception as e:
            st.error(f"❌ Error querying the RAG system: {str(e)}")
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"Error: {str(e)}",
                "sources": []
            })

# --------------------- Display Chat History ---------------------
for message in st.session_state.chat_history:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    
    else:  # assistant
        with st.chat_message("assistant"):
            st.markdown(message["content"])

            # Sources Section (Beautiful & Collapsible)
            if message.get("sources"):
                with st.expander("📚 View Sources & Relevance", expanded=False):
                    for i, src in enumerate(message["sources"], 1):
                        score = src.get('score', 0)
                        score_color = "🟢" if score > 0.45 else "🟡" if score > 0.35 else "🔴"
                        
                        st.markdown(f"""
**{score_color} Source {i}** — **{src.get('file', 'Unknown')}** (Page {src.get('page', '?')})  
**Relevance:** `{score:.3f}`  
                        """)
                        
                        st.markdown(f"> {src.get('snippet', '')}")
                        st.divider()