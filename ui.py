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
                
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Upload failed: {e}")

    st.divider()
    
    # ==================== SETTINGS ====================
    st.markdown("### ⚙️ Settings")
    
    # Re-ranking Toggle
    use_reranker = st.checkbox(
        "Enable Re-ranking", 
        value=True,
        help="Uses CrossEncoder reranker + score fusion for better relevance. Recommended."
    )
    
    # Optional: Top-K slider
    top_k = st.slider(
        "Number of chunks to retrieve", 
        min_value=1, 
        max_value=15, 
        value=TOP_K, 
        step=1,
        help="Higher value = more context but slower"
    )

    st.caption("Re-ranking improves answer quality but adds slight latency.")
    
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
            payload = {
            "question": question, 
            "top_k": top_k,           # Use the slider value
            "use_reranker": use_reranker   # Send the checkbox value
        }
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

            # Sources Section
            if message.get("sources"):
                with st.expander("📚 View Sources & Relevance", expanded=False):
                    for i, src in enumerate(message["sources"], 1):
                        final_score = src.get("final_score", src.get("score", 0))
                        rerank_score = src.get("rerank_score")
                        orig_score = src.get("score", 0)

                        color = "🟢" if final_score >= 0.5 else "🟡" if final_score >= 0.3 else "🔴"

                        st.markdown(f"""
**{color} Source {i}** — **{src.get('file', 'Unknown')}** (Page {src.get('page', '?')})  
**Final Score:** `{final_score:.4f}`
                        """)

                        if rerank_score is not None and rerank_score > 0:
                            st.caption(f"Rerank: `{rerank_score:.3f}` | Original: `{orig_score:.3f}`")
                        else:
                            st.caption(f"Original embedding score: `{orig_score:.3f}`")

                        st.markdown(f"> {src.get('snippet', '')}")
                        if i < len(message["sources"]):
                            st.divider()