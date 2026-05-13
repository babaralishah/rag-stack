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

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], key="pdf_uploader")

    if uploaded_file is not None:
        if st.button("🚀 Upload Document", type="primary"):
            with st.spinner("Uploading and indexing document..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=180)
                    response.raise_for_status()
                    st.success(f"✅ **{uploaded_file.name}** uploaded successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Upload failed: {e}")

    # ==================== UPLOADED DOCUMENTS ====================
    st.divider()
    st.subheader("📚 Uploaded Documents")

    try:
        docs_response = requests.get(f"{API_BASE_URL}/documents", timeout=10)
        if docs_response.status_code == 200:
            documents = docs_response.json()

            if documents:
                for doc in documents:
                    with st.expander(f"📄 {doc['filename']}", expanded=False):
                        st.caption(f"**Chunks:** {doc['chunk_count']} | **ID:** {doc['file_hash'][:8]}...")
                        st.caption(f"Uploaded: {doc.get('uploaded_at', 'N/A')}")

                        if st.button("🗑️ Delete Document", key=f"del_{doc['file_hash']}", type="secondary"):
                            if st.warning("Are you sure you want to delete this document? This action cannot be undone."):
                                try:
                                    with st.spinner("Deleting document..."):
                                        del_resp = requests.delete(f"{API_BASE_URL}/documents/{doc['file_hash']}")
                                        if del_resp.status_code == 200:
                                            st.success("✅ Document deleted successfully!")
                                            st.rerun()
                                        else:
                                            st.error("Failed to delete")
                                except Exception as e:
                                    st.error(f"Error: {e}")
            else:
                st.info("No documents uploaded yet. Upload some PDFs to get started!")
        else:
            st.warning("Could not load documents.")
    except Exception as e:
        st.error(f"Failed to load documents: {e}")
            
    # ==================== SETTINGS ====================
    st.divider()
    st.subheader("⚙️ Settings")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        use_reranker = st.checkbox("Enable Re-ranking", value=True, 
                                  help="Improves relevance using CrossEncoder")
    with col_r2:
        use_hybrid = st.checkbox("Enable Hybrid Search", value=True,
                                help="Semantic + Keyword (BM25)")

    top_k = st.slider("Number of chunks to retrieve", 
                      min_value=3, max_value=15, value=6, step=1)

    # ==================== CACHE STATUS ====================
    st.divider()
    st.subheader("⚡ Cache Status")

    try:
        cache_response = requests.get(f"{API_BASE_URL}/cache/stats", timeout=8)
        if cache_response.status_code == 200:
            stats = cache_response.json()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Query Hit Rate", f"{stats.get('query_hit_rate', 0)}%")
                st.metric("Embedding Hit Rate", f"{stats.get('embedding_hit_rate', 0)}%")
            with col2:
                st.metric("Cached Queries", stats.get('query_entries', 0))
                st.metric("Cached Embeddings", stats.get('embedding_entries', 0))
            
            st.caption(f"Last cleared: {stats.get('last_cleared', 'N/A')}")
            
            if st.button("🧹 Clear All Caches"):
                try:
                    clear_resp = requests.post(f"{API_BASE_URL}/cache/clear")
                    if clear_resp.status_code == 200:
                        st.success("Caches cleared successfully!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to clear cache: {e}")
        else:
            st.info("Cache stats not available yet.")
    except Exception as e:
        st.caption("Cache monitoring unavailable")

    
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