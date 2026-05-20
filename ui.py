import hashlib
import os
import streamlit as st
import requests
from typing import Dict, Any


def format_api_error(exc: Exception) -> str:
    if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
        try:
            body = exc.response.json()
            return body.get("detail") or exc.response.text
        except Exception:
            return exc.response.text
    return str(exc)


# --------------------- Config ---------------------
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# --------------------- Sidebar ---------------------
with st.sidebar:
    st.header("📄 Document & Code Upload")
    st.caption("Upload PDFs, Markdown, or code files to expand your knowledge base")

    uploaded_file = st.file_uploader(
        "Choose a file to ingest",
        type=["pdf", "md", "txt", "py", "js", "ts", "json", "csv"],
        key="pdf_uploader",
    )

    if uploaded_file is not None:
        if st.button("🚀 Upload Source File", type="primary"):
            with st.spinner("Uploading and indexing source file..."):
                try:
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    }
                    response = requests.post(
                        f"{API_BASE_URL}/upload", files=files, timeout=180
                    )
                    response.raise_for_status()
                    st.success(f"✅ **{uploaded_file.name}** uploaded successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Upload failed: {e}")

    st.divider()
    st.subheader("🌐 Live Web Articles")
    web_url = st.text_input("Enter a URL to ingest an article", key="web_url")
    if web_url and st.button("📥 Ingest Web Source", key="ingest_web"):
        with st.spinner("Fetching and indexing web content..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/ingest/url", data={"url": web_url}, timeout=180
                )
                response.raise_for_status()
                st.success("✅ Web source ingested successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Web ingestion failed: {format_api_error(e)}")
    st.divider()
    st.subheader("🔊 YouTube / Video Transcript")
    youtube_url = st.text_input("Enter a YouTube video link", key="youtube_url")
    if youtube_url and st.button("📥 Ingest YouTube Transcript", key="ingest_youtube"):
        with st.spinner("Fetching video transcript..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/ingest/youtube",
                    data={"url": youtube_url},
                    timeout=180,
                )
                response.raise_for_status()
                st.success("✅ YouTube transcript ingested successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ YouTube ingestion failed: {format_api_error(e)}")

    st.caption(
        "If YouTube auto-fetch is blocked, paste the transcript or text manually below."
    )
    manual_source_name = st.text_input(
        "Manual source label", value="Manual transcript", key="manual_source_name"
    )
    manual_text = st.text_area(
        "Paste raw transcript or text to index", key="manual_text", height=180
    )
    if manual_text and st.button(
        "📥 Ingest Manual Transcript", key="ingest_manual_text"
    ):
        with st.spinner("Indexing manual transcript..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/ingest/text",
                    data={
                        "content": manual_text,
                        "source_name": manual_source_name,
                        "source_type": "youtube_transcript",
                    },
                    timeout=180,
                )
                response.raise_for_status()
                st.success("✅ Manual transcript ingested successfully!")
                st.rerun()
            except Exception as e:
                st.error(
                    f"❌ Manual transcript ingestion failed: {format_api_error(e)}"
                )

    st.divider()
    st.subheader("🗄️ Local SQLite Database")
    sqlite_file = st.file_uploader(
        "Upload a local SQLite .db file", type=["db", "sqlite"], key="sqlite_uploader"
    )
    sqlite_table = "user_history"

    if sqlite_file is not None:
        sqlite_bytes = sqlite_file.getvalue()
        sqlite_hash = hashlib.sha256(sqlite_bytes).hexdigest()

        if st.session_state.get("sqlite_file_hash") != sqlite_hash:
            st.session_state["sqlite_file_hash"] = sqlite_hash
            st.session_state["sqlite_detected_tables"] = []
            st.session_state["sqlite_table_selected"] = ""

            with st.spinner("Inspecting SQLite file for tables..."):
                try:
                    files = {
                        "file": (
                            sqlite_file.name,
                            sqlite_bytes,
                            "application/octet-stream",
                        )
                    }
                    response = requests.post(
                        f"{API_BASE_URL}/ingest/sqlite/tables", files=files, timeout=60
                    )
                    response.raise_for_status()
                    st.session_state["sqlite_detected_tables"] = response.json().get(
                        "tables", []
                    )
                except Exception as e:
                    st.warning(
                        f"⚠️ Could not detect table names: {format_api_error(e)}"
                    )

        detected_tables = st.session_state.get("sqlite_detected_tables", [])
        if detected_tables:
            sqlite_table = st.selectbox(
                "Select a table to ingest",
                options=detected_tables,
                index=0,
                key="sqlite_table_select",
            )
            st.caption(
                "If the table you want is missing, upload a different SQLite file or enter the table name manually below."
            )
            manual_table = st.text_input(
                "Or enter a different table name",
                value=sqlite_table,
                key="sqlite_table_manual",
            )
            if manual_table.strip():
                sqlite_table = manual_table.strip()
        else:
            sqlite_table = st.text_input(
                "Table name to ingest", value="user_history", key="sqlite_table"
            )
    else:
        sqlite_table = st.text_input(
            "Table name to ingest", value="user_history", key="sqlite_table"
        )

    if sqlite_file is not None and st.button(
        "📥 Ingest SQLite Table", key="ingest_sqlite"
    ):
        with st.spinner("Reading SQLite table and indexing..."):
            try:
                files = {
                    "file": (
                        sqlite_file.name,
                        sqlite_file.getvalue(),
                        "application/octet-stream",
                    )
                }
                data = {"table_name": sqlite_table}
                response = requests.post(
                    f"{API_BASE_URL}/ingest/sqlite", files=files, data=data, timeout=180
                )
                response.raise_for_status()
                st.success("✅ SQLite table ingested successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ SQLite ingestion failed: {format_api_error(e)}")

    st.divider()
    st.subheader("📚 Uploaded Documents")

    try:
        docs_response = requests.get(f"{API_BASE_URL}/documents", timeout=10)
        if docs_response.status_code == 200:
            documents = docs_response.json()

            if documents:
                for doc in documents:
                    with st.expander(f"📄 {doc['filename']}", expanded=False):
                        st.caption(
                            f"**Chunks:** {doc['chunk_count']} | **ID:** {doc['file_hash'][:8]}..."
                        )
                        st.caption(f"Uploaded: {doc.get('uploaded_at', 'N/A')}")

                        if st.button(
                            "🗑️ Delete Document",
                            key=f"del_{doc['file_hash']}",
                            type="secondary",
                        ):
                            if st.warning(
                                "Are you sure you want to delete this document? This action cannot be undone."
                            ):
                                try:
                                    with st.spinner("Deleting document..."):
                                        del_resp = requests.delete(
                                            f"{API_BASE_URL}/documents/{doc['file_hash']}"
                                        )
                                        if del_resp.status_code == 200:
                                            st.success(
                                                "✅ Document deleted successfully!"
                                            )
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
        use_reranker = st.checkbox(
            "Enable Re-ranking",
            value=True,
            help="Improves relevance using CrossEncoder",
        )
    with col_r2:
        use_hybrid = st.checkbox(
            "Enable Hybrid Search", value=True, help="Semantic + Keyword (BM25)"
        )

    # Number of retrieved chunks to request from the backend. This value
    # is sent as `top_k` in the query payload and controls final result size.
    top_k = st.slider(
        "Number of chunks to retrieve", min_value=3, max_value=15, value=6, step=1
    )

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
                st.metric(
                    "Embedding Hit Rate", f"{stats.get('embedding_hit_rate', 0)}%"
                )
            with col2:
                st.metric("Distinct Cached Queries", stats.get("query_entries", 0))
                st.metric("Cached Embeddings", stats.get("embedding_entries", 0))

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
    except Exception:
        st.caption("Cache monitoring unavailable")


# --------------------- Main Area ---------------------
st.title("📚 Your RAG Assistant")
st.caption(
    "Ask questions about PDFs, web pages, SQLite tables, and video transcripts • Powered by Groq + FAISS"
)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat Input
question = st.chat_input(
    "Ask anything about your sources: PDFs, web pages, SQLite tables, or videos..."
)

if question:
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.spinner("🔍 Retrieving relevant chunks... Thinking..."):
        try:
            payload = {
                "question": question,
                "top_k": top_k,  # Use the slider value
                "use_reranker": use_reranker,  # Send the checkbox value
                "use_hybrid": use_hybrid,  # Send hybrid search setting
                "history": [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.chat_history
                    if m.get("role") and m.get("content")
                ],
            }
            r = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=180)
            r.raise_for_status()
            data: Dict[str, Any] = r.json()

            answer = data.get("answer", "Sorry, I couldn't generate an answer.")
            sources = data.get("sources", [])

            # Add assistant message
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )

        except Exception as e:
            st.error(f"❌ Error querying the RAG system: {str(e)}")
            st.session_state.chat_history.append(
                {"role": "assistant", "content": f"Error: {str(e)}", "sources": []}
            )

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

                        color = (
                            "🟢"
                            if final_score >= 0.5
                            else "🟡" if final_score >= 0.3 else "🔴"
                        )

                        st.markdown(f"""
**{color} Source {i}** — **{src.get('file', 'Unknown')}** (Page {src.get('page', '?')})  
**Final Score:** `{final_score:.4f}`
                        """)

                        if rerank_score is not None and rerank_score > 0:
                            st.caption(
                                f"Rerank: `{rerank_score:.3f}` | Original: `{orig_score:.3f}`"
                            )
                        else:
                            st.caption(f"Original embedding score: `{orig_score:.3f}`")

                        st.markdown(f"> {src.get('snippet', '')}")
                        if i < len(message["sources"]):
                            st.divider()
