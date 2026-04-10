import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="LangChain RAG", page_icon="📚", layout="wide")
st.title("📚 LangChain RAG App")
st.markdown("### Ask questions about your uploaded documents")

# API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")
API_BASE_URL = st.sidebar.text_input("API Base URL", value="http://127.0.0.1:8000")

st.sidebar.header("Settings")
st.sidebar.info(f"Connected to API: **{API_BASE_URL}**")

if "messages" not in st.session_state:
    st.session_state.messages = []

# File uploader
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None and st.button("Upload and Index Document"):
    with st.spinner("Uploading and indexing..."):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            response = requests.post(f"{API_BASE_URL}/upload", files=files)
            if response.status_code == 200:
                st.success(response.json().get("message", "Document indexed successfully!"))
            else:
                st.error(f"Upload failed: {response.text}")
        except Exception as e:
            st.error(f"Connection error: {e}")

# Chat interface
st.subheader("Ask a question")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = "Sorry, I couldn't get a response."
            try:
                response = requests.post(
                    f"{API_BASE_URL}/query",
                    json={"question": prompt}
                )
                if response.status_code == 200:
                    result = response.json()
                    answer = result.get("answer", "No answer returned.")
                    sources = result.get("sources", [])
                    
                    st.markdown(answer)
                    if sources:
                        st.markdown("**Sources:**")
                        for i, src in enumerate(sources, 1):
                            st.caption(f"{i}. {src.get('file', 'unknown')} (page {src.get('page', '?')})")
                else:
                    st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to API: {e}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
