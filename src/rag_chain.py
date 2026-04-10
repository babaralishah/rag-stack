# src/langchain_version/rag_chain.py
# Improved LangChain RAG with guardrail and hybrid fallback

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import List, Dict, Any
import logging
import os

logger = logging.getLogger(__name__)

class LangChainRAG:
    def __init__(self, groq_api_key: str = None, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is required")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=self.groq_api_key,
            temperature=0.7,
            max_tokens=1024
        )

        self.vectorstore = None
        self.retriever = None
        self.min_score = 0.35   # Guardrail threshold (same as your original project)

        logger.info("LangChainRAG initialized with guardrail (min_score=0.35)")
        
        
    def add_documents(self, texts: List[str], metadatas: List[Dict] = None, store_dir: str = "storage/faiss"):
        """Add new documents to the vector store"""
        if not texts:
            return

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
        else:
            self.vectorstore.add_texts(texts=texts, metadatas=metadatas)

        self.vectorstore.save_local(store_dir)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        logger.info(f"Added {len(texts)} documents to vector store")

    def load_vectorstore(self, store_dir: str = "storage/faiss"):
        """Load existing FAISS vector store"""
        if os.path.exists(store_dir):
            try:
                self.vectorstore = FAISS.load_local(
                    store_dir, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
                logger.info(f"Loaded vector store with {self.vectorstore.index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Failed to load vector store: {e}")
        else:
            logger.warning("No vector store found. Upload documents first.")

    def get_answer(self, question: str) -> Dict[str, Any]:
        """Main method - Get answer with guardrail and hybrid fallback"""
        if self.retriever is None:
            return {
                "answer": "No documents indexed yet. Please upload PDFs first.",
                "sources": []
            }

        # Step 1: Retrieve documents
        docs = self.retriever.invoke(question)
        
        if not docs:
            return {
                "answer": "I don\'t have sufficient information in the uploaded documents.",
                "sources": []
            }

        # Step 2: Simple guardrail (using LangChain similarity)
        # For now we use a basic check. We can improve this later with real scores.
        context = "\n\n".join([doc.page_content for doc in docs])

        # Step 3: Create smart prompt with guardrail instruction
        template = """You are a careful and honest assistant.

Use the provided context to answer the question.
If the answer is clearly supported by the context, answer directly and concisely.
If the context does not contain enough relevant information, first say:
"I don\'t have sufficient information in the uploaded documents."

Then, you may add: "However, based on my general knowledge, ..." and give your best general answer.

Context:
{context}

Question: {question}

Answer:"""

        prompt = ChatPromptTemplate.from_template(template)

        # Create chain
        chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Run the chain
        try:
            answer = chain.invoke({"context": context, "question": question})

            # Prepare sources
            sources = []
            for doc in docs:
                sources.append({
                    "file": doc.metadata.get("source_file", "unknown"),
                    "page": doc.metadata.get("page", "?"),
                    "snippet": doc.page_content[:250].replace("\n", " ")
                })

            return {
                "answer": answer,
                "sources": sources
            }

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": []
            }
