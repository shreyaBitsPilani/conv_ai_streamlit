import os
import streamlit as st
import torch
import numpy
# We disable parallel tokenization to prevent concurrency issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import nltk
nltk.download("punkt")
nltk.download('punkt_tab')
# Import your utility functions
from utils import (
    load_and_save_financial_statements,
    clean_financial_statements,
    split_text_semantic,
    build_bm25_index,
    load_bm25_index,
    build_faiss_index,
    load_faiss_index,
    embed_texts,
    retrieve_and_rerank,
    generate_answer,
    postprocess_answer,
    validate_query
)
torch.set_num_threads(1)
st.set_page_config(page_title="Financial RAG Demo", layout="centered")

st.title("Financial Q&A: Retrieval-Augmented Generation (RAG)")

st.write("""
**Instructions**:
1. Enter any valid **stock ticker** (e.g. `AAPL`, `MSFT`) in the sidebar.  
2. Press **Load Data** to download & index the last 2 years of statements.  
3. Ask financial questions in the text input.  
4. The system uses BM25 + FAISS for retrieval, then a small GPT model for generation.
""")

# --- Sidebar for Ticker Input ---
ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL")

if st.sidebar.button("Load Data"):
    with st.spinner(f"Loading data for {ticker}..."):
        statements = load_and_save_financial_statements(ticker)
        if not statements:
            st.error(f"Could not load statements for ticker '{ticker}'.")
        else:
            # Clean & chunk
            text_data = clean_financial_statements(statements)
            chunks = split_text_semantic(text_data, max_words=150, overlap_sentences=2, min_words=30)
            st.write(f"**Loaded** {len(chunks)} text chunks from {ticker}'s statements.")

            # BM25
            bm25, tokenized_chunks = load_bm25_index()
            if bm25 is None:
                bm25, tokenized_chunks = build_bm25_index(chunks)
                st.write("BM25 index built.")

            # Sentence-Transformers Embeddings
            from sentence_transformers import SentenceTransformer
            # We do single-threaded FAISS to reduce concurrency segfault
            import faiss
            faiss.omp_set_num_threads(1)

            @st.cache_resource
            def load_embedding_model():
                from sentence_transformers import SentenceTransformer
                return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

            dense_model = load_embedding_model()

            @st.cache_data
            def compute_dense_embeddings(chunks):
                return embed_texts(chunks, dense_model)

            dense_embeddings = compute_dense_embeddings(chunks)

            # FAISS
            faiss_index = load_faiss_index(dense_embeddings.shape[1])
            if faiss_index.ntotal == 0:
                faiss_index = build_faiss_index(dense_embeddings)
                st.write("FAISS index built.")

            st.session_state["ticker"] = ticker
            st.session_state["chunks"] = chunks
            st.session_state["bm25"] = bm25
            st.session_state["tokenized_chunks"] = tokenized_chunks
            st.session_state["dense_model"] = dense_model
            st.session_state["dense_embeddings"] = dense_embeddings
            st.session_state["faiss_index"] = faiss_index

            st.success("Data loaded & indexes ready. You can now ask questions below.")

# --- Load Generation Model (DistilGPT2 or GPT2-Medium) ---
@st.cache_resource
def load_generation_model():
    from transformers import pipeline
    # Use a bigger model if you want more consistent answers
    return pipeline("text-generation", model="distilgpt2")

generation_model = load_generation_model()

# --- Q&A Section ---
st.subheader("Ask a Financial Question")
user_query = st.text_input("e.g., 'What was Apple's total revenue last quarter?'")

if user_query:
    # Input-side guardrail
    if not validate_query(user_query):
        st.error("❌ This query doesn’t appear financial. Use words like 'revenue', 'profit', etc.")
    else:
        if "chunks" not in st.session_state:
            st.warning("⚠️ Please load data first from the sidebar.")
        else:
            with st.spinner("Retrieving relevant context..."):
                retrieved_chunks, confidence = retrieve_and_rerank(
                    user_query,
                    st.session_state["chunks"],
                    st.session_state["bm25"],
                    st.session_state["tokenized_chunks"],
                    st.session_state["dense_model"],
                    st.session_state["faiss_index"],
                    st.session_state["dense_embeddings"],
                    top_k=3
                )
                context = "\n".join(retrieved_chunks)

            st.write(f"**Confidence:** {round(confidence, 2)}")
            st.text_area("Retrieved Context", context, height=150)

            with st.spinner("Generating answer..."):
                raw_answer = generate_answer(user_query, context, generation_model)
                final_answer = postprocess_answer(raw_answer, user_query, context, st.session_state["dense_model"])

            st.write("### Answer")
            st.write(final_answer)

            # Feedback
            feedback = st.radio("Was this answer helpful?", ["Yes", "No"], index=0)
            if feedback:
                st.write("Thank you for your feedback!")
