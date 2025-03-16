# app.py
import os
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np
from rank_bm25 import BM25Okapi

from embedding import chunk_data, load_index
from retrieval import build_bm25_index, hybrid_search
from re_ranking import ReRanker
from slm_generation import SLMResponseGenerator  # <--- NEW

# Determine the base directory relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_financial_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def load_dense_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

@st.cache_resource
def load_faiss_index(index_path):
    return load_index(index_path)

def main():
    st.title("Financial QA Demo")

    # Build paths relative to the current file
    json_path = os.path.join(BASE_DIR, "..", "data", "processed", "financial_data.json")
    index_path = os.path.join(BASE_DIR, "..", "embeddings", "financial_data.index")
    
    # Load data/models
    financial_data = load_financial_data(json_path)
    chunks = chunk_data(financial_data)
    chunk_texts = [c[0] for c in chunks]
    metadata = [c[1] for c in chunks]
    
    bm25 = build_bm25_index(chunks)
    dense_model = load_dense_model()
    faiss_index = load_faiss_index(index_path)
    reranker = ReRanker()
    
    # Instantiate our small language model for final answer generation
    slm = SLMResponseGenerator(model_name="t5-small")

    user_query = st.text_input("Enter your financial question here", "")
    if st.button("Search"):
        if user_query.strip():
            # 1. Hybrid retrieval
            retrieved = hybrid_search(
                user_query, 
                bm25, 
                chunk_texts, 
                faiss_index, 
                metadata, 
                dense_model
            )
            # 2. Re-rank
            reranked = reranker.rerank(user_query, retrieved)
            
            # 3. Now let our SLM generate a final answer using top retrieved docs
            final_answer = slm.generate_response(user_query, reranked)
            
            # Display final answer
            st.subheader("Final Answer")
            st.write(final_answer)
            
            # (Optional) Show top snippet
            if reranked:
                top_snippet = reranked[0]
                st.write("---")
                st.write("**Top Retrieved Snippet**")
                st.write(top_snippet["text"])
                st.write(f"Re-rank Score: {top_snippet['re_rank_score']:.2f}")
                
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
