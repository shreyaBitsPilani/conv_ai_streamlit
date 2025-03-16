# retrieval.py
"""
Implements hybrid search using BM25 for sparse retrieval plus FAISS for dense retrieval.
Then merges the results to pick the most relevant chunks for a query.
"""

import json
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_metadata(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_bm25_index(chunks):
    """
    Builds a BM25 index using rank_bm25 library.
    """
    tokenized_corpus = [c[0].split() for c in chunks]  # naive tokenization
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

def hybrid_search(query, bm25, chunk_texts, faiss_index, metadata, dense_model, top_k=5):
    """
    1) BM25 retrieval
    2) Dense retrieval via FAISS
    3) Merge results (naive approach)
    """
    # Sparse retrieval
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    # Sort chunks by BM25 score desc
    bm25_ranked = sorted(
        list(zip(range(len(chunk_texts)), bm25_scores)),
        key=lambda x: x[1],
        reverse=True
    )
    top_bm25 = bm25_ranked[:top_k]
    
    # Dense retrieval
    query_emb = dense_model.encode([query], convert_to_numpy=True)
    # search in FAISS
    # For an inner-product index, we might want to L2 normalize 
    # the embeddings if we used IP. Adjust as needed.
    # faiss.normalize_L2(query_emb)  # if relevant
    distances, indices = faiss_index.search(query_emb, top_k)  # returns (scores, idx)
    
    # Convert results to list of (doc_id, score)
    dense_results = [(idx, float(dist)) for idx, dist in zip(indices[0], distances[0])]
    
    # Naive merge: just pick top_k from each, or unify them, re-sort by combined score
    # For demonstration, we'll place them in one list with a simple weighting
    # BM25 might be in range ~[0, X], FAISS in ~[0,1] if using IP.
    # One can tune weighting or do more advanced merges.
    
    combined = []
    # Put BM25 results with weighting factor
    for doc_id, score in top_bm25:
        combined.append((doc_id, score, "bm25"))
    # Put dense results
    for doc_id, score in dense_results:
        combined.append((doc_id, score * 100, "dense"))  # scale up for naive combination
    
    # Sort combined by score desc
    combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)
    # Deduplicate by doc_id, keep best
    seen = set()
    final_results = []
    for doc_id, score, source in combined_sorted:
        if doc_id not in seen:
            final_results.append((doc_id, score, source))
            seen.add(doc_id)
            if len(final_results) >= top_k:
                break
    
    # Map doc_ids to text + metadata
    retrieved = []
    for doc_id, score, source in final_results:
        retrieved.append({
            "doc_id": doc_id,
            "score": score,
            "source": source,
            "text": chunk_texts[doc_id],
            "metadata": metadata[doc_id]
        })
    return retrieved

if __name__ == "__main__":
    # Example usage
    # The chunk_texts and metadata come from embedding.py in the same order
    # so store them in a separate file or handle them similarly
    # For demonstration, re-construct them quickly below:
    
    json_path = os.path.join(BASE_DIR, "..", "data", "processed", "financial_data.json")
    index_path = os.path.join(BASE_DIR, "..", "embeddings", "financial_data.index")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Load data
    financial_records = load_metadata(json_path)
    
    # Reconstruct the same chunks as in embedding.py 
    from embedding import chunk_data
    chunks = chunk_data(financial_records)
    chunk_texts = [c[0] for c in chunks]
    metadata = [c[1] for c in chunks]
    
    # Build BM25 index
    bm25 = build_bm25_index(chunks)
    
    # Load FAISS index
    faiss_index = faiss.read_index(index_path)
    
    # Load dense model
    dense_model = SentenceTransformer(model_name)
    
    # Query
    query = "What is the total revenue for 2023?"
    results = hybrid_search(query, bm25, chunk_texts, faiss_index, metadata, dense_model)
    
    for r in results:
        print(r)
