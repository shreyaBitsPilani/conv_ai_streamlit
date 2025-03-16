# embedding.py
"""
Chunks the financial data from the JSON, embeds using a SentenceTransformer,
and stores the vectors in a FAISS index.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def chunk_data(financial_records, chunk_size=1):
    """
    Example chunking: each record is effectively a chunk. 
    (If your real data are textual paragraphs, adjust logic accordingly.)
    Returns a list of tuples: (text, metadata).
    """
    chunks = []
    for record in financial_records:
        # Example text: "Year 2023, Parameter: Revenue, Value: 123456"
        # Or tailor the text representation to your needs
        text = (f"Year: {record['year']}, "
                f"Parameter: {record['finance_parameter']}, "
                f"Value: {record['value']}")
        meta = {
            "year": record["year"],
            "finance_parameter": record["finance_parameter"],
            "value": record["value"]
        }
        chunks.append((text, meta))
    return chunks

def build_faiss_index(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Embeds each chunk with SentenceTransformer, builds a FAISS index,
    and returns (index, embeddings, metadata).
    """
    model = SentenceTransformer(model_name)
    
    texts = [c[0] for c in chunks]
    metadata = [c[1] for c in chunks]
    
    # Embed
    embeddings = model.encode(texts, convert_to_numpy=True)
    dim = embeddings.shape[1]  # e.g. 384 for MiniLM-L6-v2
    
    # Build FAISS index
    # Here we use a simple IndexFlatIP (inner product). For better retrieval 
    # you might consider normalizing embeddings or using IndexFlatL2, etc.
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    return index, embeddings, metadata

def save_index(index, index_path):
    """
    Serializes the FAISS index to disk.
    """
    faiss.write_index(index, index_path)

def load_index(index_path):
    """
    Loads a FAISS index from disk.
    """
    return faiss.read_index(index_path)

if __name__ == "__main__":
    # Example usage
    json_path = os.path.join(BASE_DIR, "..", "data", "processed", "financial_data.json")
    index_path = os.path.join(BASE_DIR, "..", "embeddings", "financial_data.index")
    
    with open(json_path, "r", encoding="utf-8") as f:
        financial_records = json.load(f)
    
    chunks = chunk_data(financial_records)
    index, embeddings, metadata = build_faiss_index(chunks)
    save_index(index, index_path)
    print(f"FAISS index built and saved to {index_path}")
