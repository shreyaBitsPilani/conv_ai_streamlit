# re_ranking.py
"""
Re-ranks retrieved candidates using a cross-encoder or similar model 
to refine the ordering based on semantic matching of (query, candidate_text).
"""

from sentence_transformers import CrossEncoder

class ReRanker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        A common cross-encoder for re-ranking is the MS-Marco cross-encoder model.
        """
        self.cross_encoder = CrossEncoder(model_name)
    
    def rerank(self, query, retrieved_docs):
        """
        retrieved_docs is a list of dicts like:
          { "text": <str>, "metadata": {...}, "score": <float>, ... }
        
        We compute cross-encoder score for each doc, then re-sort.
        """
        pairs = [(query, doc["text"]) for doc in retrieved_docs]
        scores = self.cross_encoder.predict(pairs)
        
        # Attach cross-encoder scores
        for i, doc in enumerate(retrieved_docs):
            doc["re_rank_score"] = float(scores[i])
        
        # Sort by cross-encoder score desc
        reranked = sorted(retrieved_docs, key=lambda d: d["re_rank_score"], reverse=True)
        return reranked

if __name__ == "__main__":
    # Example usage
    # Suppose we got 'retrieved' from the hybrid search
    # We'll do a quick example
    sample_retrieved = [
        {'doc_id': 257, 'score': 75.33212900161743, 'source': 'dense', 'text': 'Year: 2023, Parameter: Total Revenue, Value: 383285000000.0', 'metadata': {'year': 2023, 'finance_parameter': 'Total Revenue', 'value': 383285000000.0}}
        ,{'doc_id': 90, 'score': 72.27014899253845, 'source': 'dense', 'text': 'Year: 2024, Parameter: Total Revenue, Value: 391035000000.0', 'metadata': {'year': 2024, 'finance_parameter': 'Total Revenue', 'value': 391035000000.0}}
        ,{'doc_id': 256, 'score': 71.39800190925598, 'source': 'dense', 'text': 'Year: 2023, Parameter: Cost Of Revenue, Value: 214137000000.0', 'metadata': {'year': 2023, 'finance_parameter': 'Cost Of Revenue', 'value': 214137000000.0}}
        ,{'doc_id': 289, 'score': 70.35419940948486, 'source': 'dense', 'text': 'Year: 2024, Parameter: Cost Of Revenue, Value: 210352000000.0', 'metadata': {'year': 2024, 'finance_parameter': 'Cost Of Revenue', 'value': 210352000000.0}}
        ,{'doc_id': 258, 'score': 69.80007886886597, 'source': 'dense', 'text': 'Year: 2023, Parameter: Operating Revenue, Value: 383285000000.0', 'metadata': {'year': 2023, 'finance_parameter': 'Operating Revenue', 'value': 383285000000.0}}
    ]
    
    reranker = ReRanker()
    query = "What is the total revenue for 2023?"
    reranked_results = reranker.rerank(query, sample_retrieved)
    for r in reranked_results:
        print(r)
