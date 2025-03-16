import os
import pickle
import re
import numpy as np
import yfinance as yf
import pandas as pd
import nltk
from rank_bm25 import BM25Okapi
import faiss
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from transformers import GPT2Tokenizer
import streamlit as st

nltk.download("punkt", quiet=True)

DATA_FOLDER = "data"
MODELS_FOLDER = "models"
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

########################################
# 1) Download & Load Financial Statements
########################################

def download_financial_statements(ticker: str) -> dict:
    """
    Download the last 2 columns of Income, Balance, Cash Flow from Yahoo Finance.
    """
    try:
        stock = yf.Ticker(ticker)
        income = stock.financials.fillna("")
        balance = stock.balance_sheet.fillna("")
        cash_flow = stock.cashflow.fillna("")

        # Keep only the last 2 columns
        if income.shape[1] > 2:
            income = income.iloc[:, :2]
        if balance.shape[1] > 2:
            balance = balance.iloc[:, :2]
        if cash_flow.shape[1] > 2:
            cash_flow = cash_flow.iloc[:, :2]

        return {
            "income_statement": income,
            "balance_sheet": balance,
            "cash_flow": cash_flow
        }
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return {}

def load_and_save_financial_statements(ticker: str) -> dict:
    """
    If CSVs exist, load them; else download from yfinance and save to disk.
    """
    files = {
        "income_statement": f"{ticker}_income_statement.csv",
        "balance_sheet": f"{ticker}_balance_sheet.csv",
        "cash_flow": f"{ticker}_cash_flow.csv"
    }
    # Check if all exist
    if all(os.path.exists(os.path.join(DATA_FOLDER, f)) for f in files.values()):
        statements = {}
        for k, fn in files.items():
            statements[k] = pd.read_csv(os.path.join(DATA_FOLDER, fn), index_col=0)
        return statements
    else:
        data = download_financial_statements(ticker)
        if not data:
            return {}
        for k, df in data.items():
            df.to_csv(os.path.join(DATA_FOLDER, f"{ticker}_{k}.csv"))
        return data

########################################
# 2) Clean & Combine
########################################

def format_value(value):
    """Try converting to float, formatting with commas; else return ''."""
    try:
        val = float(value)
        return f"{val:,.2f}"
    except:
        return ""

def clean_financial_statements(statements: dict) -> str:
    """
    Combine statements into a single text. 
    """
    parts = []
    for stmt_name, df in statements.items():
        header = f"--- {stmt_name.replace('_',' ').title()} ---"
        lines = [header]
        for col in df.columns:
            try:
                col_str = pd.to_datetime(col).strftime("%Y-%m-%d")
            except:
                col_str = str(col)
            lines.append(f"Date: {col_str}")
            for idx in df.index:
                val = format_value(df.loc[idx, col])
                if val:
                    lines.append(f"{idx}: {val}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)

########################################
# 3) Chunking
########################################

def get_sentences(text: str) -> list:
    """
    Attempt to load the 'punkt_tab' tokenizer.
    If not available, download and load the standard 'punkt' tokenizer.
    """
    nltk.download("punkt", quiet=True)
    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    return tokenizer.tokenize(text)

def split_text_semantic(text: str, max_words=150, overlap_sentences=2, min_words=30) -> list:
    # Use our custom get_sentences function instead of nltk.sent_tokenize
    try:
        nltk.data.load("tokenizers/punkt/english.pickle")
    except LookupError:
        nltk.download("punkt", quiet=True)
    sentences = nltk.sent_tokenize(text)
    chunks = []
    cur_chunk = []
    cur_word_count = 0

    for sentence in sentences:
        words = sentence.split()
        cur_chunk.append(sentence)
        cur_word_count += len(words)

        if cur_word_count >= max_words:
            chunk_text = " ".join(cur_chunk)
            if len(chunk_text.split()) >= min_words:
                chunks.append(chunk_text)
            cur_chunk = cur_chunk[-overlap_sentences:]
            cur_word_count = sum(len(s.split()) for s in cur_chunk)

    if cur_chunk:
        chunk_text = " ".join(cur_chunk)
        if len(chunk_text.split()) >= min_words:
            chunks.append(chunk_text)

    with open(os.path.join(DATA_FOLDER, "text_chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)
    return chunks


########################################
# 4) BM25
########################################

def build_bm25_index(chunks: list):
    tokenized = [nltk.word_tokenize(ch.lower()) for ch in chunks]
    bm25 = BM25Okapi(tokenized)
    with open(os.path.join(MODELS_FOLDER, "bm25.pkl"), "wb") as f:
        pickle.dump((bm25, tokenized), f)
    return bm25, tokenized

def load_bm25_index():
    path = os.path.join(MODELS_FOLDER, "bm25.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            bm25, tokenized_chunks = pickle.load(f)
        return bm25, tokenized_chunks
    return None, None

########################################
# 5) FAISS
########################################

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    # single-threaded to reduce concurrency issues
    faiss.omp_set_num_threads(1)
    index.add(embeddings.astype("float32"))
    faiss.write_index(index, os.path.join(MODELS_FOLDER, "faiss.index"))
    return index

def load_faiss_index(emb_dim: int) -> faiss.Index:
    path = os.path.join(MODELS_FOLDER, "faiss.index")
    if os.path.exists(path):
        idx = faiss.read_index(path)
        if idx.d == emb_dim:
            faiss.omp_set_num_threads(1)
            return idx
        else:
            print("FAISS dimension mismatch! Rebuilding.")
            os.remove(path)
            return faiss.IndexFlatL2(emb_dim)
    else:
        return faiss.IndexFlatL2(emb_dim)

########################################
# 6) Embeddings
########################################

def embed_texts(chunks: list, model: SentenceTransformer) -> np.ndarray:
    return model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)

########################################
# 7) Hybrid Retrieval + Re-Ranking
########################################

def improved_hybrid_retrieve(
    query: str,
    chunks: list,
    bm25,
    tokenized_chunks,
    dense_model: SentenceTransformer,
    faiss_index,
    dense_embeddings: np.ndarray,
    top_k=3,
    dense_weight=0.8,
    bm25_weight=0.2
) -> (list, float):
    # Dense
    query_emb = dense_model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = faiss_index.search(query_emb, top_k * 2)

    max_dist = max(D[0]) if len(D[0]) > 0 else 1e-6
    dense_scores = {}
    for dist, idx in zip(D[0], I[0]):
        sim = 1.0 - (dist / max_dist)
        dense_scores[idx] = max(sim, 0.0)

    # BM25
    import nltk
    query_tokens = nltk.word_tokenize(query.lower())
    bm25_raw = bm25.get_scores(query_tokens)
    max_bm25 = max(bm25_raw) if bm25_raw.any() else 1

    combined_scores = {}
    for idx, d_score in dense_scores.items():
        s_score = (bm25_raw[idx]/max_bm25) if idx < len(bm25_raw) else 0
        combined_scores[idx] = dense_weight*d_score + bm25_weight*s_score

    sorted_candidates = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    top_indices = [i for i, _ in sorted_candidates[:top_k]]
    confidence = sorted_candidates[0][1] if sorted_candidates else 0.0
    retrieved = [chunks[i] for i in top_indices]
    return retrieved, confidence

def retrieve_and_rerank(
    query: str,
    chunks: list,
    bm25,
    tokenized_chunks,
    dense_model: SentenceTransformer,
    faiss_index,
    dense_embeddings: np.ndarray,
    top_k=3
) -> (list, float):
    candidates, hybrid_conf = improved_hybrid_retrieve(
        query, chunks, bm25, tokenized_chunks, dense_model, faiss_index,
        dense_embeddings, top_k=top_k*2
    )
    if not candidates:
        return [], 0.0

    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    # pairs = [(query, c) for c in candidates]
    # ce_scores = cross_encoder.predict(pairs)

    # if len(ce_scores) == 0:
    #     return candidates[:top_k], hybrid_conf

    # # Min-max
    # ce_min = min(ce_scores)
    # ce_max = max(ce_scores)
    # if (ce_max - ce_min) < 1e-6:
    #     norm_ce = [0.5]*len(ce_scores)
    # else:
    #     norm_ce = [(sc-ce_min)/(ce_max-ce_min) for sc in ce_scores]
    # final_conf = max(norm_ce)

    # sorted_pairs = sorted(zip(candidates, norm_ce), key=lambda x: x[1], reverse=True)
    # top_candidates = [c for c, _ in sorted_pairs[:top_k]]
    # return top_candidates, final_conf

    # part 2
    pairs = [(query, candidate) for candidate in candidates]
    ce_scores = cross_encoder.predict(pairs)
    # Normalize cross-encoder scores to [0,1]
    max_ce = max(ce_scores)
    norm_ce_scores = [score / max_ce for score in ce_scores] if max_ce > 0 else ce_scores
    reranked = sorted(zip(candidates, norm_ce_scores), key=lambda x: x[1], reverse=True)
    reranked_candidates = [cand for cand, score in reranked]
    # Update confidence with the top cross-encoder score if available.
    confidence = max(norm_ce_scores) * -1 if len(norm_ce_scores) > 0 else confidence
    return reranked_candidates, confidence

########################################
# 8) Generation & Post-Processing
########################################

def generate_answer(query: str, context: str, generation_model) -> str:
    if not context.strip():
        return "⚠️ No relevant context retrieved for this question."

    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    max_input_tokens = 1024

    instruction = (
        "You are a very knowledgeable financial analyst. "
        "Please provide a short, clear answer (1-2 sentences) only on the input financial data from yahoo financial"
        "Do NOT simply say 'I cannot find the info'. Try to provide an answer "
    )
    prompt = f"{instruction}\nContext: {context}\n\nQuestion: {query}\nAnswer:"

    # Trim prompt if too long
    prompt_tokens = tokenizer.encode(prompt)
    if len(prompt_tokens) > max_input_tokens:
        prompt_tokens = prompt_tokens[-max_input_tokens:]
        prompt = tokenizer.decode(prompt_tokens)

    # Generate
    output = generation_model(
        prompt,
        max_new_tokens=100,
        do_sample=True,
        top_p=0.95,
        num_return_sequences=1
    )

    st.write(output)
    text = output[0]["generated_text"]

    # Extract portion after "Answer:"
    if "Answer:" in text:
        answer_part = text.split("Answer:")[1].strip()
    else:
        answer_part = ""

    if not answer_part:
        answer_part = "⚠️ No answer generated. Possibly empty or unrelated."

    return answer_part

def extract_revenue(context: str) -> str:
    match = re.search(r"Total Revenue:\s*([\d,\.]+)", context)
    return match.group(1) if match else ""

def postprocess_answer(
    answer: str,
    query: str,
    context: str,
    dense_model: SentenceTransformer,
    sim_threshold: float = 0.2
) -> str:
    if "revenue" in query.lower():
        rev = extract_revenue(context)
        if rev and rev not in answer:
            answer += f" (Extracted Revenue: {rev})"

    # Compute similarity
    ans_emb = dense_model.encode([answer], convert_to_numpy=True)
    ctx_emb = dense_model.encode([context], convert_to_numpy=True)
    similarity = util.cos_sim(ans_emb, ctx_emb)[0][0].item()

    # Optionally clamp negative values
    # from [-1..1] => [0..1]
    sim_clamped = (similarity + 1) / 2

    if sim_clamped < sim_threshold:
        answer += f"\n⚠️ [Answer might not match context (similarity={sim_clamped:.2f})]."

    return answer

########################################
# 9) Input-Side Guardrail
########################################

def validate_query(query: str) -> bool:
    financial_keywords = [
        "revenue", "income", "profit", "financial", "earnings", "expenses",
        "balance sheet", "cashflow", "liabilities", "assets", "dividends",
        "equity", "EBITDA", "net income", "operating cash flow",
        "debt", "stock price", "market capitalization", "valuation", "EPS", "profit", "loss"
    ]
    return any(kw.lower() in query.lower() for kw in financial_keywords)
