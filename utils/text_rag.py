from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


_RAG_INDEX_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def _minmax_norm_1d(scores: np.ndarray) -> np.ndarray:
    s = scores.astype(np.float64, copy=False)
    lo = float(np.min(s))
    hi = float(np.max(s))
    if hi - lo < 1e-12:
        return np.ones_like(s, dtype=np.float64)
    return (s - lo) / (hi - lo)


def _build_rag_index(csv_path: str) -> dict[str, Any]:
    df = pd.read_csv(csv_path)

    text_cols = ["type", "name", "description", "process_steps", "iso", "iso title"]
    for c in text_cols:
        if c not in df.columns:
            df[c] = ""
    row_texts = (
        df[text_cols]
        .fillna("")
        .astype(str)
        .agg(" | ".join, axis=1)
        .tolist()
    )

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(row_texts)
    analyzer = vectorizer.build_analyzer()
    corpus_tokens = [analyzer(t) for t in row_texts]
    bm25 = BM25Okapi(corpus_tokens)

    return {
        "df": df,
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "bm25": bm25,
        "analyzer": analyzer,
    }


def _get_rag_index(csv_path: str) -> dict[str, Any]:
    abs_path = os.path.abspath(csv_path)
    mtime = os.path.getmtime(abs_path)
    cached = _RAG_INDEX_CACHE.get(abs_path)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    payload = _build_rag_index(abs_path)
    _RAG_INDEX_CACHE[abs_path] = (mtime, payload)
    return payload


def retrieve_relevant_data(
    prompt_text: str,
    csv_path: str,
    top_k: int = 15,
    *,
    bm25_weight: float = 0.6,
) -> list[dict]:
    """
    Hybrid lexical retrieval: BM25 + cosine TF-IDF, fused by weighted sum after per-query min-max.
    Index is cached per absolute csv_path until mtime changes.
    """
    if not (0.0 <= bm25_weight <= 1.0):
        raise ValueError("bm25_weight must be between 0 and 1.")

    idx = _get_rag_index(csv_path)
    df = idx["df"]
    vectorizer = idx["vectorizer"]
    tfidf_matrix = idx["tfidf_matrix"]
    bm25 = idx["bm25"]
    analyzer = idx["analyzer"]

    query_tokens = analyzer(prompt_text)
    prompt_vec = vectorizer.transform([prompt_text])

    bm25_scores = np.asarray(bm25.get_scores(query_tokens), dtype=np.float64)
    tfidf_scores = cosine_similarity(prompt_vec, tfidf_matrix).flatten().astype(np.float64)

    bm25_n = _minmax_norm_1d(bm25_scores)
    tfidf_n = _minmax_norm_1d(tfidf_scores)
    hybrid = bm25_weight * bm25_n + (1.0 - bm25_weight) * tfidf_n

    top_indices = np.argsort(hybrid)[-top_k:][::-1]
    return df.iloc[top_indices].to_dict(orient="records")

