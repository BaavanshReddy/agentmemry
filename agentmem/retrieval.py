"""
retrieval.py — Semantic search over stored memories.
Pure numpy, no external vector DB needed.
"""

import numpy as np
from typing import Optional
from .embeddings import from_bytes, cosine_similarity


def search(
    query_vector: np.ndarray,
    memories: list[dict],
    top_k: int = 5,
    threshold: float = 0.0,
) -> list[dict]:
    """
    Rank memories by cosine similarity to query_vector.

    Args:
        query_vector:  Embedded query (float32, L2-normalised).
        memories:      List of memory dicts from MemoryStorage.get_all().
        top_k:         Max number of results to return.
        threshold:     Minimum similarity score (0.0–1.0) to include.

    Returns:
        List of memory dicts with an added 'score' key, sorted best-first.
    """
    if not memories:
        return []

    scored = []
    for mem in memories:
        blob = mem.get("embedding")
        if blob is None:
            continue
        vec = from_bytes(blob)
        score = cosine_similarity(query_vector, vec)
        if score >= threshold:
            scored.append({**mem, "score": round(score, 4)})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def mmr_search(
    query_vector: np.ndarray,
    memories: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.5,
) -> list[dict]:
    """
    Maximal Marginal Relevance — balances relevance with diversity.
    Useful when you want a varied context window rather than near-duplicates.

    lambda_param=1.0  → pure relevance (same as search())
    lambda_param=0.0  → pure diversity
    """
    if not memories:
        return []

    candidates = []
    for mem in memories:
        blob = mem.get("embedding")
        if blob is None:
            continue
        vec = from_bytes(blob)
        score = cosine_similarity(query_vector, vec)
        candidates.append((score, vec, mem))

    selected = []
    selected_vecs = []

    while candidates and len(selected) < top_k:
        mmr_scores = []
        for rel_score, vec, mem in candidates:
            if selected_vecs:
                redundancy = max(
                    cosine_similarity(vec, sv) for sv in selected_vecs
                )
            else:
                redundancy = 0.0
            mmr = lambda_param * rel_score - (1 - lambda_param) * redundancy
            mmr_scores.append(mmr)

        best_idx = int(np.argmax(mmr_scores))
        rel_score, vec, mem = candidates.pop(best_idx)
        selected.append({**mem, "score": round(rel_score, 4)})
        selected_vecs.append(vec)

    return selected
