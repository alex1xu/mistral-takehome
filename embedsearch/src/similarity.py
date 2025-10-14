"""
Similarity computation module

Handles vector normalization and similarity scoring
calculate cosine similarity using numpy
"""

import numpy as np
from typing import List, Tuple

def cosine_similarity_matrix(query_vec: List[float], vectors: List[List[float]]) -> np.ndarray:
    """
    Return 1D array of cosine similarities (assuming normalized vectors)
    
    query_vec: a sinlge query vector (normalized)
    vectors: List of doc vectors (normalized)
    
    return numpy array of cosine similarities
    """
    if len(vectors) == 0:
        return np.array([], dtype=np.float32)
    
    query_array = np.array(query_vec, dtype=np.float32)
    vectors_array = np.array(vectors, dtype=np.float32)
    
    similarities = np.dot(vectors_array, query_array)
    
    return similarities


def top_k(similarities: np.ndarray, k: int = 3) -> List[Tuple[int, float]]:
    """
    Return list of (index, score) sorted  by score descending
    
    similarities: array of similarity scores
    k: number of top results to return
    
    return list of (index, similarity_score) tuples, sorted by score descending
    """
    if len(similarities) == 0:
        return []
    
    # limit k to the actual array size
    actual_k = min(k, len(similarities))
    
    top_indices = np.argpartition(similarities, -actual_k)[-actual_k:]
    
    top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
    
    return [(int(idx), float(similarities[idx])) for idx in top_indices]


def normalize_vector(vector: List[float]) -> List[float]:
    """
    L2 norm the vector
    
    vector: vector to normalize
    
    return normed vector as list
    """
    vec_array = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(vec_array)
    
    if norm == 0:
        return [0.0] * len(vector)
    
    normalized = vec_array / norm
    return normalized.tolist()
