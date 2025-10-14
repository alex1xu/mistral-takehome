"""
Unit tests for similarity computation module.
"""

import numpy as np
import pytest
from src.similarity import cosine_similarity_matrix, top_k, normalize_vector


class TestCosineSimilarityMatrix:
    def test_identical_vectors(self):
        query = [1.0, 0.0]
        vectors = [[1.0, 0.0], [0.0, 1.0]]
        
        similarities = cosine_similarity_matrix(query, vectors)
        
        assert len(similarities) == 2
        assert similarities[0] == pytest.approx(1.0, abs=1e-6)
        assert similarities[1] == pytest.approx(0.0, abs=1e-6)
    
    def test_normalized_vectors(self):
        query = [0.6, 0.8] 
        vectors = [[0.6, 0.8], [-0.8, 0.6]]
        
        similarities = cosine_similarity_matrix(query, vectors)
        
        assert len(similarities) == 2
        assert similarities[0] == pytest.approx(1.0, abs=1e-6)
        assert similarities[1] == pytest.approx(0.0, abs=1e-6)  # orthog
    
    def test_empty_vectors(self):
        query = [1.0, 0.0]
        vectors = []
        
        similarities = cosine_similarity_matrix(query, vectors)
        
        assert len(similarities) == 0
    
class TestTopK:
    def test_basic_top_k(self):
        similarities = np.array([0.1, 0.9, 0.3, 0.7, 0.5])
        
        result = top_k(similarities, k=3)
        
        assert len(result) == 3
        assert result[0] == (1, 0.9)
        assert result[1] == (3, 0.7)
        assert result[2] == (4, 0.5)
    
    def test_empty_array(self):
        similarities = np.array([])
        
        result = top_k(similarities, k=3)
        
        assert result == []
    
    def test_ties_in_scores(self):
        similarities = np.array([0.5, 0.5, 0.3, 0.5])
        
        result = top_k(similarities, k=3)
        
        assert len(result) == 3
        scores = [score for _, score in result]
        assert all(score == 0.5 or score == 0.3 for score in scores)
        assert scores.count(0.5) == 3
    
class TestNormalizeVector:
    def test_basic_normalization(self):
        vector = [3.0, 4.0]
        
        normalized = normalize_vector(vector)
        
        norm = np.linalg.norm(normalized)
        assert norm == pytest.approx(1.0, abs=1e-6)
        
        expected = [3.0/5.0, 4.0/5.0] 
        assert normalized[0] == pytest.approx(expected[0], abs=1e-6)
        assert normalized[1] == pytest.approx(expected[1], abs=1e-6)