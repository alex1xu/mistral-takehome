import os
import pytest
import hashlib
import numpy as np
from unittest.mock import patch, MagicMock
from src.embeddings import EmbeddingClient, deterministic_embed

class TestEmbeddingClient:
    @patch.dict(os.environ, {}, clear=True)
    def test_mock_mode_no_api_key(self):
        """Test client in mock mode when no API key provided."""
        client = EmbeddingClient(api_key=None)
        
        assert client.mock_mode is True
        assert client.api_key is None
    
    @patch.dict(os.environ, {}, clear=True)
    def test_mock_mode_empty_api_key(self):
        """Test client in mock mode when empty API key provided."""
        client = EmbeddingClient(api_key="")
        
        assert client.mock_mode is True
    
    @patch.dict(os.environ, {}, clear=True)
    def test_mock_embed_single_text(self):
        """Test mock embedding of single text."""
        client = EmbeddingClient(api_key=None)
        
        texts = ["Hello, world!"]
        embeddings = client.embed(texts)
        
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 64
        assert np.linalg.norm(embeddings[0]) == pytest.approx(1.0, abs=1e-6)
    
class TestIntegration:
    def test_client_with_similarity(self):
        from src.similarity import cosine_similarity_matrix, top_k
        
        client = EmbeddingClient(api_key=None)
        
        texts = ["machine learning", "artificial intelligence", "completely unrelated topic"]
        
        embeddings = client.embed(texts)
        
        query_embedding = embeddings[0]  # Use first as query
        similarities = cosine_similarity_matrix(query_embedding, embeddings)
        
        top_results = top_k(similarities, k=2)
        
        assert len(top_results) >= 2
        
        # first result should be the query itself
        assert top_results[0][1] == pytest.approx(1.0, abs=1e-6)