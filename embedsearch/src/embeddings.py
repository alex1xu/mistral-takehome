"""
wrapper for Mistral embeddings API 
"""

import hashlib
import os
import requests
from typing import List, Sequence
import numpy as np
from mistralai import Mistral
from src.similarity import normalize_vector


def deterministic_embed(text: str, dim: int = 64) -> List[float]:
    """
    create deterministic mock embedding from text hash
    
    text: input text to embed
    dim: dimension of the embedding vector
    
    return normalized embedding vector as list
    """
    # create hash from text -> Convert bytes to floats and tile/truncate to dim
    h = hashlib.sha256(text.encode('utf-8')).digest()
    arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
    vec = np.tile(arr, int(np.ceil(dim / arr.size)))[:dim]
    
    normalized = normalize_vector(vec.tolist())
    
    return normalized


class EmbeddingClient:
    """
    Client for generating embeddings either from Mistral API or mock
    """
    
    def __init__(self, api_key: str | None = None):
        """
        If mistral api key is missing client will run in mock mode.
        """
        self.api_key = api_key or os.getenv('MISTRAL_API_KEY')
        self.mock_mode = self.api_key is None or self.api_key == ""
        
        if not self.mock_mode:
            self.mistral_client = Mistral(api_key=self.api_key)
            self.model = "mistral-embed"
    
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """
        texts: sequence of text strings to embed
            
        Return lList of embedding vectors, one per input text
        """
        if self.mock_mode:
            return self._mock_embed(texts)
        else:
            return self._real_embed(texts)
    
    def _mock_embed(self, texts: Sequence[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            truncated_text = text[:1000] # hack for this take home
            embedding = deterministic_embed(truncated_text)
            embeddings.append(embedding)
        return embeddings
    
    def _real_embed(self, texts: Sequence[str]) -> List[List[float]]:
        try:
            text_list = list(texts)
            
            embeddings_response = self.mistral_client.embeddings.create(
                model=self.model,
                inputs=text_list
            )
            
            embeddings = []
            for item in embeddings_response.data:
                embedding = item.embedding
                normalized_embedding = normalize_vector(embedding)
                embeddings.append(normalized_embedding)
            
            return embeddings
            
        except Exception as e:
            print(f"Error calling Mistral API: {e}")
            print("Falling back to mock embeddings")
            return self._mock_embed(texts)
