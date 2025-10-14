import os
import tempfile
import pytest
from pathlib import Path
from src.indexer import build_index
from src.embeddings import EmbeddingClient

class TestBuildIndex:
    def test_build_index_basic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            file1_path = Path(temp_dir) / "test1.txt"
            file2_path = Path(temp_dir) / "test2.txt"
            
            file1_path.write_text("Machine learning algorithms")
            file2_path.write_text("Artificial intelligence systems")
            
            # Create mock embedding client
            client = EmbeddingClient(api_key=None)
            
            # Build index
            index_path = os.path.join(temp_dir, "test_index.json")
            build_index(temp_dir, client, index_path)
            
            assert os.path.exists(index_path)
            
            from src.io_utils import load_index
            index = load_index(index_path)
            
            assert len(index) == 2
