import os
import tempfile
import pytest
from pathlib import Path
from src.io_utils import read_text_files, save_index, load_index


class TestReadTextFiles:
    def test_read_text_files_basic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            file1_path = Path(temp_dir) / "test1.txt"
            file2_path = Path(temp_dir) / "test2.txt"
            
            file1_path.write_text("Hello world")
            file2_path.write_text("Machine learning")
            
            files = read_text_files(temp_dir)
            
            assert len(files) == 2
            assert any("test1.txt" in path for path, _ in files)
            assert any("test2.txt" in path for path, _ in files)