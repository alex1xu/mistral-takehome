import os
from typing import List, Dict
from src.io_utils import read_text_files, save_index
from src.embeddings import EmbeddingClient


def build_index(data_dir: str, embed_client: EmbeddingClient, output_path: str) -> None:
    """
    Build index from text files in a directory
    
    data_dir: Directory containing .txt files
    embed_client: Embedding client
    output_path: Path to save the index JSON file
    """
    text_files = read_text_files(data_dir)
    
    if not text_files:
        raise ValueError(f"No .txt files found in {data_dir}")
    
    texts = []
    file_metadata = []
    
    for rel_path, content in text_files:
        texts.append(content)
        file_metadata.append({
            'id': rel_path,
            'path': os.path.abspath(os.path.join(data_dir, rel_path))
        })
    
    embeddings = embed_client.embed(texts)
    
    index = []
    for i, (metadata, embedding) in enumerate(zip(file_metadata, embeddings)):
        entry = {
            'id': metadata['id'],
            'path': metadata['path'],
            'vector': embedding
        }
        index.append(entry)
    
    save_index(index, output_path)