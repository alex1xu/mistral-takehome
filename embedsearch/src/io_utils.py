import json
import os
from typing import List, Tuple
from pathlib import Path


def read_text_files(data_dir: str) -> List[Tuple[str, str]]:
    """
    read all .txt files from a directory recursively
    
    data_dir: path to directory containing .txt files
        
    Returns list of (relative_path, text_content) tuples
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Directory {data_dir} does not exist")
    
    if not data_path.is_dir():
        raise NotADirectoryError(f"{data_dir} is not a directory")
    
    text_files = []
    
    for txt_file in data_path.rglob("*.txt"):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
            rel_path = txt_file.relative_to(data_path)
            text_files.append((str(rel_path), content))
            
        except Exception as e:
            print(f"Warning: Could not read {txt_file}: {e}")
            continue
    
    return text_files


def save_index(index: List[dict], path: str) -> None:
    """
    save index to JSON file
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


def load_index(path: str) -> List[dict]:
    """
    Load index from JSON file
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)