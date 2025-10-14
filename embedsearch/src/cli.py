"""
CLI entrypoint for the tool

Provides index and query commands for doc retrieval
"""

import argparse
import sys
import os
from pathlib import Path
from src.indexer import build_index
from src.io_utils import load_index
from src.embeddings import EmbeddingClient
from src.similarity import cosine_similarity_matrix, top_k


def index_command(args):
    data_dir = args.data_dir
    output_path = args.output
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' does not exist")
        sys.exit(1)
    
    if not os.path.isdir(data_dir):
        print(f"Error: '{data_dir}' is not a directory")
        sys.exit(1)
    
    client = EmbeddingClient(api_key=None if args.mock else os.getenv('MISTRAL_API_KEY'))
    
    try:
        print(f"Building index from '{data_dir}'...")
        build_index(data_dir, client, output_path)
        print(f"Index saved to '{output_path}'")
    except Exception as e:
        print(f"Error building index: {e}")
        sys.exit(1)


def query_command(args):
    index_path = args.index
    query_text = args.query
    k = args.k
    
    if not os.path.exists(index_path):
        print(f"Error: Index file '{index_path}' does not exist")
        sys.exit(1)
    
    try:
        index = load_index(index_path)
        print(f"Loaded index with {len(index)} documents")
        
        client = EmbeddingClient(api_key=None if args.mock else os.getenv('MISTRAL_API_KEY'))
        
        query_embedding = client.embed([query_text])[0]
        
        doc_vectors = [entry['vector'] for entry in index]
        
        similarities = cosine_similarity_matrix(query_embedding, doc_vectors)
        
        top_results = top_k(similarities, k=k)
        
        print(f"\nTop {len(top_results)} results for: '{query_text}'")
        print("-" * 60)
        
        for i, (doc_idx, score) in enumerate(top_results, 1):
            doc = index[doc_idx]
            print(f"{i}. {doc['id']} (score: {score:.3f})")
            print(f"   Path: {doc['path']}")
            print()
        
        if not top_results:
            print("No results found.")
            
    except Exception as e:
        print(f"Error querying index: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Mistral take home local CLI embedding search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index documents (mock mode)
  python -m src.cli index --data-dir ./docs --output index.json --mock
  
  # Index documents (real API)
  python -m src.cli index --data-dir ./docs --output index.json
  
  # Query index
  python -m src.cli query --index index.json --k 3 "machine learning"
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Build index from text files')
    index_parser.add_argument('--data-dir', required=True, help='Directory containing .txt files')
    index_parser.add_argument('--output', required=True, help='Output index file path')
    index_parser.add_argument('--mock', action='store_true', help='Use mock embeddings (no API calls)')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Search for similar documents')
    query_parser.add_argument('--index', required=True, help='Index file path')
    query_parser.add_argument('--k', type=int, default=3, help='Number of top results (default: 3)')
    query_parser.add_argument('--mock', action='store_true', help='Use mock embeddings (no API calls)')
    query_parser.add_argument('query', help='Search query text')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'index':
        index_command(args)
    elif args.command == 'query':
        query_command(args)


if __name__ == '__main__':
    main()
