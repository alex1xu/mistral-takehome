# CLI Embedding Search Tool

A CLI tool for indexing and searching for text documents within a directory by similarity using Mistral embeddings

## Quick Start

### Install & Setup
```bash
pip install -r requirements.txt

# Set API key (will use mock if not set)
export MISTRAL_API_KEY="your_api_key_here"
```

### Usage

**Index documents:**
```bash
# --data-dir to specify text file directory
# --mock to use a mock client
python embedsearch.py index --data-dir ./docs --output index.json --mock

# API mode
python embedsearch.py index --data-dir ./docs --output index.json
```

**Search documents:**
```bash
python embedsearch.py query --index index.json --k 2 "machine learning"
```

### Example Output
```
Loaded index with 2 documents
Top 2 results for: 'machine learning'
------------------------------------------------------------
1. ml_algorithms.txt (score: 0.831)
   Path: /path/to/ml_algorithms.txt

2. ai_overview.txt (score: 0.785)
   Path: /path/to/ai_overview.txt
```

## Development

**Run tests:**
```bash
pytest tests/ -v
```

**Project structure:**
```
embedsearch/
├── src/
│   ├── cli.py             # CLI commands
│   ├── indexer.py         # Document indexing
│   ├── embeddings.py      # Mistral API + mock
│   ├── similarity.py      # Cosine similarity
│   └── io_utils.py        # File I/O
├── tests/                 # Unit tests
├── embedsearch.py         # Executable script
└── requirements.txt       # Dependencies
```
