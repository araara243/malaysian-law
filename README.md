# Malaysian Legal RAG System

A Retrieval-Augmented Generation (RAG) pipeline for Malaysian statutory law. This system provides AI-powered legal question answering with citation support, built on ChromaDB for vector storage and **OpenRouter for language generation (free models with dynamic model selection)**.

## Overview

This project implements a complete RAG pipeline for querying Malaysian legal statutes. The system uses hybrid retrieval (combining semantic similarity and keyword matching) to find relevant legal provisions, then generates natural language answers with proper citations.

### Supported Legal Acts

| Category | Act | Act Number | Year |
|----------|-----|------------|------|
| **Commercial** | Contracts Act | Act 136 | 1950 |
| **Commercial** | Specific Relief Act | Act 137 | 1951 |
| **Commercial** | Partnership Act | Act 135 | 1961 |
| **Commercial** | Sale of Goods Act | Act 383 | 1957 |
| **Property** | Housing Development (Control and Licensing) Act | Act 118 | 1966 |
| **Property** | Strata Titles Act | Act 318 | 1985 |
| **Civil Procedure** | Courts of Judicature Act | Act 91 | 1964 |

### Performance Metrics

Based on evaluation against a 36-question golden dataset across 7 Acts:

| Category | Questions | Hit Rate @ 1 | Hit Rate @ 3 | MRR |
|----------|-----------|--------------|--------------|-----|
| Commercial | 20 | 95.0% | 100.0% | 0.975 |
| Property | 8 | 87.5% | 100.0% | 0.938 |
| Civil Procedure | 4 | 100.0% | 100.0% | 1.000 |
| **Overall** | **32** | **94.5%** | **100.0%** | **0.969** |

### Category-Based Evaluation

To evaluate performance per legal category:

```bash
python src/evaluation/evaluate_rag.py --categories --dataset tests/golden_dataset_expanded.json
```

### Recent Updates & Progress

- **Structural Semantic Chunking**: Migrated from arbitrary token-based chunking directly to section-aware chunking. This preserves whole statutory provisions grouped by Section numbers.
- **Cross-Encoder Reranking**: Introduced a local `cross-encoder/ms-marco-MiniLM-L-6-v2` reranking step. When integrated into hybrid search (BM25 + vector similarity using Reciprocal Rank Fusion), it significantly boosts retrieval accuracy for dense legal queries.
- **Repository Cleanup**: Performed a major cleanup shifting scattered documentation logs (debugging notes, evaluations) entirely into the `docs/` folder, and removing one-off experimental scripts.

---

## PostgreSQL Migration (Optional)

The system now supports PostgreSQL with pgvector as an alternative to ChromaDB for vector storage. This provides:

- **Production-ready database** with ACID guarantees
- **Normalized schema** for better data management
- **Scalability** for larger datasets
- **Concurrent access** with connection pooling
- **SQL interface** for custom queries

### Quick Start with PostgreSQL

1. **Install PostgreSQL and pgvector**
   ```bash
   # Ubuntu/Debian
   sudo apt install postgresql postgresql-contrib libpq-dev

   # macOS
   brew install postgresql@16

   # pgvector extension (see docs/architecture/POSTGRESQL_SETUP.md)
   ```

2. **Set up environment variables** in `.env`:
   ```
   PGHOST=localhost
   PGPORT=5432
   PGDATABASE=mylaw_rag
   PGUSER=mylaw_user
   PGPASSWORD=your_password
   ```

3. **Initialize the database schema**:
   ```bash
   psql -U mylaw_user -d mylaw_rag -f scripts/migrate/init_postgres_schema.sql
   ```

4. **Migrate existing data** (if using ChromaDB):
   ```bash
   # Export from ChromaDB
   python scripts/migrate/export_from_chroma.py chromadb_export.json

   # Import to PostgreSQL
   python scripts/migrate/import_to_postgres.py chromadb_export.json

   # Validate migration
   python scripts/migrate/validate_migration.py chromadb_export.json
   ```

5. **Use PostgreSQLRetriever** in your code:
   ```python
   from src.retrieval.postgresql_retriever import PostgreSQLRetriever
   from src.config import postgresql_config

   retriever = PostgreSQLRetriever(postgresql_config)
   results = retriever.retrieve("What is consideration?", n_results=5)
   ```

### Architecture Comparison

**ChromaDB (Default):**
- Simple setup, no external dependencies
- Local file-based storage
- Best for development and small datasets

**PostgreSQL + pgvector:**
- Requires database setup
- Client-server architecture
- Best for production and larger datasets
- SQL query capabilities

### Migration Documentation

For complete setup instructions, troubleshooting, and performance optimization, see [PostgreSQL Setup Document](docs/architecture/POSTGRESQL_SETUP.md).

---

## Architecture

For a detailed visual mapping of the data ingestion, retrieval, and generation layers, please consult the [System Architecture Document](docs/architecture/SYSTEM_ARCHITECTURE.md).

### Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12+ |
| LLM Framework | LangChain |
| LLM Provider | OpenRouter (Free Models with dynamic selection) |
| Vector Database | ChromaDB (local persistence) **or PostgreSQL + pgvector** |
| Embedding Model | sentence-transformers/all-MiniLM-L6-v2 (384 dims) |
| Keyword Search | BM25 (rank_bm25) |
| Web Interface | Streamlit |
| PDF Processing | pypdf |
| Database Adapter | psycopg2 (PostgreSQL) |

---

## Project Structure

```text
MyLaw-RAG/
├── data/
│   ├── raw/                        # Original PDF files from AGC
│   ├── processed/                  # Extracted text and chunks (JSON)
│   └── vector_db/                  # ChromaDB persistence directory
├── src/                    
│   ├── app/                        # Streamlit web application
│   ├── config.py                   # Centralized configuration
│   ├── db/                         # Database layer (PostgreSQL)
│   ├── evaluation/                 # Retrieval evaluation metrics
│   ├── generation/                 # LangChain RAG pipeline & prompts
│   ├── ingestion/                  # Scraping, extraction, chunking, and embedding
│   └── retrieval/                  # Hybrid search (BM25 + ChromaDB) & reranking
├── scripts/                        # Utility and migration scripts
├── tests/                          # Unit tests & golden datasets
├── docs/                           # Project documentation
│   └── architecture/               # Detailed architecture diagrams
│       ├── SYSTEM_ARCHITECTURE.md  # System architecture diagram
│       └── POSTGRESQL_SETUP.md     # PostgreSQL setup guide
├── requirements.txt
├── .env.example
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- OpenRouter API key (free tier available) or Google API key

### Setup Steps

1. **Create and activate virtual environment**

   ```bash
   cd MyLaw-RAG
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/macOS
   source venv/bin/activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   Copy the example environment file and add your API key:

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and set your OpenRouter API key (recommended for free models):

   ```
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   ```

   Get your free OpenRouter API key at [openrouter.ai/keys](https://openrouter.ai/keys).

   **Alternatively**, you can use Google Gemini by setting `GOOGLE_API_KEY` and changing the `llm_provider` in the code to `"gemini"`.

---

## Usage

### Running the Web Interface

```bash
streamlit run src/app/app.py
```

The application will be available at `http://localhost:8501`.

### AI Model Selection

The Streamlit sidebar includes a **dynamic model selector dropdown** that allows you to choose from:

- **Auto-route to Best Free Model** (default) - OpenRouter automatically selects the best available free model
- **Specific Free Models** - Choose from 28+ free models including:
  - Google: Gemma 3 (4B, 12B, 27B)
  - Meta: Llama 3.2/3.3 (3B, 70B)
  - Mistral: Mistral Small 3.1 24B
  - NVIDIA: Nemotron series
  - Qwen: Qwen3 series
  - And many more

**Features:**
- Model list cached for 1 hour to reduce API calls
- On-the-fly model switching during your session
- Graceful fallback to default if API key is missing
- Selection resets to default on page refresh (no persistence)

### Example Queries

- "What are the requirements for specific performance of a contract?"
- "When is a contract voidable due to coercion?"
- "What are the licensing requirements for housing developers?"

---

## Data Pipeline

The data ingestion pipeline consists of four stages. These only need to be run if rebuilding the database from scratch.

### Stage 1: PDF Download

Downloads legal act PDFs from the Attorney General's Chambers (AGC) website.

```bash
python src/ingestion/agc_scraper.py
```

Output: PDF files in `data/raw/`

### Stage 2: Text Extraction

Extracts and cleans text from PDFs, removing headers, footers, and watermarks.

```bash
python src/ingestion/text_extractor.py
```

Output: JSON files in `data/processed/` containing raw and cleaned text.

### Stage 3: Semantic Chunking

Splits documents into chunks by legal section boundaries rather than arbitrary token limits.

```bash
python src/ingestion/chunker.py
```

Output: `*_chunks.json` files in `data/processed/` containing chunked text with metadata.

### Stage 4: Vector Database Ingestion

Generates embeddings and stores chunks in ChromaDB.

```bash
python src/ingestion/vector_ingest.py
```

Output: ChromaDB collection in `data/vector_db/`

---

## Testing

### Unit Tests

Run the test suite using pytest:

```bash
pytest tests/test_rag.py -v
```

Test coverage includes:
- AGC scraper URL construction
- Text extraction and cleaning
- Section detection in chunker
- Hybrid retriever functionality
- Golden dataset retrieval accuracy

### Retrieval Evaluation

Run the evaluation script to compute Hit Rate and MRR metrics:

```bash
python src/evaluation/evaluate_rag.py
```

Results are saved to `tests/evaluation_results.json`.

---

## Configuration

The project uses a centralized configuration file at `src/config.py`. You can modify the `RAGConfig` dataclass to adjust parameters such as:

- **Chunking**: `chunk_size`, `chunk_overlap`
- **Retrieval**: `top_k`, `semantic_weight`, `keyword_weight`, `rrf_k`
- **Models**: `embedding_model`, `llm_model`, `temperature`
- **Vector DB**: `collection_name`

Environment variables are managed via `.env` file (see `.env.example`).

---

## Contribution

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

Please ensure you:
- Add type hints to new functions.
- Add docstrings obeying Google style.
- Run tests before submitting.

---

## Future Strategy

### Three-Tiered Dataset Evolution
The current `golden_dataset.json` consists primarily of "Factoid / Direct Retrieval" legal trivia queries (e.g. "What is fraud under the Contracts Act?"). While this is excellent for baseline testing of the ingestion, chunking, and embedding pipelines, it doesn't reflect real-world user interaction which typically involves complex *fact patterns*.

To build a truly robust AI-driven legal chatbot with strong reasoning capabilities, the evaluation dataset will evolve into three tiers:

1. **Tier 1: Factoid / Direct Retrieval**
   * *Purpose:* Sanity checking basic system plumbing ("Is our database working?").
   * *Status:* Implemented (`golden_dataset.json`, `golden_dataset_expanded.json`).
2. **Tier 2: Layperson Situational Queries**
   * *Purpose:* Testing intent matching. These are short, messy, and emotional queries from regular citizens (e.g., "My landlord locked me out, help!").
   * *Status:* Planned.
3. **Tier 3: Law Exam / Bar Exam Fact Patterns**
   * *Purpose:* Testing deep legal reasoning, issue spotting, and complex RAG synthesis. This simulates the "IRAC" method (Issue, Rule, Application, Conclusion) on complex, multi-issue fact patterns commonly found in university law exams (e.g., CLP past years).
   * *Status:* Planned. Requires implementation of an LLM-as-a-Judge evaluation pipeline due to the complexity of the ground truth answers.

---

## Limitations

1. **Scope**: Currently supports 7 Malaysian Acts across commercial, property, and civil procedure law. Expansion to other domains (criminal law, family law) requires additional PDF sources and re-ingestion.

2. **Currency**: Legal texts are static snapshots. Updates to legislation require manual re-ingestion.

3. **Not Legal Advice**: This system provides information retrieval and AI-generated summaries. It is not a substitute for professional legal counsel.

4. **API Dependency**: LLM generation requires an active API key (OpenRouter or Google). The system falls back to displaying raw retrieved sections when the API is unavailable. OpenRouter offers 28+ free tier models with no cost.

5. **Category Coverage**: While commercial and property law are well-represented, other legal domains have limited or no coverage.

---

## License

This project is for educational and research purposes. The legal texts are sourced from the Attorney General's Chambers of Malaysia and remain subject to their terms of use.
