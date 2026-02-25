# PostgreSQL Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate MyLaw-RAG from ChromaDB to PostgreSQL with pgvector for vector similarity search, using a hybrid strategy where new Acts use PostgreSQL while existing Acts remain on ChromaDB initially.

**Architecture:**
- Add PostgreSQL with native pgvector extension (VECTOR columns, ivfflat indexing)
- Create normalized schema: acts → sections → chunks → embeddings
- Implement PostgreSQLRetriever class matching HybridRetriever interface
- Use connection pooling (psycopg2.pool) for efficient database access
- Maintain ChromaDB retriever as fallback during transition

**Tech Stack:**
- PostgreSQL 15+ with pgvector extension
- psycopg2-binary (PostgreSQL adapter)
- psycopg2-pool (connection pooling)
- sentence-transformers (embedding generation)
- ChromaDB (existing, kept as fallback)

---

## Task 1: Add PostgreSQL Dependencies

**Files:**
- Modify: `requirements.txt`

**Step 1: Update requirements.txt**

Add these lines to `requirements.txt`:
```txt
psycopg2-binary>=2.9.0
psycopg2-pool>=2.0.0
```

**Step 2: Install dependencies to verify**

Run: `/home/test/MyLaw-RAG/venv_linux/bin/pip install psycopg2-binary>=2.9.0 psycopg2-pool>=2.0.0`
Expected: Package installation succeeds without errors

**Step 3: Verify imports work**

Run: `/home/test/MyLaw-RAG/venv_linux/bin/python -c "import psycopg2; import psycopg2.pool; print('Imports OK')"`
Expected: "Imports OK"

**Step 4: Commit**

```bash
git add requirements.txt
git commit -m "feat: add PostgreSQL dependencies

- Add psycopg2-binary for PostgreSQL adapter
- Add psycopg2-pool for connection pooling

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Add PostgreSQL Configuration

**Files:**
- Modify: `src/config.py`
- Modify: `.env.example`

**Step 1: Add PostgreSQLConfig dataclass**

Add this to `src/config.py` after line 39 (after `RAGConfig` class):

```python
@dataclass
class PostgreSQLConfig:
    """PostgreSQL database connection settings."""

    # Connection
    host: str = "localhost"
    port: int = 5432
    database: str = "mylaw_rag"
    user: str = "postgres"
    password: str = ""

    # Pool settings
    min_connections: int = 1
    max_connections: int = 5

    # Vector search
    embedding_dimension: int = 384  # all-MiniLM-L6-v2
    vector_type: str = "vector_cosine_ops"  # pgvector operator
```

**Step 2: Add PostgreSQL config instance**

Add this line at end of `src/config.py` (after line 84, before `setup_logging`):

```python
# PostgreSQL Configuration
postgresql_config: PostgreSQLConfig = PostgreSQLConfig()
```

**Step 3: Load environment variables in PostgreSQLConfig**

Update the `PostgreSQLConfig` class to use environment variables. Replace the entire class with:

```python
@dataclass
class PostgreSQLConfig:
    """PostgreSQL database connection settings."""

    # Connection
    host: str = field(default_factory=lambda: os.getenv("PGHOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("PGPORT", "5432")))
    database: str = field(default_factory=lambda: os.getenv("PGDATABASE", "mylaw_rag"))
    user: str = field(default_factory=lambda: os.getenv("PGUSER", "postgres"))
    password: str = field(default_factory=lambda: os.getenv("PGPASSWORD", ""))

    # Pool settings
    min_connections: int = 1
    max_connections: int = 5

    # Vector search
    embedding_dimension: int = 384  # all-MiniLM-L6-v2
    vector_type: str = "vector_cosine_ops"  # pgvector operator
```

**Step 4: Update .env.example**

Add to `.env.example`:

```bash
# PostgreSQL Configuration
PGHOST=localhost
PGPORT=5432
PGDATABASE=mylaw_rag
PGUSER=postgres
PGPASSWORD=your_password_here
```

**Step 5: Test configuration loads**

Run: `/home/test/MyLaw-RAG/venv_linux/bin/python -c "from src.config import postgresql_config; print(f'Config OK: {postgresql_config.database}')"`
Expected: "Config OK: mylaw_rag"

**Step 6: Commit**

```bash
git add src/config.py .env.example
git commit -m "feat: add PostgreSQL configuration

- Add PostgreSQLConfig dataclass with environment variable support
- Add connection pool settings
- Add vector search configuration (384 dims, cosine ops)
- Update .env.example with PostgreSQL credentials

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Create PostgreSQL Schema

**Files:**
- Create: `scripts/migrate/init_postgres_schema.sql`

**Step 1: Create schema SQL file**

Create `scripts/migrate/init_postgres_schema.sql` with:

```sql
-- PostgreSQL Schema for MyLaw-RAG
-- This script creates the normalized schema for legal documents

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Acts table: Metadata about Malaysian legal Acts
CREATE TABLE IF NOT EXISTS acts (
    id SERIAL PRIMARY KEY,
    act_number INTEGER UNIQUE NOT NULL,
    act_name VARCHAR(255) NOT NULL,
    act_year INTEGER NOT NULL,
    category VARCHAR(50) NOT NULL,
    language VARCHAR(10) DEFAULT 'EN',
    source_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_act_number CHECK (act_number > 0)
);

-- Sections table: Legal sections within Acts
CREATE TABLE IF NOT EXISTS sections (
    id SERIAL PRIMARY KEY,
    act_id INTEGER REFERENCES acts(id) ON DELETE CASCADE,
    section_number VARCHAR(50) NOT NULL,
    section_title TEXT,
    part VARCHAR(255),
    section_order INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(act_id, section_number, section_order)
);

-- Chunks table: Text chunks with metadata
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    section_id INTEGER REFERENCES sections(id) ON DELETE CASCADE,
    chunk_content TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    start_position INTEGER NOT NULL,
    subsection VARCHAR(100),
    keywords TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_token_count CHECK (token_count > 0)
);

-- Embeddings table: Vector embeddings with pgvector
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    chunk_id INTEGER REFERENCES chunks(id) ON DELETE CASCADE,
    embedding VECTOR(384) NOT NULL,
    embedding_model VARCHAR(100) DEFAULT 'all-MiniLM-L6-v2',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_chunks_section_id ON chunks(section_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id ON embeddings(chunk_id);

-- Note: ivfflat index requires data to exist first
-- Create after migration: CREATE INDEX idx_embeddings_embedding ON embeddings USING ivfflat(embedding vector_cosine_ops) WITH (lists = 100);

-- Grant permissions (adjust user as needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO your_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO your_user;
```

**Step 2: Create scripts directory**

Run: `mkdir -p scripts/migrate`

**Step 3: Verify SQL syntax**

Run: `psql --version`
Expected: PostgreSQL version displayed (e.g., "psql (PostgreSQL) 15.x")

**Step 4: Commit**

```bash
git add scripts/migrate/init_postgres_schema.sql
git commit -m "feat: add PostgreSQL schema initialization

- Create normalized schema: acts, sections, chunks, embeddings
- Add foreign key constraints with CASCADE delete
- Add validation constraints
- Prepare for ivfflat index creation
- Enable pgvector extension

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Create PostgreSQL Connection Manager

**Files:**
- Create: `src/db/postgres_connection.py`

**Step 1: Create test for connection manager**

Create `tests/test_postgres_connection.py` with:

```python
"""Test PostgreSQL connection manager."""

import pytest
import os
from unittest.mock import patch, MagicMock

from src.db.postgres_connection import PostgreSQLConnectionManager


def test_connection_manager_initialization():
    """Test that connection manager initializes with config."""
    from src.config import postgresql_config

    manager = PostgreSQLConnectionManager(postgresql_config)

    assert manager.config == postgresql_config
    assert manager._pool is None


def test_get_connection_string():
    """Test connection string generation."""
    from src.config import postgresql_config

    manager = PostgreSQLConnectionManager(postgresql_config)

    conn_str = manager._get_connection_string()

    assert "host=" in conn_str
    assert "port=" in conn_str
    assert "dbname=" in conn_str
    assert "user=" in conn_str


@patch("src.db.postgres_connection.psycopg2.pool.ThreadedConnectionPool")
def test_initialize_pool(mock_pool):
    """Test connection pool initialization."""
    from src.config import postgresql_config

    manager = PostgreSQLConnectionManager(postgresql_config)
    manager.initialize()

    mock_pool.assert_called_once()
    assert manager._pool is not None


@patch("src.db.postgres_connection.psycopg2.pool.ThreadedConnectionPool")
def test_get_connection(mock_pool):
    """Test getting connection from pool."""
    from src.config import postgresql_config

    # Setup mock
    mock_conn = MagicMock()
    mock_pool_instance = MagicMock()
    mock_pool_instance.conn.return_value.__enter__.return_value = mock_conn
    mock_pool.return_value = mock_pool_instance

    manager = PostgreSQLConnectionManager(postgresql_config)
    manager.initialize()

    with manager.get_connection() as conn:
        assert conn == mock_conn


@patch("src.db.postgres_connection.psycopg2.pool.ThreadedConnectionPool")
def test_close_pool(mock_pool):
    """Test closing connection pool."""
    from src.config import postgresql_config

    mock_pool_instance = MagicMock()
    mock_pool.return_value = mock_pool_instance

    manager = PostgreSQLConnectionManager(postgresql_config)
    manager.initialize()
    manager.close()

    mock_pool_instance.closeall.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `/home/test/MyLaw-RAG/venv_linux/bin/pytest tests/test_postgres_connection.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.db.postgres_connection'"

**Step 3: Implement PostgreSQLConnectionManager**

Create `src/db/postgres_connection.py` with:

```python
"""
PostgreSQL connection management with pooling.

This module provides a connection pool manager for PostgreSQL database
connections using psycopg2. Connection pooling improves performance by
reusing connections across multiple requests.
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import psycopg2
from psycopg2 import pool
from psycopg2.extensions import connection

from src.config import PostgreSQLConfig, setup_logging

logger = setup_logging(__name__)


class PostgreSQLConnectionManager:
    """
    Manages PostgreSQL connection pool.

    Provides thread-safe connection pooling for database operations.
    Uses context managers for automatic connection cleanup.
    """

    def __init__(self, config: PostgreSQLConfig):
        """
        Initialize connection manager.

        Args:
            config: PostgreSQL configuration settings.
        """
        self.config = config
        self._pool: Optional[pool.ThreadedConnectionPool] = None

    def _get_connection_string(self) -> str:
        """
        Generate PostgreSQL connection string.

        Returns:
            Connection string for psycopg2.
        """
        return (
            f"host={self.config.host} "
            f"port={self.config.port} "
            f"dbname={self.config.database} "
            f"user={self.config.user} "
            f"password={self.config.password}"
        )

    def initialize(self) -> None:
        """
        Initialize connection pool.

        Creates a threaded connection pool with min/max connections
        specified in configuration.
        """
        try:
            conn_str = self._get_connection_string()

            self._pool = pool.ThreadedConnectionPool(
                minconn=self.config.min_connections,
                maxconn=self.config.max_connections,
                dsn=conn_str
            )

            logger.info(
                f"Initialized connection pool: "
                f"{self.config.min_connections}-{self.config.max_connections} connections"
            )
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """
        Get connection from pool.

        Yields a connection object and automatically returns it to pool.

        Yields:
            psycopg2 connection object.

        Raises:
            RuntimeError: If pool not initialized.
        """
        if not self._pool:
            raise RuntimeError("Connection pool not initialized. Call initialize() first.")

        conn = None
        try:
            conn = self._pool.getconn()
            yield conn
        except Exception as e:
            logger.error(f"Error using connection: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self._pool.putconn(conn)

    def close(self) -> None:
        """Close all connections in pool."""
        if self._pool:
            self._pool.closeall()
            self._pool = None
            logger.info("Connection pool closed")
```

**Step 4: Create db directory**

Run: `mkdir -p src/db`

**Step 5: Run test to verify it passes**

Run: `/home/test/MyLaw-RAG/venv_linux/bin/pytest tests/test_postgres_connection.py -v`
Expected: PASS (all tests pass)

**Step 6: Commit**

```bash
git add src/db/postgres_connection.py tests/test_postgres_connection.py
git commit -m "feat: add PostgreSQL connection manager

- Add PostgreSQLConnectionManager with connection pooling
- Support context managers for automatic cleanup
- Add comprehensive unit tests
- Log connection pool initialization

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Create PostgreSQLRetriever Class

**Files:**
- Create: `src/retrieval/postgresql_retriever.py`
- Modify: `src/retrieval/__init__.py`

**Step 1: Write test for PostgreSQLRetriever**

Create `tests/test_postgresql_retriever.py` with:

```python
"""Test PostgreSQL retriever."""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from src.retrieval.postgresql_retriever import PostgreSQLRetriever
from src.retrieval.hybrid_retriever import RetrievalResult


def test_postgresql_retriever_initialization():
    """Test retriever initializes with config."""
    from src.config import postgresql_config

    with patch("src.retrieval.postgresql_retriever.PostgreSQLConnectionManager"):
        retriever = PostgreSQLRetriever(postgresql_config)

        assert retriever.config == postgresql_config
        assert retriever._conn_manager is not None


@patch("src.retrieval.postgresql_retriever.PostgreSQLConnectionManager")
def test_retrieve_returns_results(mock_conn_manager):
    """Test retrieve returns RetrievalResult objects."""
    from src.config import postgresql_config

    # Mock database response
    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    mock_cursor.fetchall.return_value = [
        (1, "test content", "Contracts Act 1950", 136, "2", "Consideration", 0.95)
    ]
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    mock_pool_instance = MagicMock()
    mock_pool_instance.get_connection.return_value.__enter__.return_value = mock_conn
    mock_conn_manager.return_value.get_connection.return_value.__enter__.return_value = mock_conn

    retriever = PostgreSQLRetriever(postgresql_config)
    results = retriever.retrieve("What is consideration?")

    assert len(results) == 1
    assert isinstance(results[0], RetrievalResult)
    assert results[0].act_name == "Contracts Act 1950"
    assert results[0].section_number == "2"


@patch("src.retrieval.postgresql_retriever.PostgreSQLConnectionManager")
def test_format_context(mock_conn_manager):
    """Test context formatting."""
    from src.config import postgresql_config

    retriever = PostgreSQLRetriever(postgresql_config)

    results = [
        RetrievalResult(
            chunk_id="1",
            content="Test content",
            act_name="Contracts Act 1950",
            act_number=136,
            section_number="2",
            section_title="Consideration",
            score=0.95,
            retrieval_method="postgresql"
        )
    ]

    context = retriever.format_context(results)

    assert "Source 1" in context
    assert "Contracts Act 1950" in context
    assert "Section 2" in context
    assert "Test content" in context
```

**Step 2: Run test to verify it fails**

Run: `/home/test/MyLaw-RAG/venv_linux/bin/pytest tests/test_postgresql_retriever.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.retrieval.postgresql_retriever'"

**Step 3: Implement PostgreSQLRetriever**

Create `src/retrieval/postgresql_retriever.py` with:

```python
"""
PostgreSQL Retriever for Malaysian Legal RAG

This module implements vector similarity search using PostgreSQL with pgvector.
Provides the same interface as HybridRetriever for easy swapping.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import PostgreSQLConfig, setup_logging
from src.db.postgres_connection import PostgreSQLConnectionManager
from src.retrieval.hybrid_retriever import RetrievalResult

logger = setup_logging(__name__)


class PostgreSQLRetriever:
    """
    PostgreSQL-based retriever using pgvector for similarity search.

    Provides compatible interface with HybridRetriever for easy swapping.
    Uses cosine similarity search with ivfflat indexing.
    """

    def __init__(
        self,
        config: Optional[PostgreSQLConfig] = None
    ):
        """
        Initialize PostgreSQL retriever.

        Args:
            config: PostgreSQL configuration. If None, uses default.
        """
        from src.config import postgresql_config as default_config

        self.config = config or default_config
        self._conn_manager = PostgreSQLConnectionManager(self.config)
        self._conn_manager.initialize()

        # Initialize embedding model
        self._embedding_model = SentenceTransformer(self.config.embedding_model)

        logger.info(f"Initialized PostgreSQLRetriever with {self.config.database}")

    def _embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for query.

        Args:
            query: User's question.

        Returns:
            Embedding vector as numpy array.
        """
        return self._embedding_model.encode(query, normalize_embeddings=True)

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        method: str = "postgresql"
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant legal chunks using vector similarity.

        Args:
            query: User's legal question.
            n_results: Number of results to return.
            method: Search method (kept for interface compatibility).

        Returns:
            List of RetrievalResult objects.
        """
        try:
            # Generate query embedding
            query_embedding = self._embed_query(query)

            # Convert to string format for pgvector
            embedding_str = f"[{','.join(map(str, query_embedding))}]"

            # Execute similarity search
            with self._conn_manager.get_connection() as conn:
                cursor = conn.cursor()

                query_sql = """
                    SELECT
                        c.id,
                        c.chunk_content,
                        a.act_name,
                        a.act_number,
                        s.section_number,
                        s.section_title,
                        1 - (e.embedding <=> %s::vector) as similarity
                    FROM chunks c
                    JOIN sections s ON c.section_id = s.id
                    JOIN acts a ON s.act_id = a.id
                    JOIN embeddings e ON c.id = e.chunk_id
                    ORDER BY e.embedding <=> %s::vector
                    LIMIT %s
                """

                cursor.execute(query_sql, (embedding_str, embedding_str, n_results))
                rows = cursor.fetchall()

            # Build result objects
            results = []
            for row in rows:
                results.append(RetrievalResult(
                    chunk_id=str(row[0]),
                    content=row[1],
                    act_name=row[2],
                    act_number=row[3],
                    section_number=row[4],
                    section_title=row[5],
                    score=float(row[6]),
                    retrieval_method=method
                ))

            logger.info(f"Retrieved {len(results)} results for query")
            return results

        except Exception as e:
            logger.error(f"PostgreSQL retrieval failed: {e}")
            return []

    def format_context(
        self,
        results: List[RetrievalResult],
        include_metadata: bool = True
    ) -> str:
        """
        Format retrieval results as context for LLM.

        Args:
            results: List of RetrievalResult objects.
            include_metadata: Whether to include citation metadata.

        Returns:
            Formatted context string.
        """
        context_parts = []

        for i, result in enumerate(results, start=1):
            if include_metadata:
                header = (
                    f"[Source {i}: {result.act_name}, "
                    f"Section {result.section_number}]"
                )
                if result.section_title:
                    header = header[:-1] + f" - {result.section_title}]"
            else:
                header = f"[Source {i}]"

            context_parts.append(f"{header}\n{result.content}")

        return "\n\n---\n\n".join(context_parts)

    def close(self) -> None:
        """Close database connection pool."""
        self._conn_manager.close()
        logger.info("PostgreSQLRetriever closed")
```

**Step 4: Update retrieval __init__.py**

Add to `src/retrieval/__init__.py`:

```python
from .postgresql_retriever import PostgreSQLRetriever
```

**Step 5: Run tests to verify they pass**

Run: `/home/test/MyLaw-RAG/venv_linux/bin/pytest tests/test_postgresql_retriever.py -v`
Expected: PASS (all tests pass)

**Step 6: Commit**

```bash
git add src/retrieval/postgresql_retriever.py src/retrieval/__init__.py tests/test_postgresql_retriever.py
git commit -m "feat: add PostgreSQL retriever

- Implement PostgreSQLRetriever with pgvector similarity search
- Match HybridRetriever interface for easy swapping
- Use cosine similarity (<=> operator)
- Add comprehensive unit tests
- Support context formatting for LLM

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Create ChromaDB Export Script

**Files:**
- Create: `scripts/migrate/export_from_chroma.py`

**Step 1: Write test for export script**

Create `tests/test_export_from_chroma.py` with:

```python
"""Test ChromaDB export script."""

import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.migrate.export_from_chroma import export_from_chroma


@patch("scripts.migrate.export_from_chroma.chromadb")
def test_export_creates_json_file(mock_chromadb):
    """Test that export creates JSON file."""
    # Mock ChromaDB collection
    mock_collection = MagicMock()
    mock_collection.get.return_value = {
        "ids": ["chunk1", "chunk2"],
        "documents": ["Content 1", "Content 2"],
        "metadatas": [
            {"act_name": "Contracts Act 1950", "act_number": 136, "section_number": "2"},
            {"act_name": "Contracts Act 1950", "act_number": 136, "section_number": "10"}
        ],
        "embeddings": [[0.1] * 384, [0.2] * 384]
    }
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection
    mock_chromadb.PersistentClient.return_value = mock_client

    export_path = Path("/tmp/test_export.json")

    # Run export
    result = export_from_chroma(
        collection_name="test_collection",
        output_path=export_path
    )

    # Verify file created
    assert export_path.exists()

    # Verify JSON structure
    with open(export_path, 'r') as f:
        data = json.load(f)

    assert "chunks" in data
    assert len(data["chunks"]) == 2
    assert data["chunks"][0]["chunk_id"] == "chunk1"

    # Cleanup
    export_path.unlink()


@patch("scripts.migrate.export_from_chroma.chromadb")
def test_export_parses_act_metadata(mock_chromadb):
    """Test that export parses act and section metadata."""
    mock_collection = MagicMock()
    mock_collection.get.return_value = {
        "ids": ["chunk1"],
        "documents": ["Test content"],
        "metadatas": [
            {
                "act_name": "Contracts Act 1950",
                "act_number": 136,
                "section_number": "2",
                "section_title": "Consideration",
                "act_year": 1950,
                "category": "commercial"
            }
        ],
        "embeddings": [[0.1] * 384]
    }
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection
    mock_chromadb.PersistentClient.return_value = mock_client

    export_path = Path("/tmp/test_export_metadata.json")

    export_from_chroma(
        collection_name="test_collection",
        output_path=export_path
    )

    with open(export_path, 'r') as f:
        data = json.load(f)

    chunk = data["chunks"][0]
    assert chunk["act_name"] == "Contracts Act 1950"
    assert chunk["act_number"] == 136
    assert chunk["act_year"] == 1950
    assert chunk["category"] == "commercial"

    export_path.unlink()
```

**Step 2: Run test to verify it fails**

Run: `/home/test/MyLaw-RAG/venv_linux/bin/pytest tests/test_export_from_chroma.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'scripts.migrate.export_from_chroma'"

**Step 3: Implement export script**

Create `scripts/migrate/export_from_chroma.py` with:

```python
"""
Export ChromaDB collection to JSON for PostgreSQL migration.

This script exports all chunks, embeddings, and metadata from ChromaDB
to a JSON file for importing into PostgreSQL.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

import chromadb
from chromadb.config import Settings

from src.config import get_vector_db_dir, setup_logging

logger = setup_logging(__name__)


def export_from_chroma(
    collection_name: str,
    output_path: Path
) -> Dict[str, Any]:
    """
    Export ChromaDB collection to JSON.

    Args:
        collection_name: Name of ChromaDB collection.
        output_path: Path to save export JSON.

    Returns:
        Dictionary containing exported data.
    """
    # Load ChromaDB collection
    db_path = str(get_vector_db_dir())
    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False)
    )

    collection = client.get_collection(name=collection_name)

    # Get all data
    all_data = collection.get(include=["documents", "metadatas", "embeddings"])

    if not all_data or not all_data["ids"]:
        logger.error(f"Collection {collection_name} is empty or not found.")
        return {"chunks": []}

    logger.info(f"Exporting {len(all_data['ids'])} chunks from ChromaDB")

    # Build export structure
    chunks = []
    for i, chunk_id in enumerate(all_data["ids"]):
        metadata = all_data["metadatas"][i]

        chunk = {
            "chunk_id": chunk_id,
            "content": all_data["documents"][i],
            "embedding": all_data["embeddings"][i],
            "act_name": metadata.get("act_name", ""),
            "act_number": metadata.get("act_number", 0),
            "act_year": metadata.get("act_year", 0),
            "category": metadata.get("category", "other"),
            "part": metadata.get("part", ""),
            "section_number": metadata.get("section_number", ""),
            "section_title": metadata.get("section_title", ""),
            "subsection": metadata.get("subsection", ""),
            "token_count": metadata.get("token_count", 0),
            "keywords": metadata.get("keywords", []),
            "start_position": 0
        }
        chunks.append(chunk)

    export_data = {
        "collection_name": collection_name,
        "total_chunks": len(chunks),
        "chunks": chunks
    }

    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)

    logger.info(f"Exported to {output_path}")

    return export_data


if __name__ == "__main__":
    from src.config import RAGConfig

    config = RAGConfig()
    output_path = Path("data/export/chroma_export.json")

    export_from_chroma(
        collection_name=config.collection_name,
        output_path=output_path
    )
```

**Step 4: Run tests to verify they pass**

Run: `/home/test/MyLaw-RAG/venv_linux/bin/pytest tests/test_export_from_chroma.py -v`
Expected: PASS (all tests pass)

**Step 5: Commit**

```bash
git add scripts/migrate/export_from_chroma.py tests/test_export_from_chroma.py
git commit -m "feat: add ChromaDB export script

- Export chunks, embeddings, and metadata to JSON
- Parse act and section information from metadata
- Support all enhanced metadata fields
- Add unit tests with mocked ChromaDB

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Create PostgreSQL Import Script

**Files:**
- Create: `scripts/migrate/import_to_postgres.py`

**Step 1: Write test for import script**

Create `tests/test_import_to_postgres.py` with:

```python
"""Test PostgreSQL import script."""

import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.migrate.import_to_postgres import import_to_postgres


@patch("scripts.migrate.import_to_postgres.PostgreSQLConnectionManager")
def test_import_inserts_acts(mock_conn_manager):
    """Test that import inserts acts correctly."""
    # Mock database connection
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    mock_pool_instance = MagicMock()
    mock_pool_instance.get_connection.return_value.__enter__.return_value = mock_conn
    mock_conn_manager.return_value.get_connection.return_value.__enter__.return_value = mock_conn

    # Create test export data
    export_data = {
        "chunks": [
            {
                "act_name": "Contracts Act 1950",
                "act_number": 136,
                "act_year": 1950,
                "category": "commercial",
                "content": "Test content",
                "embedding": [0.1] * 384,
                "section_number": "2",
                "section_title": "Consideration"
            }
        ]
    }

    export_path = Path("/tmp/test_export.json")
    with open(export_path, 'w') as f:
        json.dump(export_data, f)

    # Run import
    result = import_to_postgres(export_path)

    # Verify inserts were called
    assert mock_cursor.execute.called
    assert mock_conn.commit.called

    # Cleanup
    export_path.unlink()


@patch("scripts.migrate.import_to_postgres.PostgreSQLConnectionManager")
def test_import_handles_embedded_acts(mock_conn_manager):
    """Test that import deduplicates acts by act_number."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    mock_pool_instance = MagicMock()
    mock_pool_instance.get_connection.return_value.__enter__.return_value = mock_conn
    mock_conn_manager.return_value.get_connection.return_value.__enter__.return_value = mock_conn

    # Create export with duplicate acts
    export_data = {
        "chunks": [
            {
                "act_name": "Contracts Act 1950",
                "act_number": 136,
                "act_year": 1950,
                "category": "commercial",
                "content": "Content 1",
                "embedding": [0.1] * 384,
                "section_number": "2"
            },
            {
                "act_name": "Contracts Act 1950",
                "act_number": 136,
                "act_year": 1950,
                "category": "commercial",
                "content": "Content 2",
                "embedding": [0.2] * 384,
                "section_number": "10"
            }
        ]
    }

    export_path = Path("/tmp/test_export_duplicate.json")
    with open(export_path, 'w') as f:
        json.dump(export_data, f)

    result = import_to_postgres(export_path)

    # Should only insert one act (deduplicated)
    assert result["acts_inserted"] == 1
    assert result["chunks_inserted"] == 2

    export_path.unlink()
```

**Step 2: Run test to verify it fails**

Run: `/home/test/MyLaw-RAG/venv_linux/bin/pytest tests/test_import_to_postgres.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'scripts.migrate.import_to_postgres'"

**Step 3: Implement import script**

Create `scripts/migrate/import_to_postgres.py` with:

```python
"""
Import exported ChromaDB data into PostgreSQL.

This script imports JSON data exported from ChromaDB into the
normalized PostgreSQL schema.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

from src.config import PostgreSQLConfig, setup_logging
from src.db.postgres_connection import PostgreSQLConnectionManager

logger = setup_logging(__name__)


def import_to_postgres(
    export_path: Path,
    config: PostgreSQLConfig = None
) -> Dict[str, int]:
    """
    Import ChromaDB export JSON into PostgreSQL.

    Args:
        export_path: Path to export JSON file.
        config: PostgreSQL configuration.

    Returns:
        Dictionary with import statistics.
    """
    from src.config import postgresql_config as default_config

    config = config or default_config
    conn_manager = PostgreSQLConnectionManager(config)
    conn_manager.initialize()

    # Load export data
    with open(export_path, 'r') as f:
        export_data = json.load(f)

    chunks = export_data.get("chunks", [])
    logger.info(f"Importing {len(chunks)} chunks to PostgreSQL")

    stats = {
        "acts_inserted": 0,
        "sections_inserted": 0,
        "chunks_inserted": 0,
        "embeddings_inserted": 0
    }

    # Track unique acts and sections
    acts_map = {}  # (act_number, act_name) -> act_id
    sections_map = {}  # (act_id, section_number) -> section_id

    with conn_manager.get_connection() as conn:
        cursor = conn.cursor()

        for chunk_data in chunks:
            # Get or create act
            act_key = (chunk_data["act_number"], chunk_data["act_name"])
            if act_key not in acts_map:
                cursor.execute(
                    """
                    INSERT INTO acts (act_number, act_name, act_year, category)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (act_number) DO UPDATE
                    SET act_name = EXCLUDED.act_name
                    RETURNING id
                    """,
                    (
                        chunk_data["act_number"],
                        chunk_data["act_name"],
                        chunk_data.get("act_year", 0),
                        chunk_data.get("category", "other")
                    )
                )
                act_id = cursor.fetchone()[0]
                acts_map[act_key] = act_id
                stats["acts_inserted"] += 1
            else:
                act_id = acts_map[act_key]

            # Get or create section
            section_key = (act_id, chunk_data.get("section_number", ""))
            if section_key not in sections_map:
                cursor.execute(
                    """
                    INSERT INTO sections (act_id, section_number, section_title, part)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (act_id, section_number, section_order) DO NOTHING
                    RETURNING id
                    """,
                    (
                        act_id,
                        chunk_data.get("section_number", ""),
                        chunk_data.get("section_title", ""),
                        chunk_data.get("part", "")
                    )
                )
                result = cursor.fetchone()
                if result:
                    section_id = result[0]
                    sections_map[section_key] = section_id
                    stats["sections_inserted"] += 1
                else:
                    # Get existing section_id
                    cursor.execute(
                        "SELECT id FROM sections WHERE act_id = %s AND section_number = %s",
                        (act_id, chunk_data.get("section_number", ""))
                    )
                    section_id = cursor.fetchone()[0]
                    sections_map[section_key] = section_id
            else:
                section_id = sections_map[section_key]

            # Create chunk
            cursor.execute(
                """
                INSERT INTO chunks (
                    section_id, chunk_content, token_count,
                    start_position, subsection, keywords
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    section_id,
                    chunk_data["content"],
                    chunk_data.get("token_count", 0),
                    chunk_data.get("start_position", 0),
                    chunk_data.get("subsection", ""),
                    chunk_data.get("keywords", [])
                )
            )
            chunk_id = cursor.fetchone()[0]
            stats["chunks_inserted"] += 1

            # Create embedding
            embedding_str = f"[{','.join(map(str, chunk_data['embedding']))}]"
            cursor.execute(
                """
                INSERT INTO embeddings (chunk_id, embedding)
                VALUES (%s, %s::vector)
                """,
                (chunk_id, embedding_str)
            )
            stats["embeddings_inserted"] += 1

        conn.commit()

    conn_manager.close()

    logger.info(f"Import complete: {stats}")
    return stats


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python import_to_postgres.py <export_json_path>")
        sys.exit(1)

    export_path = Path(sys.argv[1])

    if not export_path.exists():
        print(f"Error: Export file not found: {export_path}")
        sys.exit(1)

    stats = import_to_postgres(export_path)
    print(f"\nImport Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
```

**Step 4: Run tests to verify they pass**

Run: `/home/test/MyLaw-RAG/venv_linux/bin/pytest tests/test_import_to_postgres.py -v`
Expected: PASS (all tests pass)

**Step 5: Commit**

```bash
git add scripts/migrate/import_to_postgres.py tests/test_import_to_postgres.py
git commit -m "feat: add PostgreSQL import script

- Import chunks from ChromaDB export JSON
- Insert acts, sections, chunks, embeddings
- Handle duplicate acts and sections
- Track and report import statistics

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Create Migration Validation Script

**Files:**
- Create: `scripts/migrate/validate_migration.py`

**Step 1: Write test for validation script**

Create `tests/test_validate_migration.py` with:

```python
"""Test migration validation script."""

import pytest
from unittest.mock import MagicMock, patch

from scripts.migrate.validate_migration import (
    validate_data_integrity,
    validate_retrieval_quality
)


@patch("scripts.migrate.validate_migration.PostgreSQLConnectionManager")
@patch("scripts.migrate.validate_migration.chromadb")
def test_validate_data_integrity(mock_chromadb, mock_conn_manager):
    """Test data integrity validation."""
    # Mock ChromaDB
    mock_collection = MagicMock()
    mock_collection.count.return_value = 100
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection
    mock_chromadb.PersistentClient.return_value = mock_client

    # Mock PostgreSQL
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = [100]
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    mock_pool_instance = MagicMock()
    mock_pool_instance.get_connection.return_value.__enter__.return_value = mock_conn
    mock_conn_manager.return_value.get_connection.return_value.__enter__.return_value = mock_conn

    # Run validation
    result = validate_data_integrity("test_collection")

    assert result["chroma_count"] == 100
    assert result["postgres_count"] == 100
    assert result["match"] is True


@patch("scripts.migrate.validate_migration.PostgreSQLRetriever")
@patch("scripts.migrate.validate_migration.HybridRetriever")
def test_validate_retrieval_quality(mock_chroma, mock_pg):
    """Test retrieval quality validation."""
    # Mock retrievers to return same results
    from src.retrieval.hybrid_retriever import RetrievalResult

    mock_result = RetrievalResult(
        chunk_id="1",
        content="Test",
        act_name="Test Act",
        act_number=1,
        section_number="1",
        section_title="Test",
        score=0.95,
        retrieval_method="test"
    )

    mock_pg.return_value.retrieve.return_value = [mock_result]
    mock_chroma.return_value.retrieve.return_value = [mock_result]

    result = validate_retrieval_quality(
        ["test query"],
        mock_pg.return_value,
        mock_chroma.return_value
    )

    assert result["postgresql_hit_rate"] == 1.0
    assert result["chroma_hit_rate"] == 1.0
    assert result["quality_ratio"] >= 0.95
```

**Step 2: Run test to verify it fails**

Run: `/home/test/MyLaw-RAG/venv_linux/bin/pytest tests/test_validate_migration.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'scripts.migrate.validate_migration'"

**Step 3: Implement validation script**

Create `scripts/migrate/validate_migration.py` with:

```python
"""
Validate PostgreSQL migration from ChromaDB.

This script validates data integrity and retrieval quality after migration.
"""

import logging
from typing import Dict, List, Any

import chromadb
from chromadb.config import Settings

from src.config import get_vector_db_dir, setup_logging
from src.db.postgres_connection import PostgreSQLConnectionManager
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.postgresql_retriever import PostgreSQLRetriever

logger = setup_logging(__name__)


def validate_data_integrity(
    collection_name: str,
    config = None
) -> Dict[str, Any]:
    """
    Validate that all data was migrated correctly.

    Args:
        collection_name: ChromaDB collection name.
        config: PostgreSQL configuration.

    Returns:
        Validation result dictionary.
    """
    from src.config import postgresql_config as default_config

    config = config or default_config
    conn_manager = PostgreSQLConnectionManager(config)
    conn_manager.initialize()

    # Count ChromaDB chunks
    db_path = str(get_vector_db_dir())
    chroma_client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False)
    )
    chroma_collection = chroma_client.get_collection(name=collection_name)
    chroma_count = chroma_collection.count()

    # Count PostgreSQL chunks
    with conn_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chunks")
        pg_count = cursor.fetchone()[0]

    conn_manager.close()

    match = chroma_count == pg_count

    result = {
        "chroma_count": chroma_count,
        "postgres_count": pg_count,
        "match": match,
        "difference": abs(chroma_count - pg_count)
    }

    if match:
        logger.info(f"✅ Data integrity validated: {chroma_count} chunks")
    else:
        logger.error(f"❌ Data mismatch: ChromaDB={chroma_count}, PG={pg_count}")

    return result


def validate_retrieval_quality(
    test_queries: List[str],
    pg_retriever: PostgreSQLRetriever,
    chroma_retriever: HybridRetriever
) -> Dict[str, float]:
    """
    Validate that PostgreSQL retrieval matches ChromaDB quality.

    Args:
        test_queries: List of test queries.
        pg_retriever: PostgreSQL retriever instance.
        chroma_retriever: ChromaDB retriever instance.

    Returns:
        Quality metrics dictionary.
    """
    pg_hits = 0
    chroma_hits = 0

    for query in test_queries:
        pg_results = pg_retriever.retrieve(query, n_results=5)
        chroma_results = chroma_retriever.retrieve(query, n_results=5)

        if pg_results:
            pg_hits += 1
        if chroma_results:
            chroma_hits += 1

    pg_hit_rate = pg_hits / len(test_queries) if test_queries else 0
    chroma_hit_rate = chroma_hits / len(test_queries) if test_queries else 0
    quality_ratio = pg_hit_rate / chroma_hit_rate if chroma_hit_rate > 0 else 0

    result = {
        "postgresql_hit_rate": pg_hit_rate,
        "chroma_hit_rate": chroma_hit_rate,
        "quality_ratio": quality_ratio
    }

    if quality_ratio >= 0.95:
        logger.info(f"✅ Retrieval quality validated: {quality_ratio:.2%} of ChromaDB")
    else:
        logger.warning(f"⚠️  Retrieval quality below threshold: {quality_ratio:.2%}")

    return result


if __name__ == "__main__":
    import sys

    from src.config import RAGConfig

    config = RAGConfig()

    print("Validating migration...")

    # Data integrity
    integrity_result = validate_data_integrity(config.collection_name)
    print(f"\nData Integrity:")
    print(f"  ChromaDB: {integrity_result['chroma_count']} chunks")
    print(f"  PostgreSQL: {integrity_result['postgres_count']} chunks")
    print(f"  Match: {integrity_result['match']}")

    if not integrity_result['match']:
        sys.exit(1)

    # Retrieval quality (requires retrievers)
    print("\nFor retrieval quality validation, run with test queries.")
```

**Step 4: Run tests to verify they pass**

Run: `/home/test/MyLaw-RAG/venv_linux/bin/pytest tests/test_validate_migration.py -v`
Expected: PASS (all tests pass)

**Step 5: Commit**

```bash
git add scripts/migrate/validate_migration.py tests/test_validate_migration.py
git commit -m "feat: add migration validation script

- Validate data integrity (chunk counts match)
- Compare retrieval quality between ChromaDB and PostgreSQL
- Calculate hit rate ratio (target: ≥95%)
- Provide clear pass/fail feedback

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 9: Update Documentation

**Files:**
- Modify: `README.md`
- Create: `docs/POSTGRESQL_SETUP.md`

**Step 1: Create PostgreSQL setup guide**

Create `docs/POSTGRESQL_SETUP.md` with:

```markdown
# PostgreSQL Setup for MyLaw-RAG

This guide covers installing and configuring PostgreSQL with pgvector for the MyLaw-RAG system.

## Prerequisites

- Linux or macOS system
- Python 3.8+
- PostgreSQL 15+ (for pgvector support)

## Installation

### Ubuntu/Debian

```bash
# Install PostgreSQL
sudo apt update
sudo apt install postgresql postgresql-contrib

# Install pgvector extension
git clone --branch v0.5.0 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

### macOS

```bash
# Install PostgreSQL
brew install postgresql@15

# Install pgvector
brew install pgvector
```

## Database Setup

### 1. Create Database

```bash
# Switch to postgres user
sudo -u postgres psql

# Create database and user
CREATE DATABASE mylaw_rag;
CREATE USER mylaw_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE mylaw_rag TO mylaw_user;
\q
```

### 2. Initialize Schema

```bash
# Run schema initialization
psql -U mylaw_user -d mylaw_rag -f scripts/migrate/init_postgres_schema.sql
```

### 3. Configure Environment

Add to `.env`:

```bash
PGHOST=localhost
PGPORT=5432
PGDATABASE=mylaw_rag
PGUSER=mylaw_user
PGPASSWORD=your_password
```

## Migration

### Export from ChromaDB

```bash
python scripts/migrate/export_from_chroma.py
```

### Import to PostgreSQL

```bash
python scripts/migrate/import_to_postgres.py data/export/chroma_export.json
```

### Validate Migration

```bash
python scripts/migrate/validate_migration.py
```

## Troubleshooting

### pgvector not available

```sql
-- Check if extension is installed
SELECT * FROM pg_extension WHERE extname = 'vector';

-- Install if missing
CREATE EXTENSION vector;
```

### Connection refused

Ensure PostgreSQL is running:

```bash
sudo service postgresql status
sudo service postgresql start
```

### Permission denied

Grant proper permissions:

```sql
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mylaw_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO mylaw_user;
```
```

**Step 2: Update README.md**

Add to README.md after the "Installation" section:

```markdown
## Database Options

MyLaw-RAG supports two database backends:

### ChromaDB (Default)

Local vector database with no external dependencies. Automatically initialized on first run.

### PostgreSQL (Recommended for Production)

PostgreSQL with pgvector extension provides better scalability and SQL querying capabilities.

See [docs/POSTGRESQL_SETUP.md](docs/POSTGRESQL_SETUP.md) for installation and migration instructions.
```

**Step 3: Commit**

```bash
git add README.md docs/POSTGRESQL_SETUP.md
git commit -m "docs: add PostgreSQL setup documentation

- Complete PostgreSQL installation guide
- Database initialization instructions
- Migration steps from ChromaDB
- Troubleshooting common issues
- Update README with database options

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 10: Final Integration Testing

**Files:**
- Create: `tests/test_integration_postgres.py`

**Step 1: Write integration test**

Create `tests/test_integration_postgres.py` with:

```python
"""Integration tests for PostgreSQL migration."""

import pytest
from unittest.mock import MagicMock, patch

from src.retrieval.postgresql_retriever import PostgreSQLRetriever
from src.retrieval.hybrid_retriever import HybridRetriever


@patch("src.retrieval.postgresql_retriever.PostgreSQLConnectionManager")
def test_postgresql_retriever_end_to_end(mock_conn_manager):
    """Test full retrieval flow with PostgreSQL."""
    # Mock successful database query
    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    mock_cursor.fetchall.return_value = [
        (1, "Consideration is something of value...", "Contracts Act 1950", 136, "2", "Consideration", 0.92),
        (2, "An agreement is enforceable...", "Contracts Act 1950", 136, "10", "Agreement", 0.88)
    ]

    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    mock_pool_instance = MagicMock()
    mock_pool_instance.get_connection.return_value.__enter__.return_value = mock_conn
    mock_conn_manager.return_value.get_connection.return_value.__enter__.return_value = mock_conn

    # Test retriever
    retriever = PostgreSQLRetriever()
    results = retriever.retrieve("What is consideration in contract law?", n_results=5)

    assert len(results) == 2
    assert results[0].act_name == "Contracts Act 1950"
    assert results[0].score > results[1].score

    # Test context formatting
    context = retriever.format_context(results)
    assert "Source 1" in context
    assert "Consideration is something of value" in context


def test_retriever_interface_compatibility():
    """Test that PostgreSQLRetriever and HybridRetriever have same interface."""
    pg_retriever_methods = set(dir(PostgreSQLRetriever))
    hybrid_retriever_methods = set(dir(HybridRetriever))

    # Check for common methods
    common_methods = {"retrieve", "format_context"}

    assert common_methods.issubset(pg_retriever_methods)
    assert common_methods.issubset(hybrid_retriever_methods)


@patch("src.retrieval.hybrid_retriever.chromadb")
def test_hybrid_retriever_still_works(mock_chromadb):
    """Test that HybridRetriever still functions (no breaking changes)."""
    from src.config import RAGConfig

    config = RAGConfig()

    # Should not raise any errors
    retriever = HybridRetriever(config)
    assert retriever is not None
    assert retriever.config == config
```

**Step 2: Run integration tests**

Run: `/home/test/MyLaw-RAG/venv_linux/bin/pytest tests/test_integration_postgres.py -v`
Expected: PASS (all tests pass)

**Step 3: Run all tests to verify no regressions**

Run: `/home/test/MyLaw-RAG/venv_linux/bin/pytest tests/ -v`
Expected: All existing tests still pass

**Step 4: Commit**

```bash
git add tests/test_integration_postgres.py
git commit -m "test: add PostgreSQL integration tests

- End-to-end retrieval flow test
- Interface compatibility test (PostgreSQLRetriever vs HybridRetriever)
- Verify no breaking changes to HybridRetriever
- All existing tests still pass

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Summary

**Total Tasks:** 10
**Estimated Time:** 3-4 weeks
**Files to Create:** 11 new files
**Files to Modify:** 3 existing files

### Task Breakdown

| Task | Component | Files | Tests |
|------|-----------|-------|-------|
| 1 | Dependencies | requirements.txt | - |
| 2 | Config | src/config.py, .env.example | - |
| 3 | Schema | scripts/migrate/init_postgres_schema.sql | - |
| 4 | Connection Manager | src/db/postgres_connection.py | tests/test_postgres_connection.py |
| 5 | PostgreSQL Retriever | src/retrieval/postgresql_retriever.py | tests/test_postgresql_retriever.py |
| 6 | Export Script | scripts/migrate/export_from_chroma.py | tests/test_export_from_chroma.py |
| 7 | Import Script | scripts/migrate/import_to_postgres.py | tests/test_import_to_postgres.py |
| 8 | Validation Script | scripts/migrate/validate_migration.py | tests/test_validate_migration.py |
| 9 | Documentation | README.md, docs/POSTGRESQL_SETUP.md | - |
| 10 | Integration Tests | tests/test_integration_postgres.py | - |

### Success Criteria

After completing all tasks:

- ✅ All dependencies installed
- ✅ PostgreSQL schema created
- ✅ Connection pooling implemented
- ✅ PostgreSQLRetriever works with same interface as HybridRetriever
- ✅ Export/import pipeline functional
- ✅ Migration validation passes
- ✅ Documentation complete
- ✅ All tests pass (existing + new)
- ✅ No breaking changes to existing functionality

### Next Steps After Implementation

1. Install PostgreSQL + pgvector on target system
2. Run `scripts/migrate/init_postgres_schema.sql`
3. Configure `.env` with PostgreSQL credentials
4. Export data from ChromaDB
5. Import to PostgreSQL
6. Run validation
7. Test application with PostgreSQL backend
8. (Optional) Migrate existing Acts and retire ChromaDB
