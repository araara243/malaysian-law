"""Test PostgreSQL retriever."""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import numpy as np

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval.postgresql_retriever import PostgreSQLRetriever
from retrieval.hybrid_retriever import RetrievalResult


@patch("retrieval.postgresql_retriever.PostgreSQLConnectionManager")
@patch("retrieval.postgresql_retriever.SentenceTransformer")
def test_postgresql_retriever_initialization(mock_embedding_model, mock_conn_manager):
    """Test retriever initializes with config."""
    from config import postgresql_config

    mock_conn_manager.return_value.initialize = MagicMock()
    retriever = PostgreSQLRetriever(postgresql_config)

    assert retriever.config == postgresql_config
    assert retriever._conn_manager is not None


@patch("retrieval.postgresql_retriever.PostgreSQLConnectionManager")
@patch("retrieval.postgresql_retriever.SentenceTransformer")
def test_retrieve_returns_results(mock_embedding_model, mock_conn_manager):
    """Test retrieve returns RetrievalResult objects."""
    from config import postgresql_config

    # Mock embedding model
    mock_model_instance = MagicMock()
    mock_model_instance.encode.return_value = np.random.rand(384)
    mock_embedding_model.return_value = mock_model_instance

    # Mock database response
    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    mock_cursor.fetchall.return_value = [
        (1, "test content", "Contracts Act 1950", 136, "2", "Consideration", 0.95)
    ]
    mock_conn.cursor.return_value = mock_cursor

    mock_conn_manager.return_value.initialize = MagicMock()
    mock_conn_manager.return_value.get_connection.return_value.__enter__.return_value = mock_conn

    retriever = PostgreSQLRetriever(postgresql_config)
    results = retriever.retrieve("What is consideration?")

    assert len(results) == 1
    assert isinstance(results[0], RetrievalResult)
    assert results[0].act_name == "Contracts Act 1950"
    assert results[0].section_number == "2"


@patch("retrieval.postgresql_retriever.PostgreSQLConnectionManager")
@patch("retrieval.postgresql_retriever.SentenceTransformer")
def test_format_context(mock_embedding_model, mock_conn_manager):
    """Test context formatting."""
    from config import postgresql_config

    mock_conn_manager.return_value.initialize = MagicMock()

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


@patch("retrieval.postgresql_retriever.PostgreSQLConnectionManager")
@patch("retrieval.postgresql_retriever.SentenceTransformer")
def test_embed_query(mock_embedding_model, mock_conn_manager):
    """Test query embedding generation."""
    from config import postgresql_config

    # Mock embedding model
    mock_model_instance = MagicMock()
    test_embedding = np.random.rand(384)
    mock_model_instance.encode.return_value = test_embedding
    mock_embedding_model.return_value = mock_model_instance

    mock_conn_manager.return_value.initialize = MagicMock()

    retriever = PostgreSQLRetriever(postgresql_config)
    result = retriever._embed_query("test query")

    mock_model_instance.encode.assert_called_once_with("test query", normalize_embeddings=True)
    assert np.array_equal(result, test_embedding)


@patch("retrieval.postgresql_retriever.PostgreSQLConnectionManager")
@patch("retrieval.postgresql_retriever.SentenceTransformer")
def test_close(mock_embedding_model, mock_conn_manager):
    """Test closing database connections."""
    from config import postgresql_config

    mock_conn_manager.return_value.initialize = MagicMock()

    retriever = PostgreSQLRetriever(postgresql_config)
    retriever.close()

    mock_conn_manager.return_value.close.assert_called_once()
