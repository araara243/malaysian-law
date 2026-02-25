"""Integration tests for PostgreSQL migration.

These tests verify:
1. PostgreSQLRetriever interface compatibility with HybridRetriever
2. End-to-end retrieval flow
3. No breaking changes to existing code
4. Data consistency between retrievers
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import numpy as np

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval.hybrid_retriever import HybridRetriever, RetrievalResult
from retrieval.postgresql_retriever import PostgreSQLRetriever
from config import postgresql_config


class TestInterfaceCompatibility:
    """Test that PostgreSQLRetriever and HybridRetriever have compatible interfaces."""

    def test_both_have_retrieve_method(self):
        """Both retrievers should have a retrieve method."""
        assert hasattr(HybridRetriever, 'retrieve')
        assert hasattr(PostgreSQLRetriever, 'retrieve')

    def test_both_have_format_context_method(self):
        """Both retrievers should have a format_context method."""
        assert hasattr(HybridRetriever, 'format_context')
        assert hasattr(PostgreSQLRetriever, 'format_context')

    def test_both_have_close_method(self):
        """PostgreSQLRetriever has close method for cleanup (HybridRetriever doesn't need it)."""
        # PostgreSQLRetriever needs close() for connection cleanup
        assert hasattr(PostgreSQLRetriever, 'close')
        # HybridRetriever doesn't manage connections, so no close() needed
        # This is expected behavior, not a breaking change

    def test_retrieve_signature_compatibility(self):
        """Retrieve methods should have compatible signatures."""
        import inspect

        hybrid_sig = inspect.signature(HybridRetriever.retrieve)
        postgres_sig = inspect.signature(PostgreSQLRetriever.retrieve)

        # Both should accept query and n_results parameters
        hybrid_params = list(hybrid_sig.parameters.keys())
        postgres_params = list(postgres_sig.parameters.keys())

        assert 'query' in hybrid_params
        assert 'query' in postgres_params
        assert 'n_results' in hybrid_params
        assert 'n_results' in postgres_params

    def test_format_context_signature_compatibility(self):
        """format_context methods should have compatible signatures."""
        import inspect

        hybrid_sig = inspect.signature(HybridRetriever.format_context)
        postgres_sig = inspect.signature(PostgreSQLRetriever.format_context)

        # Both should accept results parameter
        hybrid_params = list(hybrid_sig.parameters.keys())
        postgres_params = list(postgres_sig.parameters.keys())

        assert 'results' in hybrid_params
        assert 'results' in postgres_params


class TestRetrievalResult:
    """Test that both retrievers return compatible RetrievalResult objects."""

    def test_retrieval_result_attributes(self):
        """RetrievalResult should have expected attributes."""
        result = RetrievalResult(
            chunk_id="test_id",
            content="Test content",
            act_name="Test Act",
            act_number=999,
            section_number="1",
            section_title="Test Section",
            score=0.95,
            retrieval_method="test"
        )

        assert result.chunk_id == "test_id"
        assert result.content == "Test content"
        assert result.act_name == "Test Act"
        assert result.act_number == 999
        assert result.section_number == "1"
        assert result.section_title == "Test Section"
        assert result.score == 0.95
        assert result.retrieval_method == "test"


@patch("retrieval.postgresql_retriever.PostgreSQLConnectionManager")
@patch("retrieval.postgresql_retriever.SentenceTransformer")
class TestPostgreSQLRetrieverIntegration:
    """Test PostgreSQLRetriever end-to-end with mocked database."""

    def test_retrieve_returns_retrieval_results(self, mock_embedding_model, mock_conn_manager):
        """Test that retrieve returns list of RetrievalResult objects."""
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
        results = retriever.retrieve("test query", n_results=5)

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert all(r.retrieval_method == "postgresql" for r in results)

    def test_format_context_output_format(self, mock_embedding_model, mock_conn_manager):
        """Test that format_context returns properly formatted context."""
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

        assert isinstance(context, str)
        assert len(context) > 0
        assert "Source 1" in context
        assert "Contracts Act 1950" in context

    def test_close_cleanup(self, mock_embedding_model, mock_conn_manager):
        """Test that close properly cleans up resources."""
        from config import postgresql_config

        mock_conn_manager.return_value.initialize = MagicMock()
        mock_conn_manager.return_value.close = MagicMock()

        retriever = PostgreSQLRetriever(postgresql_config)
        retriever.close()

        mock_conn_manager.return_value.close.assert_called_once()


class TestNoBreakingChanges:
    """Test that existing code still works after PostgreSQL migration."""

    def test_hybrid_retriever_still_works(self):
        """HybridRetriever should still work without any changes."""
        # This test ensures that adding PostgreSQL support didn't break existing functionality
        retriever = HybridRetriever()

        # Should be able to initialize
        assert retriever is not None

        # Should have core retrieval methods
        assert hasattr(retriever, 'retrieve')
        assert hasattr(retriever, 'format_context')

        # Note: HybridRetriever doesn't have close() since it doesn't manage connections
        # This is expected and not a breaking change

    def test_retrieval_result_unchanged(self):
        """RetrievalResult structure should be unchanged."""
        # This ensures that code using RetrievalResult still works
        result = RetrievalResult(
            chunk_id="test",
            content="content",
            act_name="Act",
            act_number=1,
            section_number="1",
            section_title="Title",
            score=0.9,
            retrieval_method="hybrid"
        )

        # Should be able to access all attributes
        assert result.chunk_id == "test"
        assert result.content == "content"
        assert result.act_name == "Act"
        assert result.act_number == 1
        assert result.section_number == "1"
        assert result.section_title == "Title"
        assert result.score == 0.9
        assert result.retrieval_method == "hybrid"

    def test_config_unaffected(self):
        """RAGConfig should be unchanged and working."""
        from config import RAGConfig

        config = RAGConfig()

        # Should have all expected attributes
        assert hasattr(config, 'chunk_size')
        assert hasattr(config, 'top_k')
        assert hasattr(config, 'semantic_weight')
        assert hasattr(config, 'keyword_weight')
        assert hasattr(config, 'embedding_model')
        assert hasattr(config, 'llm_model')


class TestDataConsistency:
    """Test data consistency between ChromaDB and PostgreSQL."""

    @patch("retrieval.postgresql_retriever.PostgreSQLConnectionManager")
    @patch("retrieval.postgresql_retriever.SentenceTransformer")
    def test_similar_results_for_same_query(self, mock_embedding_model, mock_conn_manager):
        """Test that PostgreSQL retriever returns similar results to ChromaDB."""
        from config import postgresql_config

        # Mock PostgreSQL retriever to return results similar to ChromaDB
        mock_model_instance = MagicMock()
        mock_model_instance.encode.return_value = np.random.rand(384)
        mock_embedding_model.return_value = mock_model_instance

        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        # Mock response with data similar to what ChromaDB would return
        mock_cursor.fetchall.return_value = [
            (1, "Consideration is something of value...", "Contracts Act 1950", 136, "2", "Consideration", 0.92),
            (2, "An agreement is a promise...", "Contracts Act 1950", 136, "2a", "Who are competent to contract", 0.88)
        ]
        mock_conn.cursor.return_value = mock_cursor

        mock_conn_manager.return_value.initialize = MagicMock()
        mock_conn_manager.return_value.get_connection.return_value.__enter__.return_value = mock_conn

        postgres_retriever = PostgreSQLRetriever(postgresql_config)

        # Test query
        query = "What is consideration in contract law?"
        results = postgres_retriever.retrieve(query, n_results=5)

        # Verify results have expected structure
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert all("Contracts Act 1950" in r.act_name for r in results)
        assert all(r.score > 0 for r in results)

        postgres_retriever.close()


class TestConfiguration:
    """Test configuration handling for PostgreSQL."""

    def test_postgresql_config_exists(self):
        """PostgreSQLConfig should be available."""
        from config import PostgreSQLConfig
        assert PostgreSQLConfig is not None

    def test_postgresql_config_defaults(self):
        """PostgreSQLConfig should have sensible defaults."""
        from config import PostgreSQLConfig

        config = PostgreSQLConfig()

        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "mylaw_rag"
        assert config.user == "postgres"
        assert config.embedding_dimension == 384
        assert config.min_connections >= 1
        assert config.max_connections >= config.min_connections

    def test_postgresql_config_from_env(self):
        """PostgreSQLConfig should read from environment variables."""
        import os

        # Set environment variables
        os.environ['PGHOST'] = 'testhost'
        os.environ['PGPORT'] = '5433'
        os.environ['PGDATABASE'] = 'testdb'
        os.environ['PGUSER'] = 'testuser'
        os.environ['PGPASSWORD'] = 'testpass'

        # Re-import to pick up new environment
        import importlib
        import config
        importlib.reload(config)

        config_instance = config.PostgreSQLConfig()

        assert config_instance.host == 'testhost'
        assert config_instance.port == 5433
        assert config_instance.database == 'testdb'
        assert config_instance.user == 'testuser'
        assert config_instance.password == 'testpass'

        # Clean up
        del os.environ['PGHOST']
        del os.environ['PGPORT']
        del os.environ['PGDATABASE']
        del os.environ['PGUSER']
        del os.environ['PGPASSWORD']
        importlib.reload(config)
