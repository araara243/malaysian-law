"""Test migration validation script."""

import os
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import json
from unittest.mock import MagicMock, patch


@pytest.fixture
def sample_export_data(tmp_path):
    """Create sample export data for testing."""
    export_data = {
        "collection_name": "malaysian_legal_acts",
        "total_chunks": 3,
        "chunks": [
            {
                "chunk_id": "chunk1",
                "content": "Content 1",
                "embedding": [0.1] * 384,
                "act_name": "Contracts Act 1950",
                "act_number": 136,
                "section_number": "2"
            },
            {
                "chunk_id": "chunk2",
                "content": "Content 2",
                "embedding": [0.2] * 384,
                "act_name": "Contracts Act 1950",
                "act_number": 136,
                "section_number": "10"
            },
            {
                "chunk_id": "chunk3",
                "content": "Content 3",
                "embedding": [0.3] * 384,
                "act_name": "Specific Relief Act 1951",
                "act_number": 137,
                "section_number": "11"
            }
        ]
    }

    export_path = tmp_path / "test_export.json"
    with open(export_path, 'w') as f:
        json.dump(export_data, f)

    return export_path


@patch("scripts.migrate.validate_migration.PostgreSQLConnectionManager")
def test_validate_data_integrity_pass(mock_conn_manager, sample_export_data):
    """Test data integrity validation when counts match."""
    from config import postgresql_config
    from scripts.migrate.validate_migration import validate_data_integrity

    # Mock database response
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = [3]  # Same count as export

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    mock_conn_manager.return_value.initialize = MagicMock()
    mock_conn_manager.return_value.get_connection.return_value.__enter__.return_value = mock_conn
    mock_conn_manager.return_value.close = MagicMock()

    result = validate_data_integrity(sample_export_data, postgresql_config)

    assert result["passed"] is True
    assert result["chromadb_count"] == 3
    assert result["postgres_count"] == 3


@patch("scripts.migrate.validate_migration.PostgreSQLConnectionManager")
def test_validate_data_integrity_fail(mock_conn_manager, tmp_path):
    """Test data integrity validation when counts don't match."""
    from config import postgresql_config
    from scripts.migrate.validate_migration import validate_data_integrity

    # Create export with 3 chunks
    export_data = {
        "total_chunks": 3,
        "chunks": [
            {"chunk_id": "chunk1", "content": "Content 1", "embedding": [0.1] * 384},
            {"chunk_id": "chunk2", "content": "Content 2", "embedding": [0.2] * 384},
            {"chunk_id": "chunk3", "content": "Content 3", "embedding": [0.3] * 384}
        ]
    }

    export_path = tmp_path / "test_export_mismatch.json"
    with open(export_path, 'w') as f:
        json.dump(export_data, f)

    # Mock database response with different count
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = [2]  # Different count

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    mock_conn_manager.return_value.initialize = MagicMock()
    mock_conn_manager.return_value.get_connection.return_value.__enter__.return_value = mock_conn
    mock_conn_manager.return_value.close = MagicMock()

    result = validate_data_integrity(export_path, postgresql_config)

    assert result["passed"] is False
    assert result["chromadb_count"] == 3
    assert result["postgres_count"] == 2


@patch("scripts.migrate.validate_migration.PostgreSQLConnectionManager")
@patch("scripts.migrate.validate_migration.PostgreSQLRetriever")
def test_validate_retrieval_quality(mock_retriever_class, mock_conn_manager, sample_export_data):
    """Test retrieval quality validation."""
    from config import postgresql_config
    from scripts.migrate.validate_migration import validate_retrieval_quality

    # Mock PostgreSQL retriever results
    mock_result = MagicMock()
    mock_result.act_name = "Contracts Act 1950"
    mock_result.section_number = "2"

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [mock_result]
    mock_retriever.close = MagicMock()
    mock_retriever_class.return_value = mock_retriever

    mock_conn_manager.return_value.initialize = MagicMock()
    mock_conn_manager.return_value.close = MagicMock()

    golden_questions = [
        {
            "question": "What is consideration?",
            "expected_act": "Contracts Act 1950",
            "expected_section": "Section 2"
        }
    ]

    result = validate_retrieval_quality(
        sample_export_data,
        postgresql_config,
        golden_questions
    )

    assert "passed" in result
    assert "postgres_hit_rate" in result
    assert isinstance(result["postgres_hit_rate"], float)


@patch("scripts.migrate.validate_migration.PostgreSQLConnectionManager")
def test_validate_embeddings_pass(mock_conn_manager, sample_export_data):
    """Test embedding validation when counts match."""
    from config import postgresql_config
    from scripts.migrate.validate_migration import validate_embeddings

    # Mock database response
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = [3]  # Same count as export

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    mock_conn_manager.return_value.initialize = MagicMock()
    mock_conn_manager.return_value.get_connection.return_value.__enter__.return_value = mock_conn
    mock_conn_manager.return_value.close = MagicMock()

    result = validate_embeddings(sample_export_data, postgresql_config)

    assert result["passed"] is True
    assert result["export_count"] == 3
    assert result["postgres_count"] == 3


@patch("scripts.migrate.validate_migration.PostgreSQLConnectionManager")
def test_validate_embeddings_fail(mock_conn_manager, tmp_path):
    """Test embedding validation when counts don't match."""
    from config import postgresql_config
    from scripts.migrate.validate_migration import validate_embeddings

    # Create export with 3 embeddings
    export_data = {
        "chunks": [
            {"chunk_id": "chunk1", "embedding": [0.1] * 384},
            {"chunk_id": "chunk2", "embedding": [0.2] * 384},
            {"chunk_id": "chunk3", "embedding": [0.3] * 384}
        ]
    }

    export_path = tmp_path / "test_export_embeddings.json"
    with open(export_path, 'w') as f:
        json.dump(export_data, f)

    # Mock database response with different count
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = [2]

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    mock_conn_manager.return_value.initialize = MagicMock()
    mock_conn_manager.return_value.get_connection.return_value.__enter__.return_value = mock_conn
    mock_conn_manager.return_value.close = MagicMock()

    result = validate_embeddings(export_path, postgresql_config)

    assert result["passed"] is False
    assert result["export_count"] == 3
    assert result["postgres_count"] == 2
