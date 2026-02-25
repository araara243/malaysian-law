"""Test PostgreSQL import script."""

import os
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import json
from unittest.mock import MagicMock, patch, call

from scripts.migrate.import_to_postgres import import_to_postgres


@patch("scripts.migrate.import_to_postgres.PostgreSQLConnectionManager")
@patch("scripts.migrate.import_to_postgres.postgresql_config")
def test_import_inserts_acts(mock_config, mock_conn_manager, tmp_path):
    """Test that import inserts acts correctly."""
    # Mock database connection
    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    # Configure cursor context manager and fetchone
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.fetchone.return_value = [1]

    # Configure connection manager
    mock_conn_manager.return_value.initialize = MagicMock()
    mock_conn_manager.return_value.get_connection.return_value.__enter__.return_value = mock_conn
    mock_conn_manager.return_value.close = MagicMock()

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
                "section_title": "Consideration",
                "part": "",
                "subsection": "",
                "token_count": 100,
                "keywords": [],
                "start_position": 0
            }
        ]
    }

    export_path = tmp_path / "test_export.json"
    with open(export_path, 'w') as f:
        json.dump(export_data, f)

    # Run import
    result = import_to_postgres(export_path)

    # Verify initialization
    mock_conn_manager.return_value.initialize.assert_called_once()

    # Verify statistics
    assert result["acts_inserted"] == 1
    assert result["chunks_inserted"] == 1
    assert result["embeddings_inserted"] == 1

    # Verify close was called
    mock_conn_manager.return_value.close.assert_called_once()


@patch("scripts.migrate.import_to_postgres.PostgreSQLConnectionManager")
@patch("scripts.migrate.import_to_postgres.postgresql_config")
def test_import_handles_embedded_acts(mock_config, mock_conn_manager, tmp_path):
    """Test that import deduplicates acts by act_number."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    # Configure cursor context manager
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    # Configure connection manager
    mock_conn_manager.return_value.initialize = MagicMock()
    mock_conn_manager.return_value.get_connection.return_value.__enter__.return_value = mock_conn
    mock_conn_manager.return_value.close = MagicMock()

    # Mock fetchone to return sequential IDs
    fetchone_results = [
        [1],  # First act
        [1],  # Section for first chunk
        [2],  # Chunk 1
        None,  # Second act (duplicate, ON CONFLICT returns nothing)
        [3],  # Chunk 2
    ]
    mock_cursor.fetchone.side_effect = fetchone_results

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
                "section_number": "2",
                "section_title": "Consideration",
                "part": "",
                "subsection": "",
                "token_count": 100,
                "keywords": [],
                "start_position": 0
            },
            {
                "act_name": "Contracts Act 1950",
                "act_number": 136,
                "act_year": 1950,
                "category": "commercial",
                "content": "Content 2",
                "embedding": [0.2] * 384,
                "section_number": "10",
                "section_title": "Agreements",
                "part": "",
                "subsection": "",
                "token_count": 100,
                "keywords": [],
                "start_position": 100
            }
        ]
    }

    export_path = tmp_path / "test_export_duplicate.json"
    with open(export_path, 'w') as f:
        json.dump(export_data, f)

    result = import_to_postgres(export_path)

    # Should only insert one act (deduplicated)
    assert result["acts_inserted"] == 1
    assert result["chunks_inserted"] == 2
    assert result["embeddings_inserted"] == 2


@patch("scripts.migrate.import_to_postgres.PostgreSQLConnectionManager")
@patch("scripts.migrate.import_to_postgres.postgresql_config")
def test_import_handles_missing_metadata(mock_config, mock_conn_manager, tmp_path):
    """Test that import handles missing optional metadata fields."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    # Configure cursor context manager
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.fetchone.return_value = [1]

    # Configure connection manager
    mock_conn_manager.return_value.initialize = MagicMock()
    mock_conn_manager.return_value.get_connection.return_value.__enter__.return_value = mock_conn
    mock_conn_manager.return_value.close = MagicMock()

    # Create export with minimal metadata
    export_data = {
        "chunks": [
            {
                "act_name": "Test Act",
                "act_number": 999,
                "content": "Test content",
                "embedding": [0.1] * 384
            }
        ]
    }

    export_path = tmp_path / "test_export_minimal.json"
    with open(export_path, 'w') as f:
        json.dump(export_data, f)

    result = import_to_postgres(export_path)

    # Verify defaults were applied
    assert result["acts_inserted"] == 1
    assert result["chunks_inserted"] == 1
