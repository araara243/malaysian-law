"""Test ChromaDB export script."""

import os
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import json
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


@patch("scripts.migrate.export_from_chroma.chromadb")
def test_export_handles_empty_collection(mock_chromadb):
    """Test that export handles empty collections gracefully."""
    mock_collection = MagicMock()
    mock_collection.get.return_value = {
        "ids": [],
        "documents": [],
        "metadatas": [],
        "embeddings": []
    }
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection
    mock_chromadb.PersistentClient.return_value = mock_client

    export_path = Path("/tmp/test_export_empty.json")

    result = export_from_chroma(
        collection_name="empty_collection",
        output_path=export_path
    )

    # Should return empty chunks list
    assert result["chunks"] == []
    assert result["total_chunks"] == 0

    # File should still be created
    assert export_path.exists()

    export_path.unlink()
