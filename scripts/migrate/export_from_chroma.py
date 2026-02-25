"""
Export ChromaDB collection to JSON for PostgreSQL migration.

This script exports all chunks, embeddings, and metadata from ChromaDB
to a JSON file for importing into PostgreSQL.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import chromadb
from chromadb.config import Settings

from config import get_vector_db_dir, setup_logging

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

    # Build export structure
    chunks = []

    if not all_data or not all_data["ids"]:
        logger.warning(f"Collection {collection_name} is empty or not found.")
    else:
        logger.info(f"Exporting {len(all_data['ids'])} chunks from ChromaDB")

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
    from config import RAGConfig

    config = RAGConfig()
    output_path = Path("data/export/chroma_export.json")

    export_from_chroma(
        collection_name=config.collection_name,
        output_path=output_path
    )
