"""
Import exported ChromaDB data into PostgreSQL.

This script imports JSON data exported from ChromaDB into the
normalized PostgreSQL schema.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import PostgreSQLConfig, setup_logging, postgresql_config
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
    config = config or postgresql_config
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
                    ON CONFLICT (act_number) DO NOTHING
                    RETURNING id
                    """,
                    (
                        chunk_data["act_number"],
                        chunk_data["act_name"],
                        chunk_data.get("act_year", 0),
                        chunk_data.get("category", "other")
                    )
                )
                result = cursor.fetchone()
                if result:
                    # New act inserted
                    act_id = result[0]
                    stats["acts_inserted"] += 1
                else:
                    # Act already exists, fetch its id
                    cursor.execute(
                        "SELECT id FROM acts WHERE act_number = %s",
                        (chunk_data["act_number"],)
                    )
                    act_id = cursor.fetchone()[0]
                acts_map[act_key] = act_id
            else:
                act_id = acts_map[act_key]

            # Get or create section
            section_number = chunk_data.get("section_number", "")
            section_key = (act_id, section_number)
            if section_key not in sections_map:
                cursor.execute(
                    """
                    INSERT INTO sections (act_id, section_number, section_title, part, section_order)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (act_id, section_number, section_order) DO NOTHING
                    RETURNING id
                    """,
                    (
                        act_id,
                        section_number,
                        chunk_data.get("section_title", ""),
                        chunk_data.get("part", ""),
                        0  # section_order default value
                    )
                )
                result = cursor.fetchone()
                if result:
                    # New section inserted
                    section_id = result[0]
                    stats["sections_inserted"] += 1
                else:
                    # Section already exists, fetch its id
                    cursor.execute(
                        "SELECT id FROM sections WHERE act_id = %s AND section_number = %s",
                        (act_id, section_number)
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
