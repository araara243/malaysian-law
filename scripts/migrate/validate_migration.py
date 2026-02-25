"""
Validate PostgreSQL migration from ChromaDB.

This script validates the migration by checking:
1. Data integrity: Compare chunk counts
2. Retrieval quality: Compare hit rates
3. Embeddings: Verify embeddings imported
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import PostgreSQLConfig, setup_logging
from db.postgres_connection import PostgreSQLConnectionManager
from retrieval.postgresql_retriever import PostgreSQLRetriever

logger = setup_logging(__name__)


def validate_data_integrity(
    export_path: Path,
    config: PostgreSQLConfig
) -> Dict[str, Any]:
    """
    Validate data integrity by comparing chunk counts.

    Args:
        export_path: Path to ChromaDB export JSON.
        config: PostgreSQL configuration.

    Returns:
        Dictionary with validation results.
    """
    # Load export data
    if isinstance(export_path, str):
        export_path = Path(export_path)

    with open(export_path, 'r') as f:
        export_data = json.load(f)

    chromadb_count = len(export_data.get("chunks", []))

    # Query PostgreSQL for chunk count
    conn_manager = PostgreSQLConnectionManager(config)
    conn_manager.initialize()

    with conn_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chunks")
        postgres_count = cursor.fetchone()[0]

    conn_manager.close()

    passed = chromadb_count == postgres_count

    return {
        "passed": passed,
        "chromadb_count": chromadb_count,
        "postgres_count": postgres_count
    }


def validate_retrieval_quality(
    export_path: Path,
    config: PostgreSQLConfig,
    golden_questions: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Validate retrieval quality by comparing hit rates.

    Args:
        export_path: Path to ChromaDB export JSON.
        config: PostgreSQL configuration.
        golden_questions: List of test questions with expected answers.

    Returns:
        Dictionary with validation results.
    """
    # Initialize PostgreSQL retriever
    retriever = PostgreSQLRetriever(config)

    # Run queries
    hits = 0
    total = len(golden_questions)

    for q in golden_questions:
        results = retriever.retrieve(q["question"], n_results=5)

        # Check if expected act/section is in results
        for r in results:
            act_match = q["expected_act"] in r.act_name
            section_match = q["expected_section"].lower().replace("section ", "") in str(r.section_number).lower()

            if act_match and section_match:
                hits += 1
                break

    hit_rate = hits / total if total > 0 else 0.0
    passed = hit_rate >= 0.95  # Target: 95% hit rate

    retriever.close()

    return {
        "passed": passed,
        "postgres_hit_rate": hit_rate,
        "hits": hits,
        "total": total
    }


def validate_embeddings(
    export_path: Path,
    config: PostgreSQLConfig
) -> Dict[str, Any]:
    """
    Validate embeddings were imported correctly.

    Args:
        export_path: Path to ChromaDB export JSON.
        config: PostgreSQL configuration.

    Returns:
        Dictionary with validation results.
    """
    # Load export data
    if isinstance(export_path, str):
        export_path = Path(export_path)

    with open(export_path, 'r') as f:
        export_data = json.load(f)

    # Count chunks with embeddings in export
    export_count = sum(1 for chunk in export_data.get("chunks", []) if "embedding" in chunk)

    # Query PostgreSQL for embedding count
    conn_manager = PostgreSQLConnectionManager(config)
    conn_manager.initialize()

    with conn_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM embeddings")
        postgres_count = cursor.fetchone()[0]

    conn_manager.close()

    passed = export_count == postgres_count

    return {
        "passed": passed,
        "export_count": export_count,
        "postgres_count": postgres_count
    }


def main():
    """Run all validations and print results."""
    if len(sys.argv) < 2:
        print("Usage: python validate_migration.py <export_json_path>")
        sys.exit(1)

    from config import postgresql_config

    export_path = Path(sys.argv[1])

    if not export_path.exists():
        print(f"Error: Export file not found: {export_path}")
        sys.exit(1)

    print("=" * 60)
    print("PostgreSQL Migration Validation")
    print("=" * 60)

    # Load golden dataset
    golden_dataset_path = PROJECT_ROOT / "tests" / "golden_dataset.json"
    with open(golden_dataset_path, 'r') as f:
        golden_data = json.load(f)

    golden_questions = golden_data["questions"]

    # Run validations
    print("\n1. Validating data integrity...")
    integrity_result = validate_data_integrity(export_path, postgresql_config)
    status = "PASS" if integrity_result["passed"] else "FAIL"
    print(f"   Status: {status}")
    print(f"   ChromaDB: {integrity_result['chromadb_count']} chunks")
    print(f"   PostgreSQL: {integrity_result['postgres_count']} chunks")

    print("\n2. Validating embeddings...")
    embeddings_result = validate_embeddings(export_path, postgresql_config)
    status = "PASS" if embeddings_result["passed"] else "FAIL"
    print(f"   Status: {status}")
    print(f"   Export: {embeddings_result['export_count']} embeddings")
    print(f"   PostgreSQL: {embeddings_result['postgres_count']} embeddings")

    print("\n3. Validating retrieval quality...")
    retrieval_result = validate_retrieval_quality(
        export_path,
        postgresql_config,
        golden_questions
    )
    status = "PASS" if retrieval_result["passed"] else "FAIL"
    print(f"   Status: {status}")
    print(f"   Hit Rate: {retrieval_result['postgres_hit_rate']:.1%}")
    print(f"   Hits: {retrieval_result['hits']}/{retrieval_result['total']}")

    # Overall result
    all_passed = (
        integrity_result["passed"] and
        embeddings_result["passed"] and
        retrieval_result["passed"]
    )

    print("\n" + "=" * 60)
    if all_passed:
        print("All validations PASSED!")
        sys.exit(0)
    else:
        print("Some validations FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
