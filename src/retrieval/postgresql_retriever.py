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

from config import PostgreSQLConfig, setup_logging
from db.postgres_connection import PostgreSQLConnectionManager
from retrieval.hybrid_retriever import RetrievalResult

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
        from config import postgresql_config as default_config

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
