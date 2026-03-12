"""
Hybrid Retriever for Malaysian Legal RAG

This module implements a hybrid search combining:
1. Semantic Search: Vector similarity using ChromaDB embeddings
2. Keyword Search: BM25-based exact term matching

Hybrid search is critical for legal documents because:
- Legal terms like "consideration" have specific meanings
- Semantic search alone may miss exact terminology
- BM25 provides precision; vectors provide recall

The retriever uses Reciprocal Rank Fusion (RRF) to combine results.

NEW: Two-tiered approach for fixing adjacent sibling clustering:
- Tier 1: Metadata filter for explicit section queries
- Tier 2: Post-reranker title boost for conceptual queries
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any

from rank_bm25 import BM25Okapi

from config import (
    RAGConfig,
    get_vector_db_dir,
    setup_logging
)

# Configure logging
logger = setup_logging(__name__)


# Import reranker (lazy load to avoid import errors)
_reranker = None


def get_reranker():
    """Get or create reranker singleton (LLM or Cross-Encoder based on config)."""
    global _reranker
    if _reranker is None:
        try:
            config = RAGConfig()
            if getattr(config, "use_llm_reranker", False):
                from .reranker import LLMReranker
                _reranker = LLMReranker(config=config)
                logger.info("LLM reranker initialized")
            else:
                from .reranker import CrossEncoderReranker
                _reranker = CrossEncoderReranker(
                    model_name=config.reranker_model,
                    device=config.reranker_device
                )
                logger.info("Cross-encoder reranker initialized")
        except ImportError:
            logger.warning("Reranker not available (dependencies not installed)")
            _reranker = None
        except Exception as e:
            logger.warning(f"Failed to initialize reranker: {e}")
            _reranker = None
    return _reranker


@dataclass
class RetrievalResult:
    """A single retrieval result with metadata."""
    chunk_id: str
    content: str
    act_name: str
    act_number: int
    section_number: str
    section_title: str
    score: float
    retrieval_method: str  # "semantic", "keyword", "hybrid"


class HybridRetriever:
    """
    Hybrid retriever combining semantic and keyword search.

    Uses ChromaDB for semantic search and BM25 for keyword search,
    with Reciprocal Rank Fusion (RRF) to combine results.
    """

    def __init__(
        self,
        config: Optional[RAGConfig] = None
    ):
        """
        Initialize hybrid retriever.

        Args:
            config: Optional RAGConfig object. If None, uses defaults.
        """
        self.config = config or RAGConfig()

        self.collection_name = self.config.collection_name
        self.semantic_weight = self.config.semantic_weight
        self.keyword_weight = self.config.keyword_weight
        self.rrf_k = self.config.rrf_k

        # Initialize components
        self._collection: Any = None
        self._bm25: Optional[BM25Okapi] = None
        self._documents: List[str] = []
        self._doc_ids: List[str] = []
        self._doc_metadata: List[Dict[str, Any]] = []

        self._initialize()

    def _initialize(self) -> None:
        """Initialize ChromaDB and BM25 index."""
        try:
            import chromadb
            from chromadb.config import Settings

            # Load ChromaDB collection
            db_path = str(get_vector_db_dir())
            client = chromadb.PersistentClient(
                path=db_path,
                settings=Settings(anonymized_telemetry=False)
            )

            # Build embedding function matching to configured model
            # (must match what was used during ingestion)
            try:
                from ingestion.vector_ingest import build_embedding_function
                embedding_fn = build_embedding_function(self.config.embedding_model)
            except Exception:
                embedding_fn = None  # Fall back to ChromaDB default

            self._collection = client.get_collection(
                name=self.collection_name,
                embedding_function=embedding_fn,
            )

            # Get all documents for BM25 indexing
            all_docs = self._collection.get(include=["documents", "metadatas"])

            if not all_docs or not all_docs["ids"]:
                logger.warning(f"Collection {self.collection_name} is empty or not found.")
                return

            self._doc_ids = all_docs["ids"]
            # Ensure documents are strings, handle potential None values
            self._documents = [doc if doc is not None else "" for doc in all_docs["documents"]]
            self._doc_metadata = all_docs["metadatas"]

            # Build BM25 index
            tokenized_docs = [self._tokenize(doc) for doc in self._documents]
            self._bm25 = BM25Okapi(tokenized_docs)

            logger.info(
                f"Initialized HybridRetriever with {len(self._documents)} documents"
            )
        except Exception as e:
            logger.error(f"Failed to initialize HybridRetriever: {e}")
            raise

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing.

        Uses simple whitespace + punctuation tokenization.
        Preserves legal terms like "Section 10" as single tokens.
        """
        if not text:
            return []

        # Lowercase
        text = text.lower()

        # Keep "section X" together
        text = re.sub(r"section\s+(\d+[a-z]*)", r"section_\1", text)

        # Split on whitespace and punctuation
        tokens = re.findall(r"\b\w+\b", text)

        return tokens

    def _expand_query_for_keyword(self, query: str) -> str:
        """
        Expand the query with Act-specific terms for BM25 keyword search.
        """
        q_lower = query.lower()
        expansions: List[str] = []

        # --- Sale of Goods Act 1957 ---
        sale_of_goods_signals = [
            "sale", "goods", "caveat emptor", "implied condition",
            "buyer", "seller", "merchantable", "delivery of goods",
            "title in goods", "acceptance of goods", "rejection",
        ]
        if any(sig in q_lower for sig in sale_of_goods_signals):
            expansions.append("Sale of Goods Act 1957")

        # --- Specific Relief Act 1951 ---
        specific_relief_signals = [
            "specific performance", "injunction", "rectification",
            "mandatory order", "perpetual injunction", "temporary injunction",
            "specific relief", "enforcement of contract", "declaratory",
        ]
        if any(sig in q_lower for sig in specific_relief_signals):
            expansions.append("Specific Relief Act 1951")

        # --- Housing Development Act 1966 ---
        housing_signals = [
            "housing", "developer", "licensed", "housing developer",
            "housing development", "controller", "housing account",
            "purchaser of housing", "abandoned project",
        ]
        if any(sig in q_lower for sig in housing_signals):
            expansions.append("Housing Development Act 1966")

        # --- Contracts Act 1950 — only when contract-specific terms appear ---
        contracts_signals = [
            "coercion", "undue influence", "misrepresentation",
            "fraud", "void agreement", "wagering", "restraint of trade",
            "restraint of marriage", "consideration", "free consent",
        ]
        if any(sig in q_lower for sig in contracts_signals):
            expansions.append("Contracts Act 1950")

        if expansions:
            return query + " " + " ".join(expansions)
        return query

    def _get_dynamic_weights(self, query: str) -> Tuple[float, float]:
        """
        Return (semantic_weight, keyword_weight) tuned for the query type.
        """
        q_lower = query.lower()

        # Queries that name a section explicitly → heavy keyword bias
        section_indicators = [
            "section", "under section", "what does section",
            "pursuant to section", "s.",
        ]
        if any(ind in q_lower for ind in section_indicators):
            return 0.25, 0.75

        # Queries that name an Act explicitly → moderate keyword bias
        act_indicators = [
            "act", "under the", "pursuant to the",
            "contracts act", "sale of goods", "specific relief",
            "housing development",
        ]
        if any(ind in q_lower for ind in act_indicators):
            return 0.35, 0.65

        # Short, precise term queries → lean keyword
        if len(query.split()) <= 6:
            return 0.35, 0.65

        # Default: balanced (original config values)
        return self.semantic_weight, self.keyword_weight

    def _strip_embedding_header(self, text: str) -> str:
        """
        Remove the identity header prepended during ingestion.
        """
        if text.startswith("["):
            end_bracket = text.find("]\n\n")
            if end_bracket != -1:
                return text[end_bracket + 3:]
        return text

    def _extract_section_number(self, query: str) -> Optional[str]:
        """
        Extract section number from query if present.

        Returns:
            Section number like "15", "2(d)" or None.
        """
        # Match patterns like "S15", "section 15", "Section 15"
        match = re.search(r'(?i)\b(S|section)\s*(\d+[a-z]*)', query, re.IGNORECASE)
        if match:
            return match.group(2)
        return None

    def _extract_title_keywords(self, query: str) -> List[str]:
        """
        Extract important keywords from query that might match section titles.

        Returns:
            List of stemmed keywords from query.
        """
        # Remove common words
        stop_words = {"what", "is", "are", "of", "under", "to", "can", "when", "how", "the", "a", "an"}
        words = re.findall(r'\b\w+\b', query.lower())

        # Legal terms that often appear in section titles
        legal_terms = {"coercion", "breach", "rectification", "injunction",
                   "restraint", "consideration", "performance", "consent", "fraud"}

        # Return words that are either legal terms OR longer than 2 chars
        return [w for w in words if (w in legal_terms) or (len(w) > 2 and w not in stop_words)]

    def _semantic_search(
        self,
        query: str,
        n_results: int
    ) -> List[Tuple[str, float]]:
        """
        Perform semantic search using ChromaDB.

        Returns list of (doc_id, distance) tuples.
        """
        if not self._collection:
            logger.error("Collection not initialized.")
            return []

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["distances"]
            )

            if not results["ids"]:
                return []

            # Convert distances to similarity scores (1 - distance for cosine)
            return [
                (doc_id, 1 - distance)
                for doc_id, distance in zip(
                    results["ids"][0],
                    results["distances"][0]
                )
            ]
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def _keyword_search(
        self,
        query: str,
        n_results: int
    ) -> List[Tuple[str, float]]:
        """
        Perform keyword search using BM25.

        Returns list of (doc_id, score) tuples.
        """
        if not self._bm25:
            logger.warning("BM25 index not initialized.")
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Get top N indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:n_results]

        # Return (doc_id, score) pairs
        return [
            (self._doc_ids[i], scores[i])
            for i in top_indices
            if scores[i] > 0  # Filter zero scores
        ]

    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[Tuple[str, float]],
        keyword_results: List[Tuple[str, float]]
    ) -> Dict[str, float]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).

        RRF score = sum(1 / (k + rank)) for each method

        Args:
            semantic_results: List of (doc_id, score) from semantic search.
            keyword_results: List of (doc_id, score) from keyword search.

        Returns:
            Dictionary mapping doc_id to combined RRF score.
        """
        rrf_scores: Dict[str, float] = defaultdict(float)

        # Add semantic ranks
        for rank, (doc_id, _) in enumerate(semantic_results, start=1):
            rrf_scores[doc_id] += self.semantic_weight / (self.rrf_k + rank)

        # Add keyword ranks
        for rank, (doc_id, _) in enumerate(keyword_results, start=1):
            rrf_scores[doc_id] += self.keyword_weight / (self.rrf_k + rank)

        return dict(rrf_scores)

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        method: str = "hybrid",
        use_reranker: bool = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant legal chunks for a query.

        Args:
            query: The user's legal question.
            n_results: Number of results to return.
            method: "hybrid", "semantic", or "keyword".

        Returns:
            List of RetrievalResult objects, sorted by relevance.
        """
        try:
            # Determine whether to use reranker
            current_use_reranker = self.config.enable_reranker if use_reranker is None else use_reranker

            # NEW: Tier 1 - Check for explicit section number query
            section_number = self._extract_section_number(query)

            # If explicit section query, use hard metadata filter + skip reranker
            if section_number:
                logger.info(f"Explicit section query detected: '{query}' -> Section {section_number}")

                if not self._collection:
                    logger.error("Collection not initialized.")
                    return []

                try:
                    # Query ChromaDB with metadata filter for exact section match
                    results = self._collection.query(
                        query_texts=[query],
                        n_results=n_results,
                        where={"section_number": section_number},  # METADATA FILTER
                        include=["documents", "metadatas", "distances"]
                    )

                    # Build result objects
                    retrieval_results = []
                    if results["ids"]:
                        for i in range(len(results["ids"])):
                            idx = self._doc_ids.index(results["ids"][i])
                            metadata = self._doc_metadata[idx]
                            retrieval_results.append(RetrievalResult(
                                chunk_id=results["ids"][i],
                                content=self._strip_embedding_header(self._documents[idx]),
                                act_name=metadata.get("act_name", ""),
                                act_number=metadata.get("act_number", 0),
                                section_number=metadata.get("section_number", ""),
                                section_title=metadata.get("section_title", ""),
                                score=1 - results["distances"][0][i],
                                retrieval_method=f"metadata_filter_{section_number}"
                            ))

                    logger.info(f"Metadata filter returned {len(retrieval_results)} results")
                    return retrieval_results[:n_results]

                except Exception as e:
                    logger.error(f"Metadata filter query failed: {e}")
                    return []
            # END Tier 1: Metadata Filter

            # Determine how many documents to fetch before reranking/fusion
            initial_top_k = self.config.reranker_top_k if current_use_reranker else n_results
            fetch_k = max(initial_top_k * 2, n_results * 4)

            # Perform searches based on method
            semantic_results: List[Tuple[str, float]] = []
            keyword_results: List[Tuple[str, float]] = []

            if method in ("hybrid", "semantic"):
                semantic_results = self._semantic_search(query, fetch_k)

            if method in ("hybrid", "keyword"):
                # Expand query with act-domain hints for BM25 only
                keyword_query = self._expand_query_for_keyword(query)
                keyword_results = self._keyword_search(keyword_query, fetch_k)

            # Combine results
            combined_scores: Dict[str, float] = {}
            if method == "hybrid":
                # Use per-query dynamic weights instead of fixed 50/50 split
                sem_w, kw_w = self._get_dynamic_weights(query)
                logger.debug(
                    f"Dynamic weights for query '{query[:50]}...': "
                    f"semantic={sem_w}, keyword={kw_w}"
                )
                # Temporarily override weights for this retrieval call
                original_sem_w = self.semantic_weight
                original_kw_w = self.keyword_weight
                self.semantic_weight = sem_w
                self.keyword_weight = kw_w
                combined_scores = self._reciprocal_rank_fusion(
                    semantic_results, keyword_results
                )
                # Restore original weights
                self.semantic_weight = original_sem_w
                self.keyword_weight = original_kw_w
            elif method == "semantic":
                combined_scores = {doc_id: score for doc_id, score in semantic_results}
            else:  # keyword
                combined_scores = {doc_id: score for doc_id, score in keyword_results}

            # Sort by score and take top N for initial retrieval
            # Retrieve more if reranking to have candidates
            sorted_ids = sorted(
                combined_scores.keys(),
                key=lambda x: combined_scores[x],
                reverse=True
            )[:initial_top_k]

            # Build initial result objects
            initial_results = []
            for doc_id in sorted_ids:
                if doc_id not in self._doc_ids:
                    continue
                # Find document index
                idx = self._doc_ids.index(doc_id)
                metadata = self._doc_metadata[idx]

                result = RetrievalResult(
                    chunk_id=doc_id,
                    content=self._strip_embedding_header(self._documents[idx]),
                    act_name=metadata.get("act_name", ""),
                    act_number=metadata.get("act_number", 0),
                    section_number=metadata.get("section_number", ""),
                    section_title=metadata.get("section_title", ""),
                    score=combined_scores[doc_id],
                    retrieval_method=method
                )
                initial_results.append(result)

            # Apply cross-encoder reranking if enabled
            if current_use_reranker:
                reranker = get_reranker()
                if reranker is not None:
                    logger.info(f"Reranking {len(initial_results)} results for query: {query}")

                    reranked_results = reranker.rerank_results(
                        query=query,
                        retrieval_results=initial_results,
                        top_k=n_results
                    )
                    
                    # NEW: Apply title keyword boost after reranking
                    query_keywords = self._extract_title_keywords(query)
                    for result in reranked_results:
                        if result.section_title:
                            section_title_lower = result.section_title.lower()

                            # Boost if section title contains query keywords
                            if any(kw in section_title_lower for kw in query_keywords):
                                result.score += 0.2  # Additive boost for cross-encoder logits
                                logger.debug(
                                    f"Title boost applied to '{result.section_title}' -> "
                                    f"score: {result.score:.4f}"
                                )

                    # Update retrieval method to indicate reranking
                    for result in reranked_results:
                        result.retrieval_method = f"{method}_reranked"

                    # Re-sort after applying the boost
                    reranked_results.sort(key=lambda x: x.score, reverse=True)

                    return reranked_results
                else:
                    logger.warning("Reranker requested but not available, using original results")

            # Return top N results (or all if fewer than n_results)
            return initial_results[:n_results]

        except Exception as e:
            logger.error(f"Retrieval failed for query '{query}': {e}")
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


def test_retriever():
    """Test hybrid retriever with sample queries."""
    try:
        retriever = HybridRetriever()

        test_queries = [
            "What is definition of consideration in contract law?",
            "Section 10 free consent",
            "housing developer license requirements",
            "specific performance contract enforcement",
        ]

        print("=" * 70)
        print("Hybrid Retriever Test")
        print("=" * 70)

        for query in test_queries:
            print(f"\nQuery: {query}")
            print("-" * 50)

            # Test all methods
            for method in ["semantic", "keyword", "hybrid"]:
                results = retriever.retrieve(query, n_results=3, method=method)
                print(f"\n  [{method.upper()}]")
                for r in results:
                    print(
                        f"    - {r.act_name} Section {r.section_number} "
                        f"(score: {r.score:.4f})"
                    )
    except Exception as e:
        logger.error(f"Test failed: {e}")


if __name__ == "__main__":
    test_retriever()
