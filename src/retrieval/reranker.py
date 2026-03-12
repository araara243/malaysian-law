"""
Cross-Encoder Reranker for Improved Retrieval Quality

This module implements cross-encoder reranking to improve retrieval accuracy.
Cross-encoders are more accurate than bi-encoders for relevance scoring
because they process the query-document pair together.

Expected impact: 94.5% → 97%+ Hit@1 on retrieval metrics.

Reference: https://www.sbert.net/examples/applications/retrieve_rerank/
"""

import logging
import json
import os
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional

import sys
from pathlib import Path

# Add parent directories to path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from config import RAGConfig, setup_logging

logger = setup_logging(__name__)


@dataclass
class RerankResult:
    """Result of reranking operation."""
    doc_id: str
    relevance_score: float
    original_rank: int
    new_rank: int


class CrossEncoderReranker:
    """
    Cross-encoder based reranker for improving retrieval quality.
    
    Uses a pre-trained cross-encoder model to rescore retrieved documents.
    This is more accurate than bi-encoder similarity because it evaluates
    the query-document relationship jointly.
    
    Default model: ms-marco-MiniLM-L-6-v2
    - Fast inference (~1ms per pair)
    - Trained on MS MARCO passage ranking dataset
    - Good general-purpose performance for legal text
    
    Alternative models:
    - ms-marco-MiniLM-L-12-v2: Higher accuracy, slower
    - cross-encoder/stsb-roberta-large: Semantic similarity focused
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        batch_size: int = 32
    ):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: HuggingFace model identifier.
            device: "cpu" or "cuda" for GPU acceleration.
            batch_size: Batch size for inference.
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._load_model()
    
    def _load_model(self):
        """Load the cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self.model_name,
                device=self.device
            )
            logger.info(
                f"Loaded cross-encoder reranker: {self.model_name} "
                f"(device: {self.device})"
            )
        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")
            raise
    
    def rerank(
        self,
        query: str,
        documents: List[Tuple[str, str]],  # List of (doc_id, content) pairs
        top_k: int = None
    ) -> List[RerankResult]:
        """
        Rerank documents based on query relevance.
        
        Args:
            query: The user's query.
            documents: List of (doc_id, content) tuples to rerank.
            top_k: Number of top results to return. If None, returns all.
        
        Returns:
            List of RerankResult objects sorted by relevance score.
        """
        if not documents:
            return []
        
        if self._model is None:
            logger.warning("Reranker model not loaded, returning original order")
            return [
                RerankResult(
                    doc_id=doc_id,
                    relevance_score=0.0,
                    original_rank=i,
                    new_rank=i
                )
                for i, (doc_id, _) in enumerate(documents)
            ]
        
        try:
            # Prepare query-document pairs for scoring
            pairs = [[query, content] for _, content in documents]
            
            # Get relevance scores from cross-encoder
            scores = self._model.predict(pairs, batch_size=self.batch_size)
            
            # Create rerank results
            results = []
            for idx, ((doc_id, _), score) in enumerate(zip(documents, scores)):
                results.append(RerankResult(
                    doc_id=doc_id,
                    relevance_score=float(score),
                    original_rank=idx,
                    new_rank=idx  # Will be updated after sorting
                ))
            
            # Sort by relevance score (descending)
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Update new ranks
            for new_rank, result in enumerate(results):
                result.new_rank = new_rank
            
            # Limit to top_k if specified
            if top_k is not None:
                results = results[:top_k]
            
            logger.info(
                f"Reranked {len(documents)} documents, "
                f"returning top {len(results)}"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Return original order as fallback
            return [
                RerankResult(
                    doc_id=doc_id,
                    relevance_score=0.0,
                    original_rank=i,
                    new_rank=i
                )
                for i, (doc_id, _) in enumerate(documents)
            ]
    
    def rerank_results(
        self,
        query: str,
        retrieval_results: List,
        top_k: int = None
    ) -> List:
        """
        Rerank RetrievalResult objects.
        
        Args:
            query: The user's query.
            retrieval_results: List of RetrievalResult objects.
            top_k: Number of top results to return.
        
        Returns:
            List of reranked RetrievalResult objects.
        """
        if not retrieval_results:
            return []
        
        # Extract doc_id and content for reranking
        doc_pairs = [
            (r.chunk_id, r.content)
            for r in retrieval_results
        ]
        
        # Rerank
        rerank_results = self.rerank(query, doc_pairs, top_k)
        
        # Build result mapping by doc_id
        result_map = {r.chunk_id: r for r in retrieval_results}
        
        # Return reranked results
        reranked = []
        for rr in rerank_results:
            original = result_map.get(rr.doc_id)
            if original:
                # Update score with rerank score
                original.score = rr.relevance_score
                reranked.append(original)
        
        return reranked

class LLMReranker:
    """
    LLM-based reranker that uses an instruction-following LLM to rank
    retrieved legal sections by their relevance to the query.

    Unlike cross-encoders (which score text similarity), an LLM understands
    legal concepts and section purposes, allowing it to distinguish between
    adjacent sections with overlapping vocabulary (e.g., S15 vs S19).

    Uses OpenRouter API (configured via OPENROUTER_API_KEY env variable).
    Falls back to original retrieval order on any API failure.
    """

    RANKING_PROMPT = """You are a Malaysian legal expert assistant helping to retrieve the EXACT legal section that directly answers a user's question.

User Question: {query}

You are given a list of retrieved legal sections. Your task is to rank them so that the MOST DIRECTLY RELEVANT section appears first.

CRITICAL RANKING RULES:
1. PRIORITY #1 — The section that DIRECTLY DEFINES the legal term/concept in the question MUST come first. Examples:
   - "What is coercion?" → the section titled "Coercion" that defines it, NOT a section that says when coercion voids an agreement
   - "When can specific performance be refused?" → the section on discretion to refuse, NOT the section on when it can be granted
   - "What is the Controller's power?" → the section explicitly titled about the Controller's powers
2. PRIORITY #2 — The section whose TITLE best matches the key noun/verb in the question.
3. PRIORITY #3 — The section whose content most directly answers the question without requiring inference.
4. DO NOT prefer sections just because they mention related keywords; prefer definitional/authoritative sections.

Sections:
{sections}

Respond with ONLY a flat JSON array of integers (the 1-based section indices) in order from most to least relevant.
Example: [3, 1, 5, 2, 4]
Your response:"""

    def __init__(self, config=None):
        """
        Initialize LLM reranker.

        Args:
            config: RAGConfig object with LLM settings.
        """
        from config import RAGConfig
        self.config = config or RAGConfig()
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        self.model = getattr(self.config, "llm_reranker_model", "google/gemini-2.0-flash-001")
        self.base_url = getattr(self.config, "openrouter_base_url", "https://openrouter.ai/api/v1")

        if not self.api_key:
            logger.warning(
                "OPENROUTER_API_KEY not found. LLMReranker will fall back to original order."
            )
        else:
            logger.info(f"LLMReranker initialized with model: {self.model}")

    def _build_sections_text(self, retrieval_results: List) -> str:
        """Format retrieved results into a numbered list for the LLM prompt."""
        lines = []
        for i, r in enumerate(retrieval_results, start=1):
            title_part = f" — {r.section_title}" if r.section_title else ""
            # Give the LLM section title + first 250 chars of content
            snippet = r.content[:250].replace("\n", " ").strip()
            if len(r.content) > 250:
                snippet += "..."
            lines.append(
                f"[{i}] Act: {r.act_name}, Section {r.section_number}{title_part}\n"
                f"    Content: {snippet}"
            )
        return "\n\n".join(lines)

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call OpenRouter API and return raw text response, or None on failure."""
        try:
            import urllib.request
            payload = json.dumps({
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 100,
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/mylaw-rag",
                    "X-Title": "MyLaw RAG",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning(f"LLM reranker API call failed: {e}")
            return None

    def _parse_ranking(self, response: str, n: int) -> Optional[List[int]]:
        """Parse LLM response to extract 1-based index list — robust version."""
        try:
            # Strip markdown code fences if present
            cleaned = re.sub(r'```[\w]*', '', response).strip()

            # Attempt to parse entire response as JSON first
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                # Fall back: find first occurrence of a JSON array
                match = re.search(r'\[.*?\]', cleaned, re.DOTALL)
                if not match:
                    return None
                try:
                    parsed = json.loads(match.group(0))
                except json.JSONDecodeError:
                    return None

            # Flatten nested arrays (e.g. [["4","5"],["1"],...] → [4,5,1,...])
            flat: List[int] = []
            def _flatten(obj):
                if isinstance(obj, list):
                    for item in obj:
                        _flatten(item)
                else:
                    try:
                        flat.append(int(str(obj).strip()))
                    except (ValueError, TypeError):
                        pass
            _flatten(parsed)

            # Deduplicate while preserving order, keep only valid indices
            seen: set = set()
            valid: List[int] = []
            for idx in flat:
                if 1 <= idx <= n and idx not in seen:
                    seen.add(idx)
                    valid.append(idx)

            # Append any missing indices at the end
            for i in range(1, n + 1):
                if i not in seen:
                    valid.append(i)

            return valid if valid else None
        except Exception:
            return None

    def rerank_results(
        self,
        query: str,
        retrieval_results: List,
        top_k: int = None
    ) -> List:
        """
        Rerank RetrievalResult objects using an LLM.

        Args:
            query: The user's legal question.
            retrieval_results: List of RetrievalResult objects.
            top_k: Number of top results to return.

        Returns:
            List of reranked RetrievalResult objects.
        """
        if not retrieval_results:
            return []

        if not self.api_key:
            logger.warning("No API key — returning original retrieval order.")
            return retrieval_results[:top_k] if top_k else retrieval_results

        n = len(retrieval_results)
        sections_text = self._build_sections_text(retrieval_results)
        prompt = self.RANKING_PROMPT.format(query=query, sections=sections_text)

        logger.info(f"LLM reranking {n} results for query: {query[:60]}...")
        response = self._call_llm(prompt)

        if response is None:
            logger.warning("LLM reranker returned no response — using original order.")
            return retrieval_results[:top_k] if top_k else retrieval_results

        logger.debug(f"LLM ranking response: {response}")
        ranking = self._parse_ranking(response, n)

        if ranking is None:
            logger.warning(f"Could not parse LLM ranking '{response}' — using original order.")
            return retrieval_results[:top_k] if top_k else retrieval_results

        # Reorder results based on LLM ranking (indices are 1-based)
        reranked = []
        for i, idx in enumerate(ranking):
            result = retrieval_results[idx - 1]
            # Give a synthetic descending score so downstream code sorts correctly
            result.score = 1.0 - (i * 0.01)
            result.retrieval_method = f"{result.retrieval_method}_llm_reranked"
            reranked.append(result)

        logger.info(f"LLM reranked {n} results, returning top {top_k or n}")
        return reranked[:top_k] if top_k else reranked


def test_reranker():
    """Test cross-encoder reranker with sample data."""
    import time
    
    try:
        reranker = CrossEncoderReranker()
        
        query = "What is consideration in contract law?"
        
        documents = [
            ("doc1", "Consideration is something of value given by both parties to a contract."),
            ("doc2", "A contract requires offer, acceptance, and consideration."),
            ("doc3", "Contracts can be oral or written in Malaysia."),
            ("doc4", "Consideration must be sufficient but need not be adequate."),
            ("doc5", "The Contracts Act 1950 defines essential elements of contracts."),
        ]
        
        print("=" * 70)
        print("Cross-Encoder Reranker Test")
        print("=" * 70)
        print(f"\nQuery: {query}\n")
        print("Original order:")
        for i, (doc_id, content) in enumerate(documents):
            print(f"  {i+1}. [{doc_id}] {content[:60]}...")
        
        start = time.time()
        results = reranker.rerank(query, documents, top_k=3)
        elapsed = time.time() - start
        
        print(f"\nReranked (took {elapsed*1000:.1f}ms):")
        for i, result in enumerate(results):
            idx = documents.index(next(
                (d for d in documents if d[0] == result.doc_id),
                None
            ))
            content = documents[idx][1]
            print(
                f"  {i+1}. [{result.doc_id}] (score: {result.relevance_score:.3f}) "
                f"{content[:60]}..."
            )
            print(f"     Original rank: {result.original_rank + 1} → New rank: {result.new_rank + 1}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")


if __name__ == "__main__":
    test_reranker()
