"""
Citation Verification for Legal RAG

This module provides citation verification to prevent legal hallucinations.
LLMs may invent section numbers or act names that don't exist.
This module verifies citations against the retrieved context.

Critical for legal RAG because:
- Legal professionals require accurate citations
- Hallucinated section numbers can mislead users
- Credibility depends on source verification
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import sys
from pathlib import Path

# Add parent directories to path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from config import setup_logging

logger = setup_logging(__name__)


@dataclass
class Citation:
    """A legal citation extracted from text."""
    text: str
    act_name: Optional[str] = None
    section_number: Optional[str] = None
    start_idx: int = 0
    end_idx: int = 0


@dataclass
class VerificationResult:
    """Result of citation verification."""
    total_citations: int
    verified_citations: List[Citation]
    hallucinated_citations: List[Citation]
    verification_rate: float
    warnings: List[str]


class CitationVerifier:
    """
    Verifies legal citations in LLM responses.
    
    Extracts citations from responses and checks if they exist
    in the retrieved context to prevent hallucinations.
    """
    
    # Citation patterns
    ACT_PATTERNS = [
        r'([A-Za-z\s]+(?:Act\s+)?(?:No\.?\s*)?(\d{1,4}))',
        r'Act\s+(\d{1,4})',
    ]
    
    SECTION_PATTERNS = [
        r'section\s+(\d+[a-z]?)',
        r'Section\s+(\d+[a-z]?)',
        r'\.(\d+)(?:\(|\)|$)',  # "10)" or "10."
        r's\.(\d+[a-z]?)',  # s.10 or s.10a
    ]
    
    def __init__(self):
        """Initialize citation verifier."""
        self.citations: List[Citation] = []
    
    def extract_citations(self, text: str) -> List[Citation]:
        """
        Extract all legal citations from text.
        
        Args:
            text: The text to extract citations from.
        
        Returns:
            List of Citation objects.
        """
        citations = []
        text_lower = text.lower()
        
        # Find all act citations
        for pattern in self.ACT_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                citation_text = match.group(0)
                act_name = match.group(1).strip() if match.groups() else None
                act_number = match.group(2) if match.groups() and len(match.groups()) > 1 else None
                
                citations.append(Citation(
                    text=citation_text,
                    act_name=act_name,
                    section_number=None,
                    start_idx=match.start(),
                    end_idx=match.end()
                ))
        
        # Find all section citations
        for pattern in self.SECTION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                citation_text = match.group(0)
                section_number = match.group(1) if match.groups() else None
                
                citations.append(Citation(
                    text=citation_text,
                    act_name=None,
                    section_number=section_number,
                    start_idx=match.start(),
                    end_idx=match.end()
                ))
        
        # Deduplicate citations
        seen = set()
        unique_citations = []
        for citation in citations:
            key = f"{citation.act_name}-{citation.section_number}"
            if key not in seen:
                seen.add(key)
                unique_citations.append(citation)
        
        return unique_citations
    
    def verify_citations(
        self,
        citations: List[Citation],
        context_docs: List[Dict]
    ) -> VerificationResult:
        """
        Verify citations against retrieved context documents.
        
        Args:
            citations: List of citations to verify.
            context_docs: List of context document dicts with metadata.
                        Each dict should have: content, act_name, section_number.
        
        Returns:
            VerificationResult with verification details.
        """
        verified = []
        hallucinated = []
        warnings = []
        
        # Build searchable index of context
        context_index = []
        for doc in context_docs:
            content_lower = doc.get("content", "").lower()
            act_name = doc.get("act_name", "").lower()
            section_number = doc.get("section_number", "").lower()
            
            context_index.append({
                "content": content_lower,
                "act_name": act_name,
                "section_number": section_number,
                "original": doc
            })
        
        # Verify each citation
        for citation in citations:
            is_verified = False
            
            # Check section citations
            if citation.section_number:
                section_norm = citation.section_number.lower()
                for ctx in context_index:
                    ctx_section = ctx["section_number"]
                    if ctx_section and section_norm in ctx_section:
                        is_verified = True
                        break
            
            # Check act citations
            if not is_verified and citation.act_name:
                act_norm = citation.act_name.lower()
                for ctx in context_index:
                    ctx_act = ctx["act_name"]
                    if ctx_act and act_norm in ctx_act:
                        is_verified = True
                        break
            
            # Check for exact text match
            if not is_verified:
                citation_lower = citation.text.lower()
                for ctx in context_index:
                    if citation_lower in ctx["content"]:
                        is_verified = True
                        break
            
            if is_verified:
                verified.append(citation)
            else:
                hallucinated.append(citation)
                warnings.append(
                    f"Potential hallucination: '{citation.text}' "
                    f"not found in retrieved context"
                )
        
        verification_rate = (
            len(verified) / len(citations) if citations else 1.0
        )
        
        # Log warnings for hallucinations
        for warning in warnings:
            logger.warning(warning)
        
        return VerificationResult(
            total_citations=len(citations),
            verified_citations=verified,
            hallucinated_citations=hallucinated,
            verification_rate=verification_rate,
            warnings=warnings
        )
    
    def verify_response(
        self,
        response: str,
        retrieved_context: List[Dict]
    ) -> VerificationResult:
        """
        Verify citations in a complete LLM response.
        
        Args:
            response: The LLM response text.
            retrieved_context: List of retrieved context documents.
        
        Returns:
            VerificationResult with verification details.
        """
        # Extract citations from response
        citations = self.extract_citations(response)
        
        # Verify against context
        result = self.verify_citations(citations, retrieved_context)
        
        return result
    
    def format_verification_result(
        self,
        result: VerificationResult,
        include_details: bool = True
    ) -> str:
        """
        Format verification result for display.
        
        Args:
            result: VerificationResult to format.
            include_details: Whether to include detailed breakdown.
        
        Returns:
            Formatted string for display.
        """
        if result.total_citations == 0:
            return "✅ No citations to verify"
        
        percentage = result.verification_rate * 100
        
        if result.verification_rate == 1.0:
            status = "✅ All citations verified"
        elif result.verification_rate >= 0.8:
            status = f"⚠️ {percentage:.0f}% of citations verified"
        elif result.verification_rate >= 0.5:
            status = f"⚠️ {percentage:.0f}% of citations verified"
        else:
            status = f"❌ {percentage:.0f}% of citations verified"
        
        lines = [status]
        
        if include_details:
            lines.append(f"\nTotal citations: {result.total_citations}")
            lines.append(f"Verified: {len(result.verified_citations)}")
            lines.append(f"Unverified: {len(result.hallucinated_citations)}")
            
            if result.hallucinated_citations:
                lines.append("\n⚠️ Potentially hallucinated citations:")
                for citation in result.hallucinated_citations:
                    lines.append(f"  - {citation.text}")
            
            if result.warnings:
                lines.append("\n⚠️ Warnings:")
                for warning in result.warnings:
                    lines.append(f"  - {warning}")
        
        return "\n".join(lines)


def test_citation_verifier():
    """Test citation verifier with sample data."""
    verifier = CitationVerifier()
    
    # Sample response with mixed citations
    response = """
    Under Section 10 of the Contracts Act 1950, all agreements are 
    contracts if they are made by the free consent of parties 
    competent to contract. Section 13 defines coercion and 
    Section 14 defines undue influence. According to Section 25, 
    agreements without consideration are void.
    """
    
    # Sample context documents
    context = [
        {
            "content": "Section 10. All agreements are contracts if they are made by the free consent of parties competent to contract.",
            "act_name": "Contracts Act 1950",
            "section_number": "10"
        },
        {
            "content": "Section 13. Consent is said to be free when it is not caused by coercion.",
            "act_name": "Contracts Act 1950",
            "section_number": "13"
        },
        {
            "content": "Section 25. An agreement without consideration is void.",
            "act_name": "Contracts Act 1950",
            "section_number": "25"
        }
    ]
    
    print("=" * 70)
    print("Citation Verifier Test")
    print("=" * 70)
    print(f"\nResponse:\n{response}\n")
    print(f"\nContext documents: {len(context)}\n")
    
    result = verifier.verify_response(response, context)
    
    print("\nVerification Result:")
    print("-" * 50)
    print(verifier.format_verification_result(result))


if __name__ == "__main__":
    test_citation_verifier()
