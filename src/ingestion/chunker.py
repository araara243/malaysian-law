"""
Semantic Chunking for Malaysian Legal Documents

This module implements structure-aware chunking for Malaysian legal texts.
Instead of naive token-based splitting, it chunks by legal sections to
ensure each chunk contains a complete legal definition or provision.

Key features:
- Detect Section headers (Section 1, Section 2, etc.)
- Detect Part headers (PART I, PART II, etc.)
- Keep subsections with their parent sections
- Include metadata (Act name, section number) for citation
"""

import sys
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, cast

# Add project root to sys.path for direct execution
root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import tiktoken # type: ignore

# Import config - handle both module and direct execution
try:
    from src.config import RAGConfig, get_processed_dir, setup_logging # type: ignore
except ImportError:
    from config import RAGConfig, get_processed_dir, setup_logging # type: ignore

# Configure logging
logger = setup_logging(__name__)

# Token counting
TOKENIZER: Optional[Any] = None


def get_tokenizer() -> Any:
    """Lazy load the tokenizer."""
    global TOKENIZER
    if TOKENIZER is None:
        try:
            TOKENIZER = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    return TOKENIZER


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    try:
        return len(get_tokenizer().encode(text))
    except Exception as e:
        logger.warning(f"Error counting tokens, returning 0: {e}")
        return 0


@dataclass
class LegalChunk:
    """A chunk of legal text with enhanced metadata."""
    chunk_id: str
    act_name: str
    act_number: int
    act_year: int                    # NEW: Year of enactment
    category: str                    # NEW: Legal domain category
    part: Optional[str]
    section_number: Optional[str]
    section_title: Optional[str]
    subsection: Optional[str]        # NEW: Subsection if present
    content: str
    token_count: int
    start_position: int
    cross_references: List[str]      # NEW: References to other Acts
    keywords: List[str]              # NEW: Extracted legal terms
    contained_sections: List[str]    # NEW: All section numbers mentioned in chunk
    
# Configure logging
logger = setup_logging(__name__)

# Regex patterns for Malaysian legal document structure
SECTION_PATTERN = re.compile(
    r"^(?:(?:Section|Seksyen)\s+(\d+[A-Za-z]*)|(\d+[A-Za-z]*)\.)\s*(.*)$",
    re.IGNORECASE | re.MULTILINE
)

PART_PATTERN = re.compile(
    r"^(?:PART|BAHAGIAN)\s+([IVXLCDM]+|\d+)\b[\s\-–—]*(.*)$",
    re.IGNORECASE | re.MULTILINE
)

SUBSECTION_PATTERN = re.compile(
    r"^\s*\((\d+[a-z]?)\)\s*",
    re.MULTILINE
)


def find_sections(text: str) -> List[Dict[str, Any]]:
    """
    Find all sections in the text with their positions.
    Returns all matches found by regex, sorted by position.
    Structure-aware chunking (in chunk_document) handles merging small chunks 
    (like TOC entries) and preserving large ones (Body sections).
    
    Returns a list of dicts with keys:
        - section_number: str
        - title: str
        - start: int (character position)
        - end: int (character position, exclusive)
    """
    sections = []
    
    for match in SECTION_PATTERN.finditer(text):
        # Group 1 is for "Section X", Group 2 is for "X."
        sec_num = match.group(1) or match.group(2)
        title = match.group(3).strip() if match.group(3) else ""
        
        if not sec_num:
            continue
            
        sections.append({
            "section_number": sec_num,
            "title": title,
            "start": match.start(),
            "end": None,  # Will be filled later
        })
    
    # Sort by start position
    sections.sort(key=lambda x: x["start"])
    
    # Set end positions
    for i, section in enumerate(sections):
        if i + 1 < len(sections):
            section["end"] = sections[i + 1]["start"]
        else:
            section["end"] = len(text)
    
    return sections


def find_current_part(text: str, position: int) -> Optional[str]:
    """
    Find the current PART header at a given position.
    
    Args:
        text: Full document text.
        position: Character position to check.
    
    Returns:
        Part identifier (e.g., "PART I - PRELIMINARY") or None.
    """
    current_part = None
    
    for match in PART_PATTERN.finditer(text):
        if match.start() <= position:
            part_num = match.group(1)
            part_title = match.group(2).strip() if match.group(2) else ""
            current_part = f"Part {part_num}"
            if part_title:
                current_part += f" - {part_title}"
        else:
            break
    
    return current_part


def split_large_section(
    section_text: str,
    max_tokens: int = 1000
) -> List[str]:
    """
    Split a large section into smaller chunks while preserving subsection boundaries.
    
    Args:
        section_text: The text of a single section.
        max_tokens: Maximum tokens per chunk.
    
    Returns:
        List of text chunks.
    """
    tokens = count_tokens(section_text)
    
    if tokens <= max_tokens:
        return [section_text]
    
    # Try splitting on subsection markers
    subsection_matches = list(SUBSECTION_PATTERN.finditer(section_text))
    
    if not subsection_matches:
        # No subsections, split by paragraphs
        paragraphs = section_text.split("\n\n")
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            test_chunk = current_chunk + "\n\n" + para if current_chunk else para
            if count_tokens(test_chunk) > max_tokens and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk = test_chunk
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks or [section_text]
    
    # Split on subsection boundaries
    chunks: List[str] = []
    current_chunk: str = cast(str, section_text)[:subsection_matches[0].start()] # type: ignore
    
    for i, match in enumerate(subsection_matches):
        match_start: int = match.start()
        if i + 1 < len(subsection_matches):
            next_start: int = subsection_matches[i + 1].start()
            subsection_text: str = cast(str, section_text)[match_start:next_start] # type: ignore
        else:
            subsection_text: str = cast(str, section_text)[match_start:] # type: ignore
        
        test_chunk = current_chunk + subsection_text # type: ignore
        
        if count_tokens(test_chunk) > max_tokens and current_chunk.strip():
            chunks.append(current_chunk.strip())
            current_chunk = subsection_text
        else:
            current_chunk = test_chunk
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks or [section_text]


def extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
    """
    Extract important legal keywords from text.

    Args:
        text: Legal text to analyze.
        max_keywords: Maximum number of keywords to return.

    Returns:
        List of keywords in order of importance.
    """
    if not text:
        return []

    # Common legal terms to prioritize
    legal_terms = [
        "contract", "agreement", "obligation", "liability", "breach",
        "damages", "consideration", "offer", "acceptance", "party",
        "court", "jurisdiction", "offense", "penalty", "fine",
        "property", "land", "title", "ownership", "lease",
        "criminal", "prosecution", "evidence", "witness", "trial"
    ]

    # Find legal terms present in text (case-insensitive)
    text_lower = text.lower()
    found_terms: List[str] = []

    for term in legal_terms:
        if term in text_lower:
            found_terms.append(term)

    return found_terms[:max_keywords] # type: ignore


def extract_cross_references(text: str) -> List[str]:
    """
    Extract references to other Acts or sections.

    Args:
        text: Legal text to analyze.

    Returns:
        List of cross-reference strings.
    """
    if not text:
        return []

    references = []

    # Pattern: "Section X", "Act Y", "Section X of Act Y"
    patterns = [
        r"Section\s+(\d+[A-Za-z]*)",
        r"Act\s+(\d+[A-Za-z]*)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        references.extend(matches)

    # Deduplicate and return
    refs_list: List[str] = list(set(references))
    return cast(List[str], refs_list)[:10]  # Max 10 references # type: ignore


def repeat_section_title(
    chunk: Dict[str, Any],
    repetitions: int = 3
) -> Dict[str, Any]:
    """
    Repeat section title in chunk content to boost BM25 relevance.

    This helps distinguish between adjacent sections that share vocabulary
    but have different legal purposes (captured in their titles).

    Adjacent sections often have nearly identical vocabulary but different
    section titles that capture the legal nuance:
    - S50: "Injunctions when contract rescindable or voidable"
    - S52: "Injunctions in cases of breach"
    - S53: "Mandatory injunction"

    The section titles contain the disambiguating keywords, but they're
    not given enough weight in BM25/semantic search. By repeating the
    title, we dramatically increase the TF-IDF score for title words.

    Args:
        chunk: Chunk dictionary with 'content' and 'section_title' keys
        repetitions: Number of times to repeat title (default: 3)

    Returns:
        Modified chunk with title repeated in content
    """
    if not chunk.get('section_title'):
        return chunk

    title = chunk['section_title'].strip()
    original_content = chunk['content']

    # Repeat title at beginning (not in the metadata header)
    title_repetition = ' '.join([title] * repetitions) + '. '

    # Prepend to content
    chunk['content'] = title_repetition + original_content

    return chunk


def chunk_document(
    document: Dict[str, Any],
    max_tokens: int = 1000,
    min_tokens: int = 50,
    config: Optional[RAGConfig] = None
) -> List[LegalChunk]:
    """
    Chunk a processed legal document into semantic chunks.

    Args:
        document: A processed document dict with keys:
            - metadata: dict with act_name, act_number, act_year, etc.
            - cleaned_text: str
        max_tokens: Maximum tokens per chunk.
        min_tokens: Minimum tokens per chunk (smaller chunks merged).

    Returns:
        List of LegalChunk objects with enhanced metadata.
    """
    text = document.get("cleaned_text")
    if text is None:
        text = document.get("raw_text", "")
        logger.warning(f"cleaned_text missing for {act_name}, using raw_text")
    
    metadata = document.get("metadata", {})
    act_name = metadata.get("act_name", "Unknown Act")
    act_number = metadata.get("act_number", 0)
    act_year = metadata.get("act_year", 0)
    category = metadata.get("category", "other")

    # Import get_act_category if not provided
    if category == "other":
        try:
            from src.config import get_act_category # type: ignore
        except ImportError:
            from config import get_act_category # type: ignore
        category = get_act_category(act_number)
    
    chunks: List[LegalChunk] = []
    sections = find_sections(text)
    
    if not sections:
        # No sections found, create a single chunk
        logger.warning(f"No sections found in {act_name}, creating single chunk")
        chunk = LegalChunk(
            chunk_id=f"act_{act_number}_full",
            act_name=act_name,
            act_number=act_number,
            act_year=act_year,
            category=category,
            part=None,
            section_number=None,
            section_title=None,
            subsection=None,
            content=text,
            token_count=count_tokens(text),
            start_position=0,
            cross_references=[],
            keywords=extract_keywords(text),
            contained_sections=[]
        )
        return [chunk]
    
    # Add preamble if any text before first section
    if sections[0]["start"] > 0:
        preamble = text[:sections[0]["start"]].strip()
        if preamble and count_tokens(preamble) >= min_tokens:
            chunk = LegalChunk(
                chunk_id=f"act_{act_number}_preamble",
                act_name=act_name,
                act_number=act_number,
                act_year=act_year,
                category=category,
                part=find_current_part(text, 0),
                section_number="Preamble",
                section_title="Preliminary Provisions",
                subsection=None,
                content=preamble,
                token_count=count_tokens(preamble),
                start_position=0,
                cross_references=[],
                keywords=extract_keywords(preamble),
                contained_sections=[]
            )
            chunks.append(chunk)
    
    # Process each section
    seen_ids: Dict[str, int] = {} # dictionary to track counts of each base chunk_id

    for section in sections:
        section_text = text[section["start"]:section["end"]].strip()
        current_part = find_current_part(text, section["start"])
        
        # Split large sections
        section_chunks = split_large_section(section_text, max_tokens)
        
        for i, chunk_text in enumerate(section_chunks):
            if count_tokens(chunk_text) < min_tokens and chunks:
                # Merge small chunk with previous
                prev: LegalChunk = cast(List[LegalChunk], chunks)[-1] # type: ignore
                chunks[-1] = LegalChunk(
                    chunk_id=prev.chunk_id,
                    act_name=prev.act_name,
                    act_number=prev.act_number,
                    act_year=prev.act_year,
                    category=prev.category,
                    part=prev.part,
                    section_number=prev.section_number,
                    section_title=prev.section_title,
                    subsection=prev.subsection,
                    content=prev.content + "\n\n" + chunk_text,
                    token_count=count_tokens(prev.content + "\n\n" + chunk_text),
                    start_position=prev.start_position,
                    cross_references=prev.cross_references,
                    keywords=prev.keywords,
                    contained_sections=prev.contained_sections
                )
                continue
            
            # Base ID generation
            base_id = f"act_{act_number}_s{cast(Dict[str, Any], section)['section_number']}" # type: ignore
            
            # Sub-chunk handling (from large section split)
            if len(section_chunks) > 1:
                base_id += f"_{i+1}"
            
            # Deduplication suffix handling
            if base_id in cast(Dict[str, int], seen_ids): # type: ignore
                seen_ids[base_id] = cast(int, seen_ids[base_id]) + 1 # type: ignore
                chunk_id = f"{base_id}_dup{seen_ids[base_id]}" # type: ignore
            else:
                seen_ids[base_id] = 0 # type: ignore
                chunk_id = base_id
            
            # Extract all section numbers from the chunk content
            contained_sections: List[str] = list(set(re.findall(r'Section\s+[\d]+[A-Za-z]*', chunk_text)))
            # Sort sections naturally for consistency
            contained_sections.sort(key=lambda x: int(''.join(filter(str.isdigit, x)))) # type: ignore

            chunk = LegalChunk(
                chunk_id=chunk_id,
                act_name=act_name, # type: ignore
                act_number=act_number, # type: ignore
                act_year=act_year, # type: ignore
                category=category, # type: ignore
                part=current_part, # type: ignore
                section_number=section["section_number"], # type: ignore
                section_title=section["title"], # type: ignore
                subsection=None,
                content=chunk_text,
                token_count=count_tokens(chunk_text),
                start_position=section["start"], # type: ignore
                cross_references=extract_cross_references(chunk_text),
                keywords=extract_keywords(chunk_text),
                contained_sections=contained_sections # type: ignore
            )

            # Apply title repetition for BM25 boost
            # Convert to dict, apply repetition, then back to LegalChunk
            chunk_dict: Dict[str, Any] = asdict(cast(Any, chunk))
            chunk_dict = repeat_section_title(
                chunk_dict,
                repetitions=cast(Any, config).title_repetition_count if config else 3 # type: ignore
            )
            # Re-instantiate with explicit args to maintain type safety
            chunk = LegalChunk(
                chunk_id=chunk_dict['chunk_id'],
                act_name=chunk_dict['act_name'],
                act_number=chunk_dict['act_number'],
                act_year=chunk_dict['act_year'],
                category=chunk_dict['category'],
                part=chunk_dict['part'],
                section_number=chunk_dict['section_number'],
                section_title=chunk_dict['section_title'],
                subsection=chunk_dict['subsection'],
                content=chunk_dict['content'],
                token_count=chunk_dict['token_count'],
                start_position=chunk_dict['start_position'],
                cross_references=chunk_dict['cross_references'],
                keywords=chunk_dict['keywords'],
                contained_sections=chunk_dict['contained_sections']
            )

            chunks.append(chunk)
    
    return chunks


def process_all_documents(max_tokens: int = 1000, config: Optional[RAGConfig] = None) -> Dict[str, Dict[str, int]]:
    """
    Process all documents in the processed directory and create chunks.
    
    Args:
        max_tokens: Maximum tokens per chunk.
    
    Returns:
        Dictionary mapping filenames to chunk statistics.
    """
    processed_dir = get_processed_dir()
    
    if not processed_dir.exists():
        logger.error(f"Processed directory not found: {processed_dir}")
        return {}
        
    json_files = list(processed_dir.glob("*.json"))
    
    # Filter out chunked files
    json_files = [f for f in json_files if not f.name.endswith("_chunks.json")]
    
    logger.info("=" * 60)
    logger.info("Starting Semantic Chunking")
    logger.info(f"Source directory: {processed_dir}")
    logger.info(f"Documents to process: {len(json_files)}")
    logger.info(f"Max tokens per chunk: {max_tokens}")
    logger.info("=" * 60)
    
    results: Dict[str, Dict[str, int]] = {}
    all_chunks = []
    
    for json_path in json_files:
        logger.info(f"\nProcessing: {json_path.name}")

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                document = json.load(f)

            chunks = chunk_document(document, max_tokens, config=config)
            all_chunks.extend(chunks)
            
            # Save chunks for this document
            chunks_path = json_path.with_name(json_path.stem + "_chunks.json")
            with open(chunks_path, "w", encoding="utf-8") as f:
                json.dump([asdict(cast(Any, c)) for c in chunks], f, ensure_ascii=False, indent=2)
            
            logger.info(f"Created {len(chunks)} chunks -> {chunks_path.name}")
            results[json_path.name] = {
                "chunk_count": len(chunks),
                "total_tokens": sum(c.token_count for c in chunks),
                "avg_tokens": sum(c.token_count for c in chunks) // len(chunks) if chunks else 0
            }
        except Exception as e:
            logger.error(f"Failed to process {json_path.name}: {e}")
            continue
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Chunking Summary:")
    total_chunks = 0
    for filename, stats in results.items():
        logger.info(
            f"  {filename}: {stats['chunk_count']} chunks, "
            f"avg {stats['avg_tokens']} tokens"
        )
        total_chunks += stats['chunk_count']
    logger.info(f"\nTotal chunks created: {total_chunks}")
    logger.info("=" * 60)
    
    return results


if __name__ == "__main__":
    config = RAGConfig()
    process_all_documents(max_tokens=config.chunk_size, config=config)
