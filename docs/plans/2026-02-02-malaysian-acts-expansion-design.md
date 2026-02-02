# Malaysian Legal Acts Expansion Design

**Date**: 2026-02-02
**Author**: Claude + User
**Status**: Approved

## Overview

Expand the MyLaw-RAG system from 3 Malaysian legal Acts to 11 Acts, adding comprehensive coverage across commercial, criminal, property, and civil procedure law domains.

## Current State

- **Acts**: 3 (Contracts Act 1950, Specific Relief Act 1951, Housing Development Act 1966)
- **Performance**: Hit Rate @ 1 = 95%, MRR = 0.975
- **Database**: ChromaDB with ~N chunks
- **Retrieval**: Hybrid (semantic + BM25 with RRF fusion)

## Proposed Expansion

### New Acts to Add (8 Total)

**Commercial Law** (2 Acts)
- Act 383 - Sale of Goods Act 1957
- Act 135 - Partnership Act 1961

**Criminal Law** (2 Acts)
- Act 574 - Penal Code
- Act 593 - Criminal Procedure Code

**Property Law** (2 Acts)
- Act 56 - National Land Code 1965
- Act 318 - Strata Titles Act 1985

**Civil Procedure** (2 Acts)
- Act 91 - Courts of Judicature Act 1964
- Rules of Court 2012 (special handling required)

**Total: 11 Acts** (3 existing + 8 new)

## Architecture Enhancements

### 1. AGC Scraper Enhancement

**File**: `src/ingestion/agc_scraper.py`

**Changes**:
- Update `MVP_ACTS` → `EXPANDED_ACTS` with 8 new entries
- Add `validate_pdf_download()` - check file size, detect empty PDFs
- Add `get_download_status()` - track which Acts are downloaded
- Implement incremental mode - skip already downloaded Acts
- Add progress reporting: "Processing 2/11: Sale of Goods Act 1957"
- Enhanced error handling with detailed summary

### 2. Text Extraction Enhancement

**File**: `src/ingestion/text_extractor.py`

**Changes**:
- Add `detect_pdf_type()` - distinguish text-based from scanned PDFs
- Add `validate_extraction()` - quality checks (length, section detection)
- Format-specific cleaning for different Act layouts
- Log extraction quality metrics
- Handle bilingual Acts if present

### 3. Semantic Chunker Enhancement

**File**: `src/ingestion/chunker.py`

**Changes**:
- Update `LegalChunk` dataclass with new fields:
  - `act_year: int`
  - `category: str`
  - `subsection: Optional[str]`
  - `amendment_year: Optional[int]`
  - `cross_references: List[str]`
  - `keywords: List[str]`
- Add category tagging based on Act number
- Add fallback chunking for Acts without clear section markers
- Extract and preserve cross-references
- Extract legal keywords for metadata

### 4. Configuration Update

**File**: `src/config.py`

**Add**:
```python
ACT_CATEGORIES = {
    "commercial": [135, 136, 137, 383],
    "criminal": [574, 593],
    "property": [56, 118, 318],
    "civil_procedure": [91],
}

def get_act_category(act_number: int) -> str:
    """Get category for an Act number"""
```

### 5. Batch Processing Script

**File**: `scripts/download_new_acts.py` (NEW)

**Features**:
- Process Acts by category batch
- Validation at each stage (download, extract, chunk, embed)
- Rollback capability if batch fails
- Progress reporting and logging
- Summary report after each batch

### 6. Testing & Validation

**Expanded Golden Dataset**: `tests/golden_dataset_expanded.json`

Add 25 new test questions:
- 6 commercial law questions
- 5 criminal law questions
- 5 property law questions
- 4 civil procedure questions
- 5 cross-category questions

**Category-Specific Evaluation**:

```python
TARGET_METRICS = {
    "commercial": {"hit_rate": 0.90, "mrr": 0.95},
    "criminal": {"hit_rate": 0.90, "mrr": 0.95},
    "property": {"hit_rate": 0.90, "mrr": 0.95},
    "civil_procedure": {"hit_rate": 0.85, "mrr": 0.90},
    "overall": {"hit_rate": 0.90, "mrr": 0.93},
}
```

**Regression Testing**:
- Original 20 questions must maintain Hit Rate @ 1 ≥ 90%
- Ensures new Acts don't degrade existing performance

## Implementation Phases

### Phase 0: Pre-Implementation (15 min)
- [ ] Verify current system works (pytest, evaluation)
- [ ] Backup existing data to `data_backup_YYYYMMDD/`
- [ ] Check disk space (need ~500MB)

### Phase 1: AGC Scraper Enhancement (1-2 hours)
- [ ] Update EXPANDED_ACTS list
- [ ] Add validation functions
- [ ] Add incremental mode
- [ ] Test scraper on 1 new Act

### Phase 2: Text Extraction Enhancement (1 hour)
- [ ] Add PDF type detection
- [ ] Add validation checks
- [ ] Test on variety of PDF formats

### Phase 3: Chunker Enhancement (1-2 hours)
- [ ] Update LegalChunk dataclass
- [ ] Add category tagging
- [ ] Add keyword extraction
- [ ] Test on existing Acts (regression)

### Phase 4: Configuration Update (30 min)
- [ ] Add ACT_CATEGORIES mapping
- [ ] Add get_act_category() function

### Phase 5: Execute Pipeline (2-3 hours)
- [ ] Batch 1: Commercial Acts (2)
- [ ] Batch 2: Criminal Acts (2)
- [ ] Batch 3: Property Acts (2)
- [ ] Batch 4: Civil Procedure (1-2)
- [ ] Validate after each batch

### Phase 6: Testing & Validation (1-2 hours)
- [ ] Run regression tests
- [ ] Test new Acts retrieval
- [ ] Run full evaluation
- [ ] Check per-category metrics

### Phase 7: Documentation (30 min)
- [ ] Update README.md
- [ ] Update performance metrics
- [ ] Create expansion summary doc

## File Changes Summary

| File | Type | Lines Added |
|------|------|-------------|
| `src/ingestion/agc_scraper.py` | Modify | ~80 |
| `src/ingestion/text_extractor.py` | Modify | ~60 |
| `src/ingestion/chunker.py` | Modify | ~100 |
| `src/config.py` | Modify | ~20 |
| `tests/golden_dataset.json` | Modify | ~150 |
| `src/evaluation/evaluate_rag.py` | Modify | ~40 |
| `scripts/download_new_acts.py` | New | ~120 |
| `docs/expansion_summary.md` | New | ~200 |

**Total**: ~770 lines across 8 files

## Success Criteria

1. ✅ All 11 Acts successfully downloaded and processed
2. ✅ Original Acts maintain Hit Rate @ 1 ≥ 90%
3. ✅ New Acts achieve Hit Rate @ 1 ≥ 85%
4. ✅ Overall Hit Rate @ 1 ≥ 90%
5. ✅ All categories have working retrieval
6. ✅ No regressions in existing functionality

## Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| PDF format issues for old Acts | Medium | Medium | Add OCR fallback, manual review |
| Some Acts unavailable on AGC | Low | Low | Manual download, alternate sources |
| Chunker fails on new formats | Medium | High | Fallback chunking, manual testing |
| Performance degrades | Low | High | Regression testing, incremental validation |
| ChromaDB capacity limits | Low | Medium | Monitor size, cleanup if needed |

## Next Steps

1. ✅ Design approved
2. ⏳ Create implementation worktree
3. ⏳ Execute implementation plan
4. ⏳ Testing and validation
5. ⏳ Documentation updates

---

**Design approved by user**: 2026-02-02
**Implementation start**: Pending worktree setup
