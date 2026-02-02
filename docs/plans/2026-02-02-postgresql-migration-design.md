# PostgreSQL Migration Design for MyLaw-RAG

**Date**: 2026-02-02
**Author**: Claude + User
**Status**: Approved

## Overview

Migrate the Malaysian Legal RAG system from ChromaDB to PostgreSQL with pgvector extension, using a **hybrid approach**:
- **New Acts**: Use PostgreSQL (validates new architecture)
- **Existing Acts**: Keep ChromaDB initially (minimize risk)
- **Long-term**: Fully migrate all Acts to PostgreSQL

### Why Hybrid Strategy?

1. **Safety**: Existing 7 Acts continue working without interruption
2. **Validation**: Test PostgreSQL thoroughly with new Acts before migrating existing data
3. **Learning Curve**: Learn PostgreSQL/pgvector in production with low stakes
4. **Rollback**: If issues arise, new Acts can switch back to ChromaDB

### User Choices Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Strategy** | Hybrid (new → PostgreSQL, existing → ChromaDB) | Safest approach |
| **Schema** | Normalized (acts, sections, chunks, embeddings) | Clean, queryable |
| **Deployment** | Native system installation | Full control, no container overhead |
| **Vector Storage** | Native pgvector (VECTOR column) | Best performance |
| **Connection** | Connection pool (psycopg2.pool) | Standard practice, efficient |

---

## Database Schema

### Normalized Schema Structure

```sql
-- Acts table: Metadata about Malaysian legal Acts
CREATE TABLE acts (
    id SERIAL PRIMARY KEY,
    act_number INTEGER UNIQUE NOT NULL,
    act_name VARCHAR(255) NOT NULL,
    act_year INTEGER NOT NULL,
    category VARCHAR(50) NOT NULL,
    language VARCHAR(10) DEFAULT 'EN',
    source_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sections table: Legal sections within Acts
CREATE TABLE sections (
    id SERIAL PRIMARY KEY,
    act_id INTEGER REFERENCES acts(id) ON DELETE CASCADE,
    section_number VARCHAR(50) NOT NULL,
    section_title TEXT,
    part VARCHAR(255),
    section_order INTEGER,
    UNIQUE(act_id, section_number, section_order)
);

-- Chunks table: Text chunks with metadata
CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    section_id INTEGER REFERENCES sections(id) ON DELETE CASCADE,
    chunk_content TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    start_position INTEGER NOT NULL,
    subsection VARCHAR(100),
    keywords TEXT[], -- Array of keywords for metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Embeddings table: Vector embeddings with pgvector
CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    chunk_id INTEGER REFERENCES chunks(id) ON DELETE CASCADE,
    embedding VECTOR(384) NOT NULL, -- Using all-MiniLM-L6-v2 (384 dims)
    embedding_model VARCHAR(100) DEFAULT 'all-MiniLM-L6-v2',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_chunks_section_id ON chunks(section_id);
CREATE INDEX idx_embeddings_chunk_id ON embeddings(chunk_id);
CREATE INDEX idx_embeddings_embedding ON embeddings USING ivfflat(embedding vector_cosine_ops);
```

### Key Design Decisions

1. **Normalized structure**: Acts → Sections → Chunks → Embeddings mirrors legal document hierarchy
2. **Foreign keys with CASCADE**: Deleting an Act automatically cleans up all related data
3. **Native pgvector**: `VECTOR(384)` type for efficient similarity search
4. **ivfflat index**: Fast cosine similarity search (what ChromaDB uses)
5. **Keywords array**: PostgreSQL array type stores keyword metadata

### Relationships

```
acts (1) → (many) sections (1) → (many) chunks (1) → (1) embedding
```

---

## Migration Process

### Migration Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    MIGRATION STAGES                               │
└─────────────────────────────────────────────────────────────────┘

Stage 1: PostgreSQL Setup
├─ Install PostgreSQL + pgvector extension
├─ Create database: mylaw_rag
├─ Create schema (4 tables + indexes)
└─ Verify installation

Stage 2: Data Export from ChromaDB
├─ Connect to existing ChromaDB collection
├─ Export chunks with embeddings and metadata
├─ Parse Act/Section information from metadata
└─ Create export JSON files

Stage 3: Data Import to PostgreSQL
├─ Import acts metadata → acts table
├─ Import sections → sections table
├─ Import chunks → chunks table
├─ Import embeddings → embeddings table
└─ Commit transaction

Stage 4: Application Layer Updates
├─ Create PostgreSQLRetriever class
├─ Update RAG chain to use PostgreSQL
├─ Add configuration for PostgreSQL connection
└─ Test retrieval with new Acts

Stage 5: Validation & Testing
├─ Run tests against golden dataset
├─ Compare PostgreSQL vs ChromaDB retrieval quality
├─ Performance benchmark
└─ Verify hit rates, MRR maintained
```

### Migration Script Flow

```python
# Export from ChromaDB
def export_from_chroma():
    # 1. Connect to ChromaDB
    # 2. Get all chunks with embeddings and metadata
    # 3. Parse Act, Section from metadata
    # 4. Create export JSON
    # 5. Save to data/export/

# Import to PostgreSQL
def import_to_postgresql():
    # 1. Connect to PostgreSQL
    # 2. Read export JSON
    # 3. Insert acts → acts table
    # 4. Insert sections → sections table
    # 5. Insert chunks → chunks table
    # 6. Insert embeddings → embeddings table
    # 7. Commit transaction
```

---

## Application Layer

### Application Architecture

```
Streamlit UI
    │
    ├─> LegalRAGChain (orchestrates retrieval + generation)
    │
    └─> Retriever Interface
           │
           ├─> HybridRetriever (ChromaDB - existing)
           │
           └─> PostgreSQLRetriever (PostgreSQL - NEW)
```

### New Component: PostgreSQLRetriever

**File**: `src/retrieval/postgresql_retriever.py`

Key features:
- Connection pooling (psycopg2.pool)
- Vector similarity search with pgvector
- Compatible interface with HybridRetriever
- Returns RetrievalResult objects (same format)
- Error handling and logging

### Configuration Updates

**File**: `src/config.py`

```python
# PostgreSQL Configuration
@dataclass
class PostgreSQLConfig:
    """PostgreSQL database connection settings."""

    # Connection
    host: str = "localhost"
    port: int = 5432
    database: str = "mylaw_rag"
    user: str = "postgres"
    password: str = ""

    # Pool settings
    min_connections: int = 1
    max_connections: int = 5

    # Vector search
    embedding_dimension: int = 384  # all-MiniLM-L6-v2
    vector_type: str = "vector_cosine_ops"  # pgvector operator

# Add to config
postgresql_config: PostgreSQLConfig = PostgreSQLConfig()
```

### Environment Variables

**File**: `.env`

```bash
# PostgreSQL Configuration
PGHOST=localhost
PGPORT=5432
PGDATABASE=mylaw_rag
PGUSER=postgres
PGPASSWORD=your_password_here
```

---

## Testing & Validation

### Testing Strategy

```
Phase 1: Unit Tests
├─ Test PostgreSQLRetriever class
├─ Test vector similarity search
├─ Test database connection pooling
└─ Test query embedding generation

Phase 2: Integration Tests
├─ Test retrieval with new Acts (PostgreSQL)
├─ Compare results: ChromaDB vs PostgreSQL
├─ Validate retrieval quality metrics
└─ Ensure no regressions

Phase 3: Performance Tests
├─ Benchmark: PostgreSQL vs ChromaDB retrieval speed
├─ Concurrent connection testing
├─ Memory usage comparison
└─ Query execution time analysis

Phase 4: Data Validation
├─ Verify all chunks migrated successfully
├─ Check embedding integrity
├─ Validate foreign key constraints
└─ Confirm metadata accuracy
```

### Success Criteria

Migration considered successful when:

1. ✅ **Data Integrity**: All 100% of chunks migrated with correct embeddings
2. ✅ **Retrieval Quality**: PostgreSQL achieves ≥95% of ChromaDB hit rate on same queries
3. ✅ **Performance**: PostgreSQL retrieval time ≤ 1.5x ChromaDB time
4. ✅ **Stability**: Connection pool handles concurrent requests without errors
5. ✅ **Functionality**: All application features work with PostgreSQL

### Test Scenarios

**Scenario 1: Retrieval Accuracy Test**
```python
def test_postgresql_retrieval_quality():
    """Test that PostgreSQL retrieval matches ChromaDB quality."""
    golden_questions = [
        "What is consideration in contract law?",
        "Explain specific performance obligations.",
    ]

    chroma_results = test_retriever(chroma_retriever, golden_questions)
    chroma_hit_rate = calculate_hit_rate(chroma_results)

    postgresql_results = test_retriever(postgresql_retriever, golden_questions)
    postgresql_hit_rate = calculate_hit_rate(postgresql_results)

    assert postgresql_hit_rate >= chroma_hit_rate * 0.95
```

**Scenario 2: Data Integrity Test**
```python
def test_data_migration():
    """Verify all data migrated correctly."""
    chroma_count = chroma_collection.count()

    with pg_pool.conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM chunks")
        pg_count = cur.fetchone()[0]

    assert pg_count == chroma_count
```

**Scenario 3: Performance Benchmark**
```python
def test_performance_comparison():
    """Compare retrieval speed."""
    chroma_time = benchmark(chroma_retriever, queries)
    postgresql_time = benchmark(postgresql_retriever, queries)

    assert postgresql_time <= chroma_time * 1.5
```

**Scenario 4: Connection Pool Stress Test**
```python
def test_connection_pool():
    """Test connection pool under load."""
    # Spawn 5 concurrent threads
    # Each performs 10 retrievals
    # All should succeed without errors
```

---

## Implementation Plan

### Implementation Timeline

```
Week 1: Foundation (Days 1-3)
├─ Day 1: Install PostgreSQL + pgvector
├─ Day 2: Create database schema + indexes
├─ Day 3: Create PostgreSQLRetriever class (basic version)
└─ Deliverable: PostgreSQL running with empty tables

Week 2: Migration Tooling (Days 4-7)
├─ Day 4: Create export script (ChromaDB → JSON)
├─ Day 5: Create import script (JSON → PostgreSQL)
├─ Day 6: Test migration with small dataset
├─ Day 7: Fix issues and refine scripts
└─ Deliverable: Working migration pipeline

Week 3: Application Integration (Days 8-10)
├─ Day 8: Update config with PostgreSQL settings
├─ Day 9: Integrate PostgreSQLRetriever into RAG chain
├─ Day 10: Add connection pool management
└─ Deliverable: Application using PostgreSQL for new Acts

Week 4: Testing & Validation (Days 11-14)
├─ Day 11-12: Run full test suite, compare results
├─ Day 13: Performance benchmarking, optimize queries
├─ Day 14: Fix remaining issues
└─ Deliverable: PostgreSQL validated and ready

Week 5: Production Migration (Days 15-18)
├─ Migrate existing Acts from ChromaDB to PostgreSQL
├─ Final validation, documentation
└─ Complete: All Acts using PostgreSQL
```

### File Changes Summary

| **New Files** | **Modified Files** |
|----------------|-------------------|
| `src/retrieval/postgresql_retriever.py` | `src/config.py` (add PostgreSQL config) |
| `scripts/migrate/export_from_chroma.py` | `src/retrieval/hybrid_retriever.py` (maybe add fallback) |
| `scripts/migrate/import_to_postgres.py` | `.env` (add PostgreSQL credentials) |
| `scripts/migrate/validate_migration.py` | `requirements.txt` (add psycopg2-binary) |
| `tests/test_postgresql_retriever.py` | `README.md` (add PostgreSQL setup docs) |
| `docs/plans/YYYY-MM-DD-postgresql-migration-design.md` | |

### Dependencies to Add

**requirements.txt additions**:
```
psycopg2-binary>=2.9.0           # PostgreSQL adapter with pgvector support
psycopg2-pool>=2.0.0             # Connection pooling
```

---

## Risk Mitigation

| **Risk** | **Mitigation** |
|----------|-------------|
| PostgreSQL installation fails | Use Docker fallback, provide detailed setup guide |
| Migration data corruption | Keep ChromaDB backup, validate before committing |
| Performance degradation | Connection pooling, query optimization, indexes |
| Breaking existing functionality | Keep ChromaDB retriever as fallback, gradual rollout |
| pgvector compatibility issues | Test with small dataset first, use standard operations |

---

## Success Criteria

Migration considered successful when:

- ✅ All chunks migrated (100% data integrity)
- ✅ PostgreSQL hit rate ≥95% of ChromaDB hit rate
- ✅ PostgreSQL retrieval time ≤1.5x ChromaDB time
- ✅ Connection pool handles 5+ concurrent requests
- ✅ All tests pass with PostgreSQL backend
- ✅ Documentation updated with PostgreSQL setup

---

## Next Steps After Design Approval

1. ✅ Design document created and committed
2. **Set up implementation worktree** (using git worktree)
3. **Write detailed implementation plan** (subagent-driven development)
4. **Execute in phases** following the timeline above
5. **Test thoroughly** at each phase
6. **Document learnings** and update README

---

**Design Status**: ✅ COMPLETE
**Ready for**: Implementation worktree setup and detailed planning

**Total Estimated Time**: 3-4 weeks for full migration
**Files to Create**: 7 new files
**Files to Modify**: 5 existing files
**Dependencies to Add**: 2 packages
