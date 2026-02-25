-- PostgreSQL Schema for MyLaw-RAG
-- This script creates the normalized schema for legal documents

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Acts table: Metadata about Malaysian legal Acts
CREATE TABLE IF NOT EXISTS acts (
    id SERIAL PRIMARY KEY,
    act_number INTEGER UNIQUE NOT NULL,
    act_name VARCHAR(255) NOT NULL,
    act_year INTEGER NOT NULL,
    category VARCHAR(50) NOT NULL,
    language VARCHAR(10) DEFAULT 'EN',
    source_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_act_number CHECK (act_number > 0)
);

-- Sections table: Legal sections within Acts
CREATE TABLE IF NOT EXISTS sections (
    id SERIAL PRIMARY KEY,
    act_id INTEGER REFERENCES acts(id) ON DELETE CASCADE,
    section_number VARCHAR(50) NOT NULL,
    section_title TEXT,
    part VARCHAR(255),
    section_order INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(act_id, section_number, section_order)
);

-- Chunks table: Text chunks with metadata
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    section_id INTEGER REFERENCES sections(id) ON DELETE CASCADE,
    chunk_content TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    start_position INTEGER NOT NULL,
    subsection VARCHAR(100),
    keywords TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_token_count CHECK (token_count > 0)
);

-- Embeddings table: Vector embeddings with pgvector
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    chunk_id INTEGER REFERENCES chunks(id) ON DELETE CASCADE,
    embedding VECTOR(384) NOT NULL,
    embedding_model VARCHAR(100) DEFAULT 'all-MiniLM-L6-v2',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_chunks_section_id ON chunks(section_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id ON embeddings(chunk_id);

-- Note: ivfflat index requires data to exist first
-- Create after migration: CREATE INDEX idx_embeddings_embedding ON embeddings USING ivfflat(embedding vector_cosine_ops) WITH (lists = 100);

-- Grant permissions (adjust user as needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO your_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO your_user;
