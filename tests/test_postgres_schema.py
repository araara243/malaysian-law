#!/usr/bin/env python3
"""
Tests for PostgreSQL schema validation
"""

import unittest
import re
from pathlib import Path


class TestPostgreSQLSchema(unittest.TestCase):
    """Test the PostgreSQL schema SQL file"""

    def setUp(self):
        """Set up test fixtures"""
        # Schema file is in scripts/migrate relative to project root
        project_root = Path(__file__).parent.parent
        self.schema_file = project_root / "scripts" / "migrate" / "init_postgres_schema.sql"

        if not self.schema_file.exists():
            self.skipTest(f"Schema file not found: {self.schema_file}")

        with open(self.schema_file, 'r') as f:
            self.sql_content = f.read()

    def test_pgvector_extension_enabled(self):
        """Test that pgvector extension is enabled"""
        self.assertIn(
            'CREATE EXTENSION IF NOT EXISTS vector',
            self.sql_content,
            "pgvector extension should be enabled"
        )

    def test_acts_table_exists(self):
        """Test that acts table is created"""
        self.assertIn(
            'CREATE TABLE IF NOT EXISTS acts',
            self.sql_content,
            "acts table should be created"
        )

    def test_sections_table_exists(self):
        """Test that sections table is created"""
        self.assertIn(
            'CREATE TABLE IF NOT EXISTS sections',
            self.sql_content,
            "sections table should be created"
        )

    def test_chunks_table_exists(self):
        """Test that chunks table is created"""
        self.assertIn(
            'CREATE TABLE IF NOT EXISTS chunks',
            self.sql_content,
            "chunks table should be created"
        )

    def test_embeddings_table_exists(self):
        """Test that embeddings table is created"""
        self.assertIn(
            'CREATE TABLE IF NOT EXISTS embeddings',
            self.sql_content,
            "embeddings table should be created"
        )

    def test_acts_table_columns(self):
        """Test that acts table has required columns"""
        required_columns = [
            'act_number INTEGER UNIQUE NOT NULL',
            'act_name VARCHAR(255) NOT NULL',
            'act_year INTEGER NOT NULL',
            'category VARCHAR(50) NOT NULL',
            'language VARCHAR(10) DEFAULT \'EN\'',
            'source_url TEXT'
        ]

        for col in required_columns:
            self.assertIn(col, self.sql_content, f"acts table should have column: {col}")

    def test_chunks_table_columns(self):
        """Test that chunks table has required columns"""
        required_columns = [
            'chunk_content TEXT NOT NULL',
            'token_count INTEGER NOT NULL',
            'start_position INTEGER NOT NULL',
            'subsection VARCHAR(100)',
            'keywords TEXT[]'
        ]

        for col in required_columns:
            self.assertIn(col, self.sql_content, f"chunks table should have column: {col}")

    def test_vector_column_dimension(self):
        """Test that embedding vector has correct dimension (384)"""
        self.assertIn(
            'VECTOR(384)',
            self.sql_content,
            "embedding column should be VECTOR(384) for all-MiniLM-L6-v2"
        )

    def test_cascade_delete_constraints(self):
        """Test that foreign key constraints use CASCADE delete"""
        # Count occurrences of CASCADE delete
        cascade_count = self.sql_content.count('ON DELETE CASCADE')

        # Should have at least 3 CASCADE deletes (sections->acts, chunks->sections, embeddings->chunks)
        self.assertGreaterEqual(
            cascade_count,
            3,
            f"Should have at least 3 CASCADE delete constraints, found {cascade_count}"
        )

    def test_validation_constraints(self):
        """Test that validation constraints exist"""
        self.assertIn(
            'CONSTRAINT valid_act_number CHECK (act_number > 0)',
            self.sql_content,
            "Should have valid_act_number constraint"
        )

        self.assertIn(
            'CONSTRAINT valid_token_count CHECK (token_count > 0)',
            self.sql_content,
            "Should have valid_token_count constraint"
        )

    def test_indexes_exist(self):
        """Test that performance indexes are created"""
        indexes = [
            'idx_chunks_section_id',
            'idx_embeddings_chunk_id'
        ]

        for idx in indexes:
            self.assertIn(
                f'CREATE INDEX IF NOT EXISTS {idx}',
                self.sql_content,
                f"Index {idx} should be created"
            )

    def test_ivfflat_comment_present(self):
        """Test that ivfflat index documentation is present"""
        self.assertIn(
            'ivfflat',
            self.sql_content,
            "Should include ivfflat index documentation"
        )

    def test_embedding_model_default(self):
        """Test that embedding_model has correct default value"""
        self.assertIn(
            "DEFAULT 'all-MiniLM-L6-v2'",
            self.sql_content,
            "embedding_model should default to 'all-MiniLM-L6-v2'"
        )

    def test_foreign_key_relationships(self):
        """Test that foreign key relationships are defined"""
        # sections should reference acts
        self.assertIn(
            'act_id INTEGER REFERENCES acts(id) ON DELETE CASCADE',
            self.sql_content,
            "sections.act_id should reference acts(id)"
        )

        # chunks should reference sections
        self.assertIn(
            'section_id INTEGER REFERENCES sections(id) ON DELETE CASCADE',
            self.sql_content,
            "chunks.section_id should reference sections(id)"
        )

        # embeddings should reference chunks
        self.assertIn(
            'chunk_id INTEGER REFERENCES chunks(id) ON DELETE CASCADE',
            self.sql_content,
            "embeddings.chunk_id should reference chunks(id)"
        )

    def test_unique_constraints(self):
        """Test that appropriate unique constraints exist"""
        # act_number should be unique in acts table
        self.assertIn(
            'act_number INTEGER UNIQUE',
            self.sql_content,
            "act_number should be UNIQUE in acts table"
        )

        # sections should have unique constraint on act_id, section_number, section_order
        self.assertIn(
            'UNIQUE(act_id, section_number, section_order)',
            self.sql_content,
            "sections should have UNIQUE constraint on (act_id, section_number, section_order)"
        )


if __name__ == '__main__':
    unittest.main()
