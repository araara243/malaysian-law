# PostgreSQL Setup for MyLaw-RAG

This guide covers setting up PostgreSQL with pgvector for the MyLaw-RAG system.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Ubuntu/Debian](#ubuntudebian)
  - [macOS](#macos)
- [pgvector Extension](#pgvector-extension)
- [Database Setup](#database-setup)
- [Configuration](#configuration)
- [Running the Migration](#running-the-migration)
- [Validation](#validation)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Python 3.10+
- PostgreSQL 14+ (pgvector requires PostgreSQL 11+)
- pip

## Installation

### Ubuntu/Debian

```bash
# Install PostgreSQL
sudo apt update
sudo apt install postgresql postgresql-contrib

# Install development headers (required for psycopg2)
sudo apt install libpq-dev

# Start PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

### macOS

```bash
# Install PostgreSQL using Homebrew
brew install postgresql@16

# Start PostgreSQL service
brew services start postgresql@16
```

## pgvector Extension

pgvector is required for vector similarity search.

### Install from Source (Recommended)

```bash
# Clone pgvector repository
git clone --branch v0.7.4 https://github.com/pgvector/pgvector.git
cd pgvector

# Build and install
make
sudo make install

# Restart PostgreSQL to load the extension
# Ubuntu/Debian:
sudo systemctl restart postgresql
# macOS:
brew services restart postgresql@16
```

### Verify Installation

```bash
# Connect to PostgreSQL
sudo -u postgres psql

# In psql, check if pgvector is available
SELECT * FROM pg_available_extensions WHERE name = 'vector';
```

Expected output:
```
  name  | default_version | installed_version |        comment
--------+-----------------+-------------------+------------------------
 vector | 0.7.4           |                   | vector data type and ...
```

## Database Setup

### Create Database and User

```bash
# Connect to PostgreSQL as postgres user
sudo -u postgres psql

# Run these commands in psql:
```

```sql
-- Create database
CREATE DATABASE mylaw_rag;

-- Create user with password
CREATE USER mylaw_user WITH PASSWORD 'your_secure_password';

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE mylaw_rag TO mylaw_user;

-- Connect to the database
\c mylaw_rag

-- Grant schema privileges
GRANT ALL ON SCHEMA public TO mylaw_user;

-- Exit psql
\q
```

### Initialize Schema

```bash
# From the project root, run the schema initialization
psql -U mylaw_user -d mylaw_rag -f scripts/migrate/init_postgres_schema.sql
```

Expected output:
```
CREATE EXTENSION
CREATE TABLE
CREATE TABLE
CREATE TABLE
CREATE TABLE
CREATE INDEX
CREATE INDEX
```

### Verify Tables

```bash
psql -U mylaw_user -d mylaw_rag -c "\dt"
```

Expected output:
```
          List of relations
 Schema |    Name     | Type  |   Owner
--------+-------------+-------+--------------
 public | acts        | table | mylaw_user
 public | chunks      | table | mylaw_user
 public | embeddings  | table | mylaw_user
 public | sections    | table | mylaw_user
```

## Configuration

### Set Environment Variables

Create or update `.env` file in the project root:

```bash
# PostgreSQL Configuration
PGHOST=localhost
PGPORT=5432
PGDATABASE=mylaw_rag
PGUSER=mylaw_user
PGPASSWORD=your_secure_password
```

### Verify Connection

```bash
python -c "
from src.config import postgresql_config
from src.db.postgres_connection import PostgreSQLConnectionManager

conn_mgr = PostgreSQLConnectionManager(postgresql_config)
conn_mgr.initialize()

with conn_mgr.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute('SELECT version()')
    print('Connected to:', cursor.fetchone()[0])

conn_mgr.close()
print('Connection successful!')
"
```

## Running the Migration

### Step 1: Export from ChromaDB

```bash
python scripts/migrate/export_from_chroma.py chromadb_export.json
```

Expected output:
```
Exporting 1234 chunks from ChromaDB...
Export complete: chromadb_export.json
- Chunks: 1234
- Embeddings: 1234
- Acts: 3
```

### Step 2: Import to PostgreSQL

```bash
python scripts/migrate/import_to_postgres.py chromadb_export.json
```

Expected output:
```
Importing 1234 chunks to PostgreSQL...
Import complete: {'acts_inserted': 3, 'sections_inserted': 456, 'chunks_inserted': 1234, 'embeddings_inserted': 1234}
```

### Step 3: Validate Migration

```bash
python scripts/migrate/validate_migration.py chromadb_export.json
```

Expected output:
```
============================================================
PostgreSQL Migration Validation
============================================================

1. Validating data integrity...
   Status: PASS
   ChromaDB: 1234 chunks
   PostgreSQL: 1234 chunks

2. Validating embeddings...
   Status: PASS
   Export: 1234 embeddings
   PostgreSQL: 1234 embeddings

3. Validating retrieval quality...
   Status: PASS
   Hit Rate: 95.0%
   Hits: 19/20

============================================================
All validations PASSED!
```

## Validation

### Create Vector Index (After Migration)

For optimal performance on large datasets, create an ivfflat index:

```sql
-- Connect to database
psql -U mylaw_user -d mylaw_rag

-- Create index (requires at least 1000 rows)
CREATE INDEX idx_embeddings_embedding
ON embeddings
USING ivfflat(embedding vector_cosine_ops)
WITH (lists = 100);

-- Analyze table for query planner
ANALYZE embeddings;
```

### Test Retrieval

```bash
python -c "
from src.config import postgresql_config
from src.retrieval.postgresql_retriever import PostgreSQLRetriever

retriever = PostgreSQLRetriever(postgresql_config)
results = retriever.retrieve('What is consideration?', n_results=3)

for i, r in enumerate(results, 1):
    print(f'{i}. {r.act_name} - Section {r.section_number}')
    print(f'   Score: {r.score:.3f}')

retriever.close()
"
```

## Troubleshooting

### Connection Refused

**Error:** `connection refused` or `could not connect to server`

**Solution:**
```bash
# Check if PostgreSQL is running
sudo systemctl status postgresql

# Start if not running
sudo systemctl start postgresql

# Check port
sudo netstat -tlnp | grep 5432
```

### Authentication Failed

**Error:** `FATAL: password authentication failed`

**Solution:**
```bash
# Reset password
sudo -u postgres psql
ALTER USER mylaw_user WITH PASSWORD 'new_password';
\q

# Update .env with new password
```

### pgvector Not Available

**Error:** `could not open extension control file`

**Solution:**
```bash
# Verify pgvector installation
sudo -u postgres psql
SHOW shared_preload_libraries;

# If pgvector not listed, add to postgresql.conf:
# sudo nano /etc/postgresql/16/main/postgresql.conf
# Add: shared_preload_libraries = 'vector'

# Restart PostgreSQL
sudo systemctl restart postgresql
```

### Permission Denied on Tables

**Error:** `permission denied for table acts`

**Solution:**
```bash
sudo -u postgres psql -d mylaw_rag

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mylaw_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO mylaw_user;
\q
```

### Import Fails with Unique Constraint

**Error:** `duplicate key value violates unique constraint`

**Solution:**
```bash
# Clear database and re-import
psql -U mylaw_user -d mylaw_rag -c "
TRUNCATE TABLE embeddings CASCADE;
TRUNCATE TABLE chunks CASCADE;
TRUNCATE TABLE sections CASCADE;
TRUNCATE TABLE acts CASCADE;
"

# Re-run import
python scripts/migrate/import_to_postgres.py chromadb_export.json
```

### Slow Query Performance

**Solution:**
```sql
-- Check if index exists
SELECT indexname FROM pg_indexes WHERE tablename = 'embeddings';

-- Create index if missing
CREATE INDEX idx_embeddings_embedding
ON embeddings
USING ivfflat(embedding vector_cosine_ops)
WITH (lists = 100);

-- Update statistics
ANALYZE embeddings;
```

## Performance Tips

1. **Connection Pooling**: Use the built-in connection pooling (default: 1-5 connections)
2. **Vector Indexing**: Create ivfflat index for datasets >1000 rows
3. **Batch Operations**: Import data in batches for large datasets
4. **Query Limits**: Use `n_results` parameter to limit returned results

## Next Steps

- Configure application to use `PostgreSQLRetriever` instead of `HybridRetriever`
- Monitor query performance and optimize indexes
- Set up automated backups of PostgreSQL database
- Consider read replicas for high-traffic deployments
