"""Test PostgreSQL connection manager."""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from db.postgres_connection import PostgreSQLConnectionManager


def test_connection_manager_initialization():
    """Test that connection manager initializes with config."""
    from config import postgresql_config

    manager = PostgreSQLConnectionManager(postgresql_config)

    assert manager.config == postgresql_config
    assert manager._pool is None


def test_get_connection_string():
    """Test connection string generation."""
    from config import postgresql_config

    manager = PostgreSQLConnectionManager(postgresql_config)

    conn_str = manager._get_connection_string()

    assert "host=" in conn_str
    assert "port=" in conn_str
    assert "dbname=" in conn_str
    assert "user=" in conn_str


@patch("db.postgres_connection.psycopg2.pool.ThreadedConnectionPool")
def test_initialize_pool(mock_pool):
    """Test connection pool initialization."""
    from config import postgresql_config

    manager = PostgreSQLConnectionManager(postgresql_config)
    manager.initialize()

    mock_pool.assert_called_once()
    assert manager._pool is not None


@patch("db.postgres_connection.psycopg2.pool.ThreadedConnectionPool")
def test_get_connection(mock_pool):
    """Test getting connection from pool."""
    from config import postgresql_config

    # Setup mock
    mock_conn = MagicMock()
    mock_pool_instance = MagicMock()
    mock_pool_instance.getconn.return_value = mock_conn
    mock_pool.return_value = mock_pool_instance

    manager = PostgreSQLConnectionManager(postgresql_config)
    manager.initialize()

    with manager.get_connection() as conn:
        assert conn == mock_conn


@patch("db.postgres_connection.psycopg2.pool.ThreadedConnectionPool")
def test_close_pool(mock_pool):
    """Test closing connection pool."""
    from config import postgresql_config

    mock_pool_instance = MagicMock()
    mock_pool.return_value = mock_pool_instance

    manager = PostgreSQLConnectionManager(postgresql_config)
    manager.initialize()
    manager.close()

    mock_pool_instance.closeall.assert_called_once()
