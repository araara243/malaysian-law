"""
PostgreSQL connection management with pooling.

This module provides a connection pool manager for PostgreSQL database
connections using psycopg2. Connection pooling improves performance by
reusing connections across multiple requests.
"""

import logging
from contextlib import contextmanager
from typing import Optional

import psycopg2
from psycopg2 import pool

from config import PostgreSQLConfig, setup_logging

logger = setup_logging(__name__)


class PostgreSQLConnectionManager:
    """
    Manages PostgreSQL connection pool.

    Provides thread-safe connection pooling for database operations.
    Uses context managers for automatic connection cleanup.
    """

    def __init__(self, config: PostgreSQLConfig):
        """
        Initialize connection manager.

        Args:
            config: PostgreSQL configuration settings.
        """
        self.config = config
        self._pool: Optional[pool.ThreadedConnectionPool] = None

    def _get_connection_string(self) -> str:
        """
        Generate PostgreSQL connection string.

        Returns:
            Connection string for psycopg2.
        """
        return (
            f"host={self.config.host} "
            f"port={self.config.port} "
            f"dbname={self.config.database} "
            f"user={self.config.user} "
            f"password={self.config.password}"
        )

    def initialize(self) -> None:
        """
        Initialize connection pool.

        Creates a threaded connection pool with min/max connections
        specified in configuration.
        """
        try:
            conn_str = self._get_connection_string()

            self._pool = pool.ThreadedConnectionPool(
                minconn=self.config.min_connections,
                maxconn=self.config.max_connections,
                dsn=conn_str
            )

            logger.info(
                f"Initialized connection pool: "
                f"{self.config.min_connections}-{self.config.max_connections} connections"
            )
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """
        Get connection from pool.

        Yields a connection object and automatically returns it to pool.

        Yields:
            psycopg2 connection object.

        Raises:
            RuntimeError: If pool not initialized.
        """
        if not self._pool:
            raise RuntimeError("Connection pool not initialized. Call initialize() first.")

        conn = None
        try:
            conn = self._pool.getconn()
            yield conn
        except Exception as e:
            logger.error(f"Error using connection: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self._pool.putconn(conn)

    def close(self) -> None:
        """Close all connections in pool."""
        if self._pool:
            self._pool.closeall()
            self._pool = None
            logger.info("Connection pool closed")
