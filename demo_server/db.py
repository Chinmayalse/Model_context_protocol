"""
Database connection module for the MCP application.
"""

import logging
import psycopg2
from psycopg2.extras import Json, DictCursor
import os
from contextlib import contextmanager

# Database connection parameters
DB_NAME = "mcp_database"
DB_USER = "postgres"  # Default PostgreSQL user, change if needed
DB_PASSWORD = "postgres"  # Default password, change to match your PostgreSQL setup
DB_HOST = "localhost"
DB_PORT = "5432"

# For initial connection to postgres database (to create our database)
POSTGRES_DB = "postgres"

@contextmanager
def get_db_connection():
    """
    Context manager for database connections.
    Usage:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM users")
    """
    conn = None
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        yield conn
    except Exception as e:
        print(f"Database connection error: {e}")
        raise
    finally:
        if conn is not None:
            conn.close()

@contextmanager
def get_db_cursor(commit=False, cursor_factory=None):
    """
    Context manager for database cursors.
    
    Args:
        commit: If True, commit the transaction when done
        cursor_factory: Optional cursor factory (e.g., DictCursor)
    """
    with get_db_connection() as conn:
        try:
            # Create cursor with the specified factory
            cursor = conn.cursor(cursor_factory=cursor_factory) if cursor_factory else conn.cursor()
            try:
                yield cursor
                if commit:
                    conn.commit()
            except Exception as e:
                conn.rollback()
                logger = logging.getLogger(__name__)
                logger.error(f"Database cursor error: {e}", exc_info=True)
                raise
            finally:
                cursor.close()
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error creating database cursor: {e}", exc_info=True)
            raise

def test_connection():
    """Test the database connection."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                version = cur.fetchone()[0]
                print(f"Connected to PostgreSQL: {version}")
                return True
    except Exception as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return False

if __name__ == "__main__":
    # Test the connection when this module is run directly
    test_connection()
