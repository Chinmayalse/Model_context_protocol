import psycopg2
from psycopg2.extras import DictCursor
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Database connection parameters
DB_NAME = "mcp_database"
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = "localhost"
DB_PORT = "5432"

def get_db_connection():
    """Get a database connection."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        raise

def check_batch_jobs_table():
    """Check the batch_jobs table schema."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        # Check table columns
        cursor.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'batch_jobs'
            ORDER BY ordinal_position
        """)
        
        columns = cursor.fetchall()
        
        print("Batch_jobs table schema:")
        for col in columns:
            print(f"- {col['column_name']}: {col['data_type']} (Nullable: {col['is_nullable']})")
        
        # Get a sample row
        cursor.execute("SELECT * FROM batch_jobs LIMIT 1")
        row = cursor.fetchone()
        
        if row:
            print("\nSample row:")
            for col_name, value in row.items():
                print(f"- {col_name}: {value}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error checking batch_jobs table: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    check_batch_jobs_table()
