import psycopg2
from psycopg2.extras import DictCursor
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('schema_fix.log')
    ]
)
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

def check_column_exists(table, column):
    """Check if a column exists in a table."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = %s AND column_name = %s
        """, (table, column))
        
        exists = cursor.fetchone() is not None
        
        cursor.close()
        conn.close()
        
        return exists
    except Exception as e:
        logger.error(f"Error checking if column exists: {e}")
        return False

def fix_batch_jobs_table():
    """Fix the batch_jobs table schema."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if the processed column exists
        if not check_column_exists('batch_jobs', 'processed'):
            print("Adding 'processed' column to batch_jobs table...")
            cursor.execute("""
                ALTER TABLE batch_jobs 
                ADD COLUMN processed INTEGER DEFAULT 0
            """)
            conn.commit()
            print("Added 'processed' column")
        else:
            print("'processed' column already exists")
            
        # Check if the failed column exists
        if not check_column_exists('batch_jobs', 'failed'):
            print("Adding 'failed' column to batch_jobs table...")
            cursor.execute("""
                ALTER TABLE batch_jobs 
                ADD COLUMN failed INTEGER DEFAULT 0
            """)
            conn.commit()
            print("Added 'failed' column")
        else:
            print("'failed' column already exists")
            
        # Check if the total_files column exists
        if not check_column_exists('batch_jobs', 'total_files'):
            print("Adding 'total_files' column to batch_jobs table...")
            cursor.execute("""
                ALTER TABLE batch_jobs 
                ADD COLUMN total_files INTEGER DEFAULT 0
            """)
            conn.commit()
            print("Added 'total_files' column")
        else:
            print("'total_files' column already exists")
            
        # Check if processed_files exists but processed doesn't
        if check_column_exists('batch_jobs', 'processed_files') and not check_column_exists('batch_jobs', 'processed'):
            print("Copying data from 'processed_files' to 'processed'...")
            cursor.execute("""
                ALTER TABLE batch_jobs 
                ADD COLUMN processed INTEGER DEFAULT 0
            """)
            cursor.execute("""
                UPDATE batch_jobs 
                SET processed = processed_files
            """)
            conn.commit()
            print("Copied data from 'processed_files' to 'processed'")
            
        # Check if success_count exists but processed doesn't
        if check_column_exists('batch_jobs', 'success_count') and not check_column_exists('batch_jobs', 'processed'):
            print("Copying data from 'success_count' to 'processed'...")
            cursor.execute("""
                ALTER TABLE batch_jobs 
                ADD COLUMN processed INTEGER DEFAULT 0
            """)
            cursor.execute("""
                UPDATE batch_jobs 
                SET processed = success_count
            """)
            conn.commit()
            print("Copied data from 'success_count' to 'processed'")
            
        # Check if failed_count exists but failed doesn't
        if check_column_exists('batch_jobs', 'failed_count') and not check_column_exists('batch_jobs', 'failed'):
            print("Copying data from 'failed_count' to 'failed'...")
            cursor.execute("""
                ALTER TABLE batch_jobs 
                ADD COLUMN failed INTEGER DEFAULT 0
            """)
            cursor.execute("""
                UPDATE batch_jobs 
                SET failed = failed_count
            """)
            conn.commit()
            print("Copied data from 'failed_count' to 'failed'")
        
        cursor.close()
        conn.close()
        
        print("Batch_jobs table schema fixed successfully")
        
    except Exception as e:
        logger.error(f"Error fixing batch_jobs table: {e}")
        print(f"Error: {e}")

def check_batch_jobs_table():
    """Check the batch_jobs table schema."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        
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
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error checking batch_jobs table: {e}")
        print(f"Error: {e}")

def fix_results_batch_relationship():
    """Fix the relationship between results and batch_jobs tables."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if batch_id is properly indexed
        cursor.execute("""
            SELECT indexname 
            FROM pg_indexes 
            WHERE tablename = 'results' AND indexname = 'idx_results_batch_id'
        """)
        
        if cursor.fetchone() is None:
            print("Creating index on results.batch_id...")
            cursor.execute("""
                CREATE INDEX idx_results_batch_id ON results (batch_id)
            """)
            conn.commit()
            print("Created index on results.batch_id")
        else:
            print("Index on results.batch_id already exists")
        
        cursor.close()
        conn.close()
        
        print("Results-batch relationship fixed successfully")
        
    except Exception as e:
        logger.error(f"Error fixing results-batch relationship: {e}")
        print(f"Error: {e}")

def test_batch_result_relationship():
    """Test the relationship between batch_jobs and results."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        # Get a sample batch_id from results
        cursor.execute("""
            SELECT batch_id FROM results 
            WHERE batch_id IS NOT NULL 
            LIMIT 1
        """)
        
        row = cursor.fetchone()
        
        if row and row['batch_id']:
            batch_id = row['batch_id']
            print(f"Found batch_id in results: {batch_id}")
            
            # Try to find the corresponding batch_job
            cursor.execute("""
                SELECT * FROM batch_jobs 
                WHERE metadata->>'batch_id' = %s
            """, (batch_id,))
            
            batch = cursor.fetchone()
            
            if batch:
                print(f"Found corresponding batch_job with ID: {batch['id']}")
                print(f"Status: {batch['status']}")
                if 'processed' in batch:
                    print(f"Processed: {batch['processed']}")
                if 'failed' in batch:
                    print(f"Failed: {batch['failed']}")
            else:
                print(f"Could not find batch_job with batch_id: {batch_id}")
        else:
            print("No results with batch_id found")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error testing batch-result relationship: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Starting database schema fix...")
    
    print("\n1. Checking batch_jobs table before fix:")
    check_batch_jobs_table()
    
    print("\n2. Fixing batch_jobs table:")
    fix_batch_jobs_table()
    
    print("\n3. Checking batch_jobs table after fix:")
    check_batch_jobs_table()
    
    print("\n4. Fixing results-batch relationship:")
    fix_results_batch_relationship()
    
    print("\n5. Testing batch-result relationship:")
    test_batch_result_relationship()
    
    print("\nSchema fix complete.")
