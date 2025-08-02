import os
import json
import logging
from datetime import datetime
import uuid
import traceback
import psycopg2
from psycopg2.extras import Json, DictCursor

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('debug_save.log')
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

def test_save_result():
    """Test saving a result to the database."""
    try:
        # Create a test result
        test_result = {
            'success': True,
            'report_type': 'TEST_REPORT',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'test_data': 'This is a test result'
        }
        
        # Generate a unique filename
        filename = f"test_result_{uuid.uuid4()}.json"
        
        # Database parameters
        user_id = 1  # Assuming user ID 1 exists
        original_filename = "test_file.pdf"
        processing_type = "process"
        tool_used = "test_tool"
        batch_id = str(uuid.uuid4())
        
        # Insert into database
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        query = """
            INSERT INTO results 
            (user_id, filename, original_filename, processing_type, tool_used, 
             status, result_data, batch_id, created_at, is_deleted)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        
        params = (
            user_id, filename, original_filename, processing_type, tool_used,
            'completed', Json(test_result), batch_id,
            datetime.now(), False
        )
        
        logger.info(f"Executing query: {query}")
        logger.info(f"With parameters: {params}")
        
        cursor.execute(query, params)
        result = cursor.fetchone()
        result_id = result[0]
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Successfully saved result with ID: {result_id}")
        print(f"Successfully saved result with ID: {result_id}")
        
        # Now verify we can retrieve it
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        cursor.execute("SELECT * FROM results WHERE id = %s", (result_id,))
        row = cursor.fetchone()
        
        if row:
            logger.info(f"Retrieved result: {row['filename']}")
            print(f"Retrieved result: {row['filename']}")
            print(f"Result data type: {type(row['result_data'])}")
            print(f"Result data: {row['result_data']}")
        else:
            logger.error("Could not retrieve the result we just inserted!")
            print("Could not retrieve the result we just inserted!")
        
        cursor.close()
        conn.close()
        
        return result_id
        
    except Exception as e:
        logger.error(f"Error in test_save_result: {e}")
        logger.error(traceback.format_exc())
        print(f"Error: {e}")
        return None

def check_table_schema():
    """Check the schema of the results table."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        # Check table columns
        cursor.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'results'
            ORDER BY ordinal_position
        """)
        
        columns = cursor.fetchall()
        
        print("Results table schema:")
        for col in columns:
            print(f"- {col['column_name']}: {col['data_type']} (Nullable: {col['is_nullable']})")
        
        # Check indexes
        cursor.execute("""
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE tablename = 'results'
        """)
        
        indexes = cursor.fetchall()
        
        print("\nIndexes on results table:")
        for idx in indexes:
            print(f"- {idx['indexname']}: {idx['indexdef']}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error checking table schema: {e}")
        print(f"Error: {e}")

def check_batch_jobs_table():
    """Check the batch_jobs table."""
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
        
        # Check content
        cursor.execute("SELECT * FROM batch_jobs LIMIT 5")
        rows = cursor.fetchall()
        
        print(f"\nFound {len(rows)} rows in batch_jobs table")
        for row in rows:
            print(f"- ID: {row['id']}, Status: {row['status']}, Total: {row.get('total_files', 'N/A')}, Processed: {row.get('processed_files', 'N/A')}")
            if 'metadata' in row and row['metadata']:
                print(f"  Metadata: {row['metadata']}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error checking batch_jobs table: {e}")
        print(f"Error: {e}")

def test_batch_result_relationship():
    """Test the relationship between batch_jobs and results tables."""
    try:
        # First create a batch
        batch_id = str(uuid.uuid4())
        
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        # Insert batch
        cursor.execute(
            """INSERT INTO batch_jobs 
               (user_id, status, total_files, processed, failed, metadata) 
               VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
            (1, "processing", 10, 0, 0, Json({'batch_id': batch_id}))
        )
        
        batch_db_id = cursor.fetchone()[0]
        conn.commit()
        
        print(f"Created test batch with ID: {batch_db_id} and batch_id: {batch_id}")
        
        # Now create a result linked to this batch
        test_result = {
            'success': True,
            'report_type': 'TEST_BATCH_REPORT',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'batch_id': batch_id
        }
        
        filename = f"test_batch_result_{uuid.uuid4()}.json"
        
        cursor.execute(
            """INSERT INTO results 
               (user_id, filename, original_filename, processing_type, tool_used, 
                status, result_data, batch_id, created_at, is_deleted)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
               RETURNING id""",
            (1, filename, "test_file.pdf", "process", "test_tool",
             'completed', Json(test_result), batch_id, datetime.now(), False)
        )
        
        result_id = cursor.fetchone()[0]
        conn.commit()
        
        print(f"Created test result with ID: {result_id} linked to batch: {batch_id}")
        
        # Now query to check the relationship
        cursor.execute(
            """SELECT r.id, r.filename, r.batch_id, b.id as batch_db_id, b.status
               FROM results r
               JOIN batch_jobs b ON r.batch_id = b.metadata->>'batch_id'
               WHERE r.id = %s""",
            (result_id,)
        )
        
        row = cursor.fetchone()
        
        if row:
            print(f"Found relationship: Result {row['id']} is linked to batch {row['batch_db_id']}")
            print(f"Batch status: {row['status']}")
        else:
            print("Could not find the relationship between result and batch!")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error testing batch relationship: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Starting database debugging...")
    print("\n1. Checking results table schema:")
    check_table_schema()
    
    print("\n2. Checking batch_jobs table:")
    check_batch_jobs_table()
    
    print("\n3. Testing saving a result:")
    test_save_result()
    
    print("\n4. Testing batch-result relationship:")
    test_batch_result_relationship()
    
    print("\nDebugging complete.")
