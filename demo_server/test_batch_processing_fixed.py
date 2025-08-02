import os
import glob
import json
import logging
import uuid
from datetime import datetime
import psycopg2
from psycopg2.extras import Json, DictCursor

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('batch_test.log')
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

def save_result_to_database(result_file, user_id=1, batch_id=None):
    """Save a result file to the database."""
    try:
        # Load the result file
        with open(result_file, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
        
        # Extract filename and determine processing type
        filename = os.path.basename(result_file)
        original_filename = os.path.splitext(os.path.basename(result_file))[0]
        
        # Determine processing type based on result data
        if 'report_type' in result_data:
            processing_type = result_data['report_type'].lower()
        else:
            processing_type = 'process'
        
        # Determine tool used
        if 'extraction_method' in result_data:
            tool_used = result_data['extraction_method']
        else:
            tool_used = 'gemini_2.0_flash'
        
        # Determine status
        status = 'completed' if result_data.get('success', False) else 'failed'
        
        # Connect to database
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        # Insert into database
        query = """
            INSERT INTO results 
            (user_id, filename, original_filename, processing_type, tool_used, 
             status, result_data, batch_id, created_at, is_deleted)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        
        # Use timestamp from result if available, otherwise use current time
        if 'timestamp' in result_data:
            try:
                created_at = datetime.strptime(result_data['timestamp'], '%Y-%m-%d %H:%M:%S')
            except ValueError:
                created_at = datetime.now()
        else:
            created_at = datetime.now()
        
        params = (
            user_id, filename, original_filename, processing_type, tool_used,
            status, Json(result_data), batch_id, created_at, False
        )
        
        cursor.execute(query, params)
        result = cursor.fetchone()
        result_id = result[0]
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Saved result {filename} to database with ID: {result_id}")
        print(f"Saved result {filename} to database with ID: {result_id}")
        
        return result_id
        
    except Exception as e:
        logger.error(f"Error saving result to database: {e}")
        print(f"Error: {e}")
        return None

def create_batch_job(user_id=1, tool_name="process_medical_report", total_files=0):
    """Create a batch job in the database."""
    try:
        batch_id = str(uuid.uuid4())
        
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        # Insert batch - using the actual schema fields
        cursor.execute(
            """INSERT INTO batch_jobs 
               (user_id, status, total_files, processed_files, success_count, failed_count, 
                processed, failed, tool_name, tool_args, created_at, read, folder_path) 
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
               RETURNING id""",
            (user_id, "processing", total_files, 0, 0, 0, 0, 0, 
             tool_name, Json({'batch_id': batch_id}), datetime.now(), False, "batch_import")
        )
        
        batch_db_id = cursor.fetchone()[0]
        conn.commit()
        
        # Now store the batch_id in the tool_args JSONB field
        cursor.execute(
            """UPDATE batch_jobs SET tool_args = %s WHERE id = %s""",
            (Json({'batch_id': batch_id}), batch_db_id)
        )
        conn.commit()
        
        cursor.close()
        conn.close()
        
        logger.info(f"Created batch job with ID: {batch_db_id} and batch_id: {batch_id}")
        print(f"Created batch job with ID: {batch_db_id} and batch_id: {batch_id}")
        
        return batch_id
        
    except Exception as e:
        logger.error(f"Error creating batch job: {e}")
        print(f"Error: {e}")
        return None

def update_batch_status(batch_id, processed, failed, status="completed"):
    """Update batch job status."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """UPDATE batch_jobs 
               SET processed = %s, 
                   failed = %s, 
                   processed_files = %s,
                   success_count = %s,
                   failed_count = %s,
                   status = %s,
                   completed_at = CURRENT_TIMESTAMP
               WHERE tool_args->>'batch_id' = %s""",
            (processed, failed, processed, processed, failed, status, batch_id)
        )
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Updated batch {batch_id}: processed={processed}, failed={failed}, status={status}")
        print(f"Updated batch {batch_id}: processed={processed}, failed={failed}, status={status}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error updating batch status: {e}")
        print(f"Error: {e}")
        return False

def process_result_folder(folder_path, user_id=1):
    """Process all result files in a folder and save them to the database."""
    try:
        # Find all JSON files in the folder
        result_files = glob.glob(os.path.join(folder_path, "*.json"))
        
        if not result_files:
            print(f"No JSON files found in {folder_path}")
            return
        
        print(f"Found {len(result_files)} result files in {folder_path}")
        
        # Create a batch job
        batch_id = create_batch_job(user_id=user_id, total_files=len(result_files))
        
        if not batch_id:
            print("Failed to create batch job")
            return
        
        # Process each file
        processed = 0
        failed = 0
        
        for result_file in result_files:
            try:
                result_id = save_result_to_database(result_file, user_id=user_id, batch_id=batch_id)
                
                if result_id:
                    processed += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Error processing {result_file}: {e}")
                failed += 1
        
        # Update batch status
        status = "completed" if failed == 0 else "partial_success" if processed > 0 else "failed"
        update_batch_status(batch_id, processed, failed, status)
        
        print(f"Batch processing complete. Processed: {processed}, Failed: {failed}")
        
    except Exception as e:
        logger.error(f"Error processing folder: {e}")
        print(f"Error: {e}")

def verify_results_in_database():
    """Verify that results are in the database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        # Get count of results
        cursor.execute("SELECT COUNT(*) FROM results")
        count = cursor.fetchone()[0]
        
        print(f"Found {count} results in database")
        
        # Get count of batch jobs
        cursor.execute("SELECT COUNT(*) FROM batch_jobs")
        batch_count = cursor.fetchone()[0]
        
        print(f"Found {batch_count} batch jobs in database")
        
        # Get recent results
        cursor.execute("""
            SELECT id, filename, processing_type, status, created_at, batch_id 
            FROM results 
            ORDER BY created_at DESC 
            LIMIT 10
        """)
        
        rows = cursor.fetchall()
        
        print("\nRecent results:")
        for row in rows:
            print(f"- ID: {row['id']}, Filename: {row['filename']}, Type: {row['processing_type']}, Status: {row['status']}, Batch: {row['batch_id']}")
        
        # Get recent batch jobs
        cursor.execute("""
            SELECT id, status, total_files, processed, failed, created_at, tool_args->>'batch_id' as batch_id 
            FROM batch_jobs 
            ORDER BY created_at DESC 
            LIMIT 5
        """)
        
        batches = cursor.fetchall()
        
        print("\nRecent batch jobs:")
        for batch in batches:
            print(f"- ID: {batch['id']}, Status: {batch['status']}, Total: {batch['total_files']}, Processed: {batch['processed']}, Failed: {batch['failed']}, Batch ID: {batch['batch_id']}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error verifying results: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Batch Processing Test")
    print("====================")
    
    # Verify current database state
    print("\n1. Verifying current database state:")
    verify_results_in_database()
    
    # Process a folder of results
    results_folder = r"c:\Users\chinmay alse\Desktop\MCP_1\demo_server\results\ihc_hcg"
    
    print(f"\n2. Processing results folder: {results_folder}")
    process_result_folder(results_folder)
    
    # Verify updated database state
    print("\n3. Verifying updated database state:")
    verify_results_in_database()
    
    print("\nTest complete.")
