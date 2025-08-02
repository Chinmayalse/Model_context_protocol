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
        logging.FileHandler('batch_test_updated.log')
    ]
)
logger = logging.getLogger(__name__)

# Import our batch helper
from db_batch_helper import BatchManager

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
        conn = psycopg2.connect(
            dbname="mcp_database",
            user="postgres",
            password="postgres",
            host="localhost",
            port="5432"
        )
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

def process_result_folder(folder_path, user_id=1):
    """Process all result files in a folder and save them to the database."""
    try:
        # Find all JSON files in the folder
        result_files = glob.glob(os.path.join(folder_path, "*.json"))
        
        if not result_files:
            print(f"No JSON files found in {folder_path}")
            return
        
        print(f"Found {len(result_files)} result files in {folder_path}")
        
        # Create a batch job using BatchManager
        batch_id = BatchManager.create_batch_job(
            user_id=user_id,
            tool_name="process_medical_report",
            total_files=len(result_files),
            folder_path=folder_path
        )
        
        if not batch_id:
            print("Failed to create batch job")
            return
        
        print(f"Created batch job with ID: {batch_id}")
        
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
                    
                # Update batch status after each file
                BatchManager.update_batch_status(
                    batch_id=batch_id,
                    processed=processed,
                    failed=failed,
                    status="processing"
                )
                
            except Exception as e:
                logger.error(f"Error processing {result_file}: {e}")
                failed += 1
        
        # Update final batch status
        status = "completed" if failed == 0 else "partial_success" if processed > 0 else "failed"
        BatchManager.update_batch_status(
            batch_id=batch_id,
            processed=processed,
            failed=failed,
            status=status
        )
        
        print(f"Batch processing complete. Processed: {processed}, Failed: {failed}")
        
        # Get batch status
        final_status = BatchManager.get_batch_status(batch_id)
        print(f"Final batch status: {final_status}")
        
        # Get batch results
        results = get_batch_results(batch_id)
        print(f"Found {len(results)} results for batch {batch_id}")
        
    except Exception as e:
        logger.error(f"Error processing folder: {e}")
        print(f"Error: {e}")

def get_batch_results(batch_id, limit=10):
    """Get results for a batch."""
    try:
        conn = psycopg2.connect(
            dbname="mcp_database",
            user="postgres",
            password="postgres",
            host="localhost",
            port="5432"
        )
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        cursor.execute("""
            SELECT id, filename, processing_type, status, created_at
            FROM results
            WHERE batch_id = %s
            ORDER BY created_at DESC
            LIMIT %s
        """, (batch_id, limit))
        
        rows = cursor.fetchall()
        
        print(f"\nResults for batch {batch_id}:")
        for row in rows:
            print(f"- ID: {row['id']}, Filename: {row['filename']}, Type: {row['processing_type']}, Status: {row['status']}")
        
        cursor.close()
        conn.close()
        
        return rows
        
    except Exception as e:
        logger.error(f"Error getting batch results: {e}")
        print(f"Error: {e}")
        return []

def verify_database_state():
    """Verify the current state of the database."""
    try:
        conn = psycopg2.connect(
            dbname="mcp_database",
            user="postgres",
            password="postgres",
            host="localhost",
            port="5432"
        )
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        # Get count of results
        cursor.execute("SELECT COUNT(*) FROM results")
        count = cursor.fetchone()[0]
        
        print(f"Found {count} results in database")
        
        # Get count of batch jobs
        cursor.execute("SELECT COUNT(*) FROM batch_jobs")
        batch_count = cursor.fetchone()[0]
        
        print(f"Found {batch_count} batch jobs in database")
        
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
            print(f"- ID: {batch['id']}, Status: {batch['status']}, Total: {batch['total_files']}, " +
                  f"Processed: {batch['processed']}, Failed: {batch['failed']}, Batch ID: {batch['batch_id']}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error verifying database state: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Batch Processing Test (Updated)")
    print("==============================")
    
    # Verify current database state
    print("\n1. Verifying current database state:")
    verify_database_state()
    
    # Process a folder of results
    results_folder = r"c:\Users\chinmay alse\Desktop\MCP_1\demo_server\results\ihc_hcg"
    
    print(f"\n2. Processing results folder: {results_folder}")
    process_result_folder(results_folder)
    
    # Verify updated database state
    print("\n3. Verifying updated database state:")
    verify_database_state()
    
    print("\nTest complete.")
