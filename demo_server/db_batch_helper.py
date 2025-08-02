import os
import logging
import uuid
from datetime import datetime
import psycopg2
from psycopg2.extras import Json, DictCursor

# Configure logging
logging.basicConfig(level=logging.INFO)
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

class BatchManager:
    """Manager for batch job operations in the database."""
    
    @staticmethod
    def create_batch_job(user_id, tool_name, total_files=0, folder_path=""):
        """Create a new batch job in the database."""
        try:
            batch_id = str(uuid.uuid4())
            
            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=DictCursor)
                
                # Insert batch with batch_id in tool_args
                cursor.execute(
                    """INSERT INTO batch_jobs 
                       (user_id, status, total_files, processed_files, success_count, failed_count, 
                        processed, failed, tool_name, tool_args, created_at, read, folder_path) 
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
                       RETURNING id""",
                    (user_id, "processing", total_files, 0, 0, 0, 0, 0, 
                     tool_name, Json({'batch_id': batch_id}), datetime.now(), False, folder_path)
                )
                
                batch_db_id = cursor.fetchone()[0]
                conn.commit()
            
            logger.info(f"Created batch job with ID: {batch_db_id} and batch_id: {batch_id}")
            return batch_id
            
        except Exception as e:
            logger.error(f"Error creating batch job: {e}")
            return None
    
    @staticmethod
    def update_batch_status(batch_id, processed=None, failed=None, status=None, total_files=None, error_message=None):
        """Update batch job status in the database."""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Build update query parts
                update_parts = []
                params = []
                
                if processed is not None:
                    update_parts.append("processed = %s")
                    params.append(processed)
                    update_parts.append("processed_files = %s")
                    params.append(processed)
                    update_parts.append("success_count = %s")
                    params.append(processed)
                
                if failed is not None:
                    update_parts.append("failed = %s")
                    params.append(failed)
                    update_parts.append("failed_count = %s")
                    params.append(failed)
                
                if status is not None:
                    update_parts.append("status = %s")
                    params.append(status)
                    
                    # Set completed_at if status indicates completion
                    if status in ["completed", "partial_success", "completed_with_errors"]:
                        update_parts.append("completed_at = CURRENT_TIMESTAMP")
                
                # Handle error message if provided
                if error_message is not None:
                    # Store error message in the tool_args JSONB column
                    update_parts.append("tool_args = jsonb_set(tool_args, '{error}', %s)")
                    params.append(Json(error_message))
                
                if total_files is not None:
                    update_parts.append("total_files = %s")
                    params.append(total_files)
                
                # Only update if we have parameters
                if update_parts:
                    query = f"""UPDATE batch_jobs 
                                SET {', '.join(update_parts)} 
                                WHERE tool_args->>'batch_id' = %s"""
                    params.append(batch_id)
                    
                    cursor.execute(query, params)
                    rows_updated = cursor.rowcount
                    conn.commit()
                    
                    logger.info(f"Updated batch {batch_id}: {rows_updated} rows affected")
                    return rows_updated > 0
                
                return False
                
        except Exception as e:
            logger.error(f"Error updating batch status: {e}")
            return False
    
    @staticmethod
    def get_batch_status(batch_id):
        """Get the current status of a batch job."""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=DictCursor)
                
                cursor.execute(
                    """SELECT id, status, total_files, processed, failed, 
                              processed_files, success_count, failed_count,
                              created_at, completed_at
                       FROM batch_jobs 
                       WHERE tool_args->>'batch_id' = %s""",
                    (batch_id,)
                )
                
                row = cursor.fetchone()
                
                if row:
                    return dict(row)
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting batch status: {e}")
            return None
    
    @staticmethod
    def get_batch_results(batch_id, limit=100):
        """Get results associated with a batch job."""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=DictCursor)
                
                cursor.execute(
                    """SELECT id, filename, original_filename, processing_type, 
                              tool_used, status, created_at
                       FROM results 
                       WHERE batch_id = %s
                       ORDER BY created_at DESC
                       LIMIT %s""",
                    (batch_id, limit)
                )
                
                rows = cursor.fetchall()
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error getting batch results: {e}")
            return []

# Example usage
if __name__ == "__main__":
    # Create a test batch
    batch_id = BatchManager.create_batch_job(
        user_id=1, 
        tool_name="test_batch", 
        total_files=5,
        folder_path="/test/path"
    )
    
    if batch_id:
        print(f"Created batch job: {batch_id}")
        
        # Update status
        BatchManager.update_batch_status(
            batch_id=batch_id,
            processed=2,
            failed=1,
            status="processing"
        )
        
        # Get status
        status = BatchManager.get_batch_status(batch_id)
        print(f"Batch status: {status}")
        
        # Complete the batch
        BatchManager.update_batch_status(
            batch_id=batch_id,
            processed=4,
            failed=1,
            status="completed"
        )
        
        # Get final status
        final_status = BatchManager.get_batch_status(batch_id)
        print(f"Final batch status: {final_status}")
    else:
        print("Failed to create batch job")
