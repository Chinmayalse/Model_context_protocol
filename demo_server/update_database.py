"""
Script to update the database schema for batch processing.
"""

import os
import psycopg2
from psycopg2.extras import execute_batch
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database connection parameters
DB_NAME = "mcp_database"
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = "localhost"
DB_PORT = "5432"

def get_connection():
    """Get a connection to the database."""
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

def execute_sql_file(file_path):
    """Execute SQL commands from a file."""
    try:
        with open(file_path, 'r') as f:
            sql = f.read()
            
        conn = get_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql)
        logger.info(f"Successfully executed SQL from {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error executing SQL from {file_path}: {e}")
        return False

def migrate_batch_data_from_user_objects():
    """
    Migrate batch data from user objects to the batch_jobs table.
    This is needed if batch jobs were previously stored in user.batch_jobs.
    """
    try:
        conn = get_connection()
        with conn:
            with conn.cursor() as cur:
                # Check if the users table has a batch_jobs column
                cur.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'users' AND column_name = 'batch_jobs'
                """)
                has_batch_jobs_column = cur.fetchone() is not None
                
                if not has_batch_jobs_column:
                    logger.info("No batch_jobs column in users table, skipping migration")
                    return
                
                # Get users with batch_jobs data
                cur.execute("SELECT id, batch_jobs FROM users WHERE batch_jobs IS NOT NULL")
                users = cur.fetchall()
                
                if not users:
                    logger.info("No users with batch_jobs data found")
                    return
                
                logger.info(f"Found {len(users)} users with batch_jobs data")
                
                # For each user, migrate their batch jobs
                for user_id, batch_jobs in users:
                    if not batch_jobs:
                        continue
                    
                    # Convert batch_jobs to a list of tuples for batch insert
                    batch_data = []
                    for batch_id, batch_info in batch_jobs.items():
                        # Extract relevant fields
                        status = batch_info.get('status', 'unknown')
                        total_files = batch_info.get('file_count', 0)
                        processed = batch_info.get('processed_count', 0)
                        failed = batch_info.get('failed_count', 0)
                        
                        # Create metadata JSON
                        metadata = {
                            'batch_id': batch_id,
                            'tool_name': batch_info.get('tool_name'),
                            'start_time': batch_info.get('start_time'),
                            'end_time': batch_info.get('end_time'),
                            'chat_id': batch_info.get('chat_id')
                        }
                        
                        batch_data.append((
                            user_id, status, total_files, processed, failed, 
                            psycopg2.extras.Json(metadata)
                        ))
                    
                    # Insert batch jobs
                    if batch_data:
                        execute_batch(cur, """
                            INSERT INTO batch_jobs 
                            (user_id, status, total_files, processed, failed, metadata)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT DO NOTHING
                        """, batch_data)
                        
                        logger.info(f"Migrated {len(batch_data)} batch jobs for user {user_id}")
                
                conn.commit()
                logger.info("Batch data migration completed successfully")
    except Exception as e:
        logger.error(f"Error migrating batch data: {e}")
        raise

def main():
    """Main function to update the database schema."""
    try:
        # Execute the SQL update script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sql_file = os.path.join(script_dir, 'update_schema.sql')
        
        if not os.path.exists(sql_file):
            logger.error(f"SQL file not found: {sql_file}")
            return False
        
        # Execute the SQL file
        if not execute_sql_file(sql_file):
            return False
        
        # Migrate batch data from user objects if needed
        migrate_batch_data_from_user_objects()
        
        logger.info("Database schema update completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error updating database schema: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("Database schema update completed successfully")
    else:
        print("Database schema update failed")
