import psycopg2
from psycopg2.extras import DictCursor
import os
import json

# Database connection parameters
DB_NAME = "mcp_database"
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = "localhost"
DB_PORT = "5432"

def check_result_in_db(filename):
    """Check if a specific result file is in the database"""
    try:
        # Connect to the database
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        
        # Create a cursor
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        # Get the result from the database
        cursor.execute("SELECT * FROM results WHERE filename = %s", (os.path.basename(filename),))
        row = cursor.fetchone()
        
        if row:
            print(f"Result found in database with ID: {row['id']}")
            print(f"Status: {row.get('status', 'Not specified')}")
            print(f"Created at: {row.get('created_at', 'Not specified')}")
            print(f"Batch ID: {row.get('batch_id', 'None')}")
            
            # Check if result_data is populated
            if row['result_data']:
                print("Result data is present in the database")
                
                # Compare with local file
                try:
                    with open(filename, 'r') as f:
                        local_data = json.load(f)
                    
                    # Check if key fields match
                    if isinstance(row['result_data'], dict):
                        db_data = row['result_data']
                    else:
                        db_data = json.loads(row['result_data'])
                    
                    print("\nComparing key fields:")
                    
                    # Check report_type
                    if 'report_type' in local_data and 'report_type' in db_data:
                        print(f"Report type match: {'Yes' if local_data['report_type'] == db_data['report_type'] else 'No'}")
                    
                    # Check success
                    if 'success' in local_data and 'success' in db_data:
                        print(f"Success match: {'Yes' if local_data['success'] == db_data['success'] else 'No'}")
                    
                    # Check timestamp
                    if 'timestamp' in local_data and 'timestamp' in db_data:
                        print(f"Timestamp match: {'Yes' if local_data['timestamp'] == db_data['timestamp'] else 'No'}")
                    
                except Exception as e:
                    print(f"Error comparing with local file: {e}")
            else:
                print("Result data is NOT present in the database")
        else:
            print(f"Result NOT found in database")
            
            # Check if there are any results with similar filenames
            cursor.execute("SELECT filename FROM results WHERE filename LIKE %s", (f"%{os.path.basename(filename)[:10]}%",))
            similar = cursor.fetchall()
            
            if similar:
                print("\nSimilar filenames found:")
                for s in similar:
                    print(f"- {s['filename']}")
        
        # Check batch_jobs table for related batches
        cursor.execute("SELECT * FROM batch_jobs WHERE id IN (SELECT DISTINCT batch_job_id FROM results WHERE filename = %s)", 
                      (os.path.basename(filename),))
        batch = cursor.fetchone()
        
        if batch:
            print(f"\nAssociated batch job found with ID: {batch['id']}")
            print(f"Status: {batch.get('status', 'Not specified')}")
            print(f"Created at: {batch.get('created_at', 'Not specified')}")
            print(f"Total files: {batch.get('total_files', 'Not specified')}")
            print(f"Processed files: {batch.get('processed_files', 'Not specified')}")
        
        # Close the connection
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Check the specific result file
    result_file = r"c:\Users\chinmay alse\Desktop\MCP_1\demo_server\results\ihc_hcg\IN-423-YENAP_Ventana Negative_20250601_150646.json"
    check_result_in_db(result_file)
