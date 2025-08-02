import psycopg2
from psycopg2.extras import DictCursor

# Database connection parameters
DB_NAME = "mcp_database"
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = "localhost"
DB_PORT = "5432"

def check_tables():
    """Check tables in the database"""
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
        
        # Get list of tables
        cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
        tables = cursor.fetchall()
        
        print("Tables in database:")
        for table in tables:
            print(f"- {table['table_name']}")
            
        # Check if results table exists and get count
        if any(table['table_name'] == 'results' for table in tables):
            cursor.execute("SELECT COUNT(*) FROM results")
            count = cursor.fetchone()[0]
            print(f"\nResults table has {count} rows")
            
            # Get a sample row if there are any
            if count > 0:
                cursor.execute("SELECT * FROM results LIMIT 1")
                row = cursor.fetchone()
                print("\nSample row columns:")
                for column in row.keys():
                    print(f"- {column}")
                    
        # Check if batch_jobs table exists and get count
        if any(table['table_name'] == 'batch_jobs' for table in tables):
            cursor.execute("SELECT COUNT(*) FROM batch_jobs")
            count = cursor.fetchone()[0]
            print(f"\nBatch_jobs table has {count} rows")
            
            # Get a sample row if there are any
            if count > 0:
                cursor.execute("SELECT * FROM batch_jobs LIMIT 1")
                row = cursor.fetchone()
                print("\nSample row columns:")
                for column in row.keys():
                    print(f"- {column}")
        
        # Close the connection
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_tables()
