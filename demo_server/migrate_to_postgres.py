"""
Migration script to transfer data from JSON files to PostgreSQL database.
"""

import os
import json
import psycopg2
from psycopg2.extras import Json
from datetime import datetime
import glob

# Database connection parameters
DB_NAME = "mcp_database"
DB_USER = "postgres"  # Default PostgreSQL user, change if needed
DB_PASSWORD = "postgres"  # Default password, change to match your PostgreSQL setup
DB_HOST = "localhost"
DB_PORT = "5432"

# File paths
USERS_FILE = 'users.json'
RESULTS_FOLDER = 'chat_results'

def connect_to_db():
    """Connect to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        print(f"Connected to database: {DB_NAME}")
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def load_users():
    """Load users from JSON file."""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def migrate_users(conn, users_data):
    """Migrate users to the database."""
    cursor = conn.cursor()
    users_map = {}  # Map to store old user_id -> new user_id
    
    print("Migrating users...")
    for user_id, user_data in users_data.items():
        try:
            # Insert user
            cursor.execute(
                """
                INSERT INTO users (username, email, password_hash, created_at)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (
                    user_data.get('username', ''),
                    user_data.get('email', ''),
                    user_data.get('password_hash', ''),
                    datetime.now()
                )
            )
            new_user_id = cursor.fetchone()[0]
            users_map[user_id] = new_user_id
            print(f"Migrated user: {user_data.get('username')} (ID: {new_user_id})")
            
            # Migrate chats for this user
            if 'chat_history' in user_data:
                migrate_chats(conn, new_user_id, user_data['chat_history'])
                
        except Exception as e:
            print(f"Error migrating user {user_id}: {e}")
    
    conn.commit()
    return users_map

def migrate_chats(conn, user_id, chats):
    """Migrate chats to the database."""
    cursor = conn.cursor()
    
    for chat in chats:
        try:
            # Insert chat
            cursor.execute(
                """
                INSERT INTO chats (id, user_id, title, date, messages, created_at, is_deleted)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    chat.get('id', ''),
                    user_id,
                    chat.get('title', ''),
                    chat.get('date', ''),
                    Json(chat.get('messages', [])),
                    datetime.now(),
                    False
                )
            )
            print(f"Migrated chat: {chat.get('title')} (ID: {chat.get('id')})")
            
        except Exception as e:
            print(f"Error migrating chat {chat.get('id')}: {e}")
    
    conn.commit()

def migrate_results(conn, users_map):
    """Migrate result files to the database."""
    cursor = conn.cursor()
    
    # Get all JSON result files
    result_files = glob.glob(os.path.join(RESULTS_FOLDER, '*.json'))
    
    for file_path in result_files:
        try:
            filename = os.path.basename(file_path)
            
            # Read the result file
            with open(file_path, 'r') as f:
                result_data = json.load(f)
            
            # Extract information from the filename
            # Example: CBC_-_Complete_Blood_Count_process_medical_report_20250519_203659.json
            parts = filename.split('_')
            
            # Try to determine processing type and tool used from filename
            processing_type = 'process'
            if 'summarize' in filename:
                processing_type = 'summary'
                
            tool_used = 'process_medical_report'
            if 'summarize' in filename:
                tool_used = 'summarize_medical_report'
            
            # Use the first user as default (you might need to adjust this logic)
            user_id = list(users_map.values())[0] if users_map else 1
            
            # Insert result
            cursor.execute(
                """
                INSERT INTO results 
                (user_id, filename, original_filename, processing_type, tool_used, result_data, created_at, is_deleted)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    user_id,
                    filename,
                    result_data.get('original_filename', filename),
                    processing_type,
                    tool_used,
                    Json(result_data),
                    datetime.now(),
                    False
                )
            )
            print(f"Migrated result: {filename}")
            
        except Exception as e:
            print(f"Error migrating result {file_path}: {e}")
    
    conn.commit()

def main():
    """Main migration function."""
    # Connect to database
    conn = connect_to_db()
    if not conn:
        return
    
    try:
        # Load users data
        users_data = load_users()
        
        # Migrate users and chats
        users_map = migrate_users(conn, users_data)
        
        # Migrate results
        migrate_results(conn, users_map)
        
        print("Migration completed successfully!")
        
    except Exception as e:
        print(f"Error during migration: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
