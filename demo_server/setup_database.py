"""
Setup script for PostgreSQL database.
This script:
1. Creates the database if it doesn't exist
2. Creates the tables
3. Migrates data from JSON files
"""

import os
import subprocess
import sys
import psycopg2
from migrate_to_postgres import main as migrate_data

# Database connection parameters
DB_NAME = "mcp_database"
DB_USER = "postgres"  # Default PostgreSQL user, change if needed
DB_PASSWORD = "postgres"  # Default password, change to match your PostgreSQL setup
DB_HOST = "localhost"
DB_PORT = "5432"

def create_database():
    """Create the database if it doesn't exist."""
    try:
        print(f"Connecting to PostgreSQL with user '{DB_USER}'...")
        # Connect to the default 'postgres' database to create our database
        conn = psycopg2.connect(
            dbname="postgres",
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}'")
        exists = cursor.fetchone()
        
        if not exists:
            print(f"Creating database '{DB_NAME}'...")
            cursor.execute(f"CREATE DATABASE {DB_NAME}")
            print(f"Database '{DB_NAME}' created successfully!")
        else:
            print(f"Database '{DB_NAME}' already exists.")
        
        cursor.close()
        conn.close()
        return True
    except psycopg2.OperationalError as e:
        print(f"PostgreSQL connection error: {e}")
        print("\nPossible solutions:")
        print("1. Verify PostgreSQL is running")
        print("2. Check your username and password")
        print("3. Make sure you have permission to create databases")
        print("\nDefault PostgreSQL credentials are:")
        print("Username: postgres")
        print("Password: Usually set during installation")
        return False
    except Exception as e:
        print(f"Error creating database: {e}")
        return False

def create_tables():
    """Create tables in the database."""
    try:
        # Read the SQL script
        sql_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "create_database.sql")
        with open(sql_file, 'r') as f:
            sql_content = f.read()
        
        # Connect to the database
        print(f"Connecting to '{DB_NAME}' database...")
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        conn.autocommit = True
        
        # Split the SQL into individual statements
        # This handles multi-statement SQL scripts better
        statements = sql_content.split(';')
        
        cursor = conn.cursor()
        for statement in statements:
            # Skip empty statements
            if statement.strip():
                try:
                    cursor.execute(statement)
                    print(".", end="", flush=True)  # Progress indicator
                except Exception as stmt_error:
                    print(f"\nError executing statement: {stmt_error}")
                    print(f"Statement: {statement[:100]}...")
        
        print("\nTables created successfully!")
        cursor.close()
        conn.close()
        return True
    except psycopg2.OperationalError as e:
        print(f"\nDatabase connection error: {e}")
        return False
    except Exception as e:
        print(f"\nError creating tables: {e}")
        return False

def main():
    """Main setup function."""
    print("Setting up PostgreSQL database for MCP...")
    
    # Create database
    if not create_database():
        print("Failed to create database. Exiting.")
        return False
    
    # Create tables
    if not create_tables():
        print("Failed to create tables. Exiting.")
        return False
    
    # Migrate data
    print("\nMigrating data from JSON files to PostgreSQL...")
    migrate_data()
    
    print("\nDatabase setup completed!")
    return True

if __name__ == "__main__":
    main()
