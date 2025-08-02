# PostgreSQL Migration Guide for MCP

This guide explains how to migrate the Medical Report Processing application from JSON file storage to PostgreSQL.

## Prerequisites

- PostgreSQL 12+ installed and running
- Python 3.8+ with pip
- psycopg2 package (`pip install psycopg2` or `pip install psycopg2-binary`)

## Setup Instructions

1. **Configure Database Connection**

   Edit the following files and update the database connection parameters:
   - `db.py`
   - `migrate_to_postgres.py`
   - `setup_database.py`

   Update these variables:
   ```python
   DB_NAME = "mcp_database"
   DB_USER = "postgres"  # Change if needed
   DB_PASSWORD = ""      # Add your password
   DB_HOST = "localhost"
   DB_PORT = "5432"
   ```

2. **Run the Setup Script**

   ```bash
   python setup_database.py
   ```

   This will:
   - Create the database if it doesn't exist
   - Create the required tables
   - Migrate existing data from JSON files

3. **Test the Database Connection**

   ```bash
   python db.py
   ```

   You should see a message confirming the connection to PostgreSQL.

4. **Update the Application**

   To switch the application to use PostgreSQL:
   
   ```bash
   # Make a backup of the current users.py
   cp users.py users_json.py
   
   # Replace with the PostgreSQL version
   cp db_users.py users.py
   ```

## Database Schema

The database consists of three main tables:

1. **users** - Stores user authentication information
2. **chats** - Stores chat history with messages as JSON
3. **results** - Stores medical report processing results

## Switching Back to JSON Storage

If needed, you can switch back to the JSON storage by restoring the original users.py file:

```bash
cp users_json.py users.py
```

## Additional Notes

- The application will work exactly the same way after migration - all functionality is preserved
- Chat history and results are accessible in the same way as before
- The "View Result" button will continue to work as expected

## Troubleshooting

- If you encounter connection errors, verify that PostgreSQL is running and the connection parameters are correct
- If migration fails, check the error messages for details on what went wrong
- For permission issues, ensure your PostgreSQL user has the necessary privileges
