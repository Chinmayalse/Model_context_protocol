# Medical Report Processing Application Database Schema

## Overview

This document outlines the simplified database schema for the Medical Report Processing application. The schema focuses on the three essential tables needed to store user information, chat history, and processing results. Currently, the application uses JSON files for storage, but this schema will be used to migrate to a proper database solution.

## Database Tables

### 1. Users

Stores information about application users.

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(100) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);
```

### 2. Chats

Stores chat history including messages and metadata.

```sql
CREATE TABLE chats (
    id VARCHAR(100) PRIMARY KEY, -- UUID format for chat session
    user_id INTEGER NOT NULL,
    title VARCHAR(255) NOT NULL,
    date VARCHAR(50) NOT NULL, -- Format: YYYY-MM-DD HH:MM
    messages JSON NOT NULL, -- Array of message objects with content, timestamp, is_user, and metadata
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
```

### 3. Results

Stores the results of processing medical reports.

```sql
CREATE TABLE results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    filename VARCHAR(255) NOT NULL, -- The JSON result filename
    original_filename VARCHAR(255) NOT NULL, -- Original medical report filename
    processing_type VARCHAR(50) NOT NULL, -- 'summary', 'process', etc.
    tool_used VARCHAR(100) NOT NULL, -- Tool used for processing
    result_data JSON NOT NULL, -- JSON data with processing results
    chat_id VARCHAR(100), -- Reference to the chat where this result was generated (optional)
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE SET NULL
);
```

## Indexes

For optimal query performance, the following indexes should be created:

```sql
-- Users table
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);

-- MedicalReports table
CREATE INDEX idx_medical_reports_user_id ON medical_reports(user_id);
CREATE INDEX idx_medical_reports_report_type ON medical_reports(report_type);
CREATE INDEX idx_medical_reports_uploaded_at ON medical_reports(uploaded_at);

-- ProcessingResults table
CREATE INDEX idx_processing_results_report_id ON processing_results(report_id);
CREATE INDEX idx_processing_results_user_id ON processing_results(user_id);
CREATE INDEX idx_processing_results_processing_type ON processing_results(processing_type);

-- ChatMessages table
CREATE INDEX idx_chat_messages_user_id ON chat_messages(user_id);
CREATE INDEX idx_chat_messages_session_id ON chat_messages(session_id);
CREATE INDEX idx_chat_messages_created_at ON chat_messages(created_at);

-- ChatSessions table
CREATE INDEX idx_chat_sessions_user_id ON chat_sessions(user_id);
CREATE INDEX idx_chat_sessions_last_activity ON chat_sessions(last_activity);

-- ReportClassifications table
CREATE INDEX idx_report_classifications_report_id ON report_classifications(report_id);
CREATE INDEX idx_report_classifications_classification_type ON report_classifications(classification_type);

-- ExtractedText table
CREATE INDEX idx_extracted_text_report_id ON extracted_text(report_id);
```

## Relationships

1. **Users** to **MedicalReports**: One-to-many (one user can upload many reports)
2. **Users** to **ChatSessions**: One-to-many (one user can have many chat sessions)
3. **MedicalReports** to **ProcessingResults**: One-to-many (one report can have multiple processing results)
4. **MedicalReports** to **ReportClassifications**: One-to-many (one report can have multiple classifications)
5. **MedicalReports** to **ExtractedText**: One-to-many (one report can have multiple text extractions, e.g., per page)
6. **ChatSessions** to **ChatMessages**: One-to-many (one chat session contains many messages)

## Implementation Notes

1. **Authentication**: The system will use password hashing for security. Consider using a library like `bcrypt` or `passlib` for password hashing.

2. **File Storage**: Actual files (reports and results) will be stored on the filesystem, with paths stored in the database.

3. **JSON Data**: The `result_data` field in the `ProcessingResults` table stores JSON data, which allows for flexible storage of different result types.

4. **Soft Deletion**: The `is_deleted` flag in several tables allows for "soft deletion" where records are marked as deleted but not actually removed from the database.

5. **Database Choice**: This schema is designed to work with SQLite for simplicity, but can be easily adapted for PostgreSQL, MySQL, or other relational databases.

6. **ORM Integration**: The schema is compatible with SQLAlchemy or other ORMs for Python.

## Migration Strategy

When implementing this schema, consider the following migration strategy:

1. Create a migration script to create all tables
2. Add a script to migrate existing data from the file-based system to the database
3. Update the application code to use the database instead of the filesystem
4. Add a backup mechanism for the database

## Next Steps

After implementing this schema:

1. Add user authentication and authorization
2. Implement session management
3. Update the UI to show user-specific data
4. Add admin features for user management
