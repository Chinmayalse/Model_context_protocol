-- Create the database
CREATE DATABASE mcp_database;

-- Connect to the database
\c mcp_database

-- Create the users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

-- Create the chats table
CREATE TABLE chats (
    id VARCHAR(100) PRIMARY KEY, -- UUID format for chat session
    user_id INTEGER NOT NULL,
    title VARCHAR(255) NOT NULL,
    date VARCHAR(50) NOT NULL, -- Format: YYYY-MM-DD HH:MM
    messages JSONB NOT NULL, -- Array of message objects with content, timestamp, is_user, and metadata
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Create the results table
CREATE TABLE results (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    filename VARCHAR(255) NOT NULL, -- The JSON result filename
    original_filename VARCHAR(255) NOT NULL, -- Original medical report filename
    processing_type VARCHAR(50) NOT NULL, -- 'summary', 'process', etc.
    tool_used VARCHAR(100) NOT NULL, -- Tool used for processing
    result_data JSONB NOT NULL, -- JSON data with processing results
    chat_id VARCHAR(100), -- Reference to the chat where this result was generated (optional)
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE SET NULL
);

-- Create indexes for better performance
-- Users table
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);

-- Chats table
CREATE INDEX idx_chats_user_id ON chats(user_id);
CREATE INDEX idx_chats_date ON chats(date);

-- Results table
CREATE INDEX idx_results_user_id ON results(user_id);
CREATE INDEX idx_results_chat_id ON results(chat_id);
CREATE INDEX idx_results_filename ON results(filename);
