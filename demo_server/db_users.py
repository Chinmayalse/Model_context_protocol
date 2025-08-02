"""
User class for PostgreSQL database interaction.
"""

from flask_login import UserMixin
import uuid
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from db import get_db_cursor
from psycopg2.extras import Json

class User(UserMixin):
    def __init__(self, id, username, email, password_hash, chat_history=None):
        self.id = id
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.chat_history = chat_history or []

    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)

    def get_id(self):
        return str(self.id)
        
    def add_chat(self, chat_id, title, date):
        """Add a new chat to the user's history."""
        # Create new chat entry
        chat_entry = {
            'id': chat_id,
            'title': title,
            'date': date,
            'messages': []
        }
        
        # Add to database
        with get_db_cursor(commit=True) as cur:
            cur.execute(
                """
                INSERT INTO chats (id, user_id, title, date, messages, created_at, is_deleted)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    chat_id,
                    self.id,
                    title,
                    date,
                    Json([]),  # Empty messages array
                    datetime.now(),
                    False
                )
            )
            
        # Add to local chat_history
        self.chat_history.append(chat_entry)
        return True
        
    def add_message_to_chat(self, chat_id, message, is_user=True, metadata=None):
        """Add a message to an existing chat."""
        # Create message entry
        message_entry = {
            'content': message,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'is_user': is_user
        }
        
        # Add metadata if provided (for assistant responses with result files, etc.)
        if metadata and not is_user:
            message_entry['metadata'] = metadata
        
        # Find the chat in local history and update it
        chat_found = False
        for chat in self.chat_history:
            if chat['id'] == chat_id:
                if 'messages' not in chat:
                    chat['messages'] = []
                chat['messages'].append(message_entry)
                chat_found = True
                break
                
        if not chat_found:
            return False
            
        # Update the database
        with get_db_cursor(commit=True) as cur:
            # First get the current messages
            cur.execute(
                "SELECT messages FROM chats WHERE id = %s",
                (chat_id,)
            )
            result = cur.fetchone()
            
            if not result:
                return False
                
            # Handle both tuple and dictionary access for the result
            if isinstance(result, dict):
                messages = result['messages']
            else:  # It's a tuple
                messages = result[0]  # messages is the first (and only) column
                
            # Ensure messages is a list
            if not isinstance(messages, list):
                messages = []
                
            # Append the new message
            messages.append(message_entry)
            
            # Update the database
            cur.execute(
                "UPDATE chats SET messages = %s WHERE id = %s",
                (Json(messages), chat_id)
            )
            
        return True
        
    def get_chat_messages(self, chat_id):
        """Get all messages for a specific chat."""
        # Try to get from local cache first
        for chat in self.chat_history:
            if chat['id'] == chat_id:
                return chat.get('messages', [])
                
        # If not found in cache, query the database
        with get_db_cursor() as cur:
            cur.execute(
                "SELECT messages FROM chats WHERE id = %s",
                (chat_id,)
            )
            result = cur.fetchone()
            
            if result:
                return result['messages']
                
        return []
        
    def delete_chat(self, chat_id):
        """Delete a chat by its ID."""
        # Remove from local cache
        for i, chat in enumerate(self.chat_history):
            if chat['id'] == chat_id:
                self.chat_history.pop(i)
                break
        
        # Mark as deleted in database (soft delete)
        with get_db_cursor(commit=True) as cur:
            cur.execute(
                "UPDATE chats SET is_deleted = TRUE WHERE id = %s AND user_id = %s",
                (chat_id, self.id)
            )
            
            # Check if any rows were affected
            if cur.rowcount > 0:
                return True
                
        return False

    @staticmethod
    def get(user_id):
        """Get a user by ID."""
        with get_db_cursor() as cur:
            cur.execute(
                "SELECT * FROM users WHERE id = %s",
                (user_id,)
            )
            user_data = cur.fetchone()
            
            if not user_data:
                return None
                
            # Convert tuple to dict if needed
            if isinstance(user_data, tuple):
                columns = [desc[0] for desc in cur.description] if cur.description else []
                user_data = dict(zip(columns, user_data))
                
            # Get user's chats
            cur.execute(
                "SELECT * FROM chats WHERE user_id = %s AND is_deleted = FALSE ORDER BY created_at DESC",
                (user_id,)
            )
            chats = cur.fetchall()
            
            # Convert to list of dictionaries
            chat_history = []
            for chat in chats:
                # Convert tuple to dict if needed
                if isinstance(chat, tuple):
                    chat_columns = [desc[0] for desc in cur.description] if cur.description else []
                    chat = dict(zip(chat_columns, chat))
                    
                chat_history.append({
                    'id': chat['id'],
                    'title': chat['title'],
                    'date': chat['date'],
                    'messages': chat['messages']
                })
                
            return User(
                id=user_data['id'],
                username=user_data['username'],
                email=user_data['email'],
                password_hash=user_data['password_hash'],
                chat_history=chat_history
            )

    @staticmethod
    def get_by_email(email):
        """Get a user by email."""
        with get_db_cursor() as cur:
            cur.execute(
                "SELECT * FROM users WHERE email = %s",
                (email,)
            )
            user_data = cur.fetchone()
            
            if not user_data:
                return None
                
            # Convert tuple to dict if needed
            if isinstance(user_data, tuple):
                columns = [desc[0] for desc in cur.description] if cur.description else []
                user_data = dict(zip(columns, user_data))
                
            # Get user's chats (same as in get method)
            cur.execute(
                "SELECT * FROM chats WHERE user_id = %s AND is_deleted = FALSE ORDER BY created_at DESC",
                (user_data['id'],)
            )
            chats = cur.fetchall()
            
            # Convert to list of dictionaries
            chat_history = []
            for chat in chats:
                # Convert tuple to dict if needed
                if isinstance(chat, tuple):
                    chat_columns = [desc[0] for desc in cur.description] if cur.description else []
                    chat = dict(zip(chat_columns, chat))
                    
                chat_history.append({
                    'id': chat['id'],
                    'title': chat['title'],
                    'date': chat['date'],
                    'messages': chat['messages']
                })
                
            return User(
                id=user_data['id'],
                username=user_data['username'],
                email=user_data['email'],
                password_hash=user_data['password_hash'],
                chat_history=chat_history
            )

    @staticmethod
    def create(username, email, password):
        """Create a new user."""
        # Check if email already exists
        with get_db_cursor() as cur:
            cur.execute(
                "SELECT * FROM users WHERE email = %s",
                (email,)
            )
            if cur.fetchone():
                return None, 'Email already registered'
        
        # Create new user
        password_hash = generate_password_hash(password)
        
        with get_db_cursor(commit=True) as cur:
            cur.execute(
                """
                INSERT INTO users (username, email, password_hash, created_at)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (username, email, password_hash, datetime.now())
            )
            user_id = cur.fetchone()['id']
            
            return User(
                id=user_id,
                username=username,
                email=email,
                password_hash=password_hash,
                chat_history=[]
            ), None
