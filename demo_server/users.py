from flask_login import UserMixin
import os
import json
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

USERS_FILE = 'users.json'

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)

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
        users = load_users()
        if self.id in users:
            if 'chat_history' not in users[self.id]:
                users[self.id]['chat_history'] = []
                
            # Add new chat to history
            chat_entry = {
                'id': chat_id,
                'title': title,
                'date': date,
                'messages': []
            }
            users[self.id]['chat_history'].append(chat_entry)
            self.chat_history.append(chat_entry)
            
            # Save to file
            save_users(users)
            return True
        return False
        
    def add_message_to_chat(self, chat_id, message, is_user=True, metadata=None):
        users = load_users()
        if self.id in users:
            # Find the chat in user's history
            chat_found = False
            for i, chat in enumerate(users[self.id]['chat_history']):
                if chat['id'] == chat_id:
                    # Initialize messages array if it doesn't exist
                    if 'messages' not in chat:
                        chat['messages'] = []
                    
                    # Add message to chat
                    message_entry = {
                        'content': message,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'is_user': is_user
                    }
                    
                    # Add metadata if provided (for assistant responses with result files, etc.)
                    if metadata and not is_user:
                        message_entry['metadata'] = metadata
                    
                    chat['messages'].append(message_entry)
                    
                    # Update the chat in the user's local chat_history
                    for local_chat in self.chat_history:
                        if local_chat['id'] == chat_id:
                            if 'messages' not in local_chat:
                                local_chat['messages'] = []
                            local_chat['messages'].append(message_entry)
                            break
                    
                    chat_found = True
                    break
            
            if chat_found:
                # Save to file
                save_users(users)
                return True
        return False
        
    def get_chat_messages(self, chat_id):
        """Get all messages for a specific chat."""
        for chat in self.chat_history:
            if chat['id'] == chat_id:
                return chat.get('messages', [])
        return []
        
    def delete_chat(self, chat_id):
        """Delete a chat by its ID."""
        for i, chat in enumerate(self.chat_history):
            if chat['id'] == chat_id:
                self.chat_history.pop(i)
                users = load_users()
                if self.id in users:
                    for j, user_chat in enumerate(users[self.id]['chat_history']):
                        if user_chat['id'] == chat_id:
                            users[self.id]['chat_history'].pop(j)
                            save_users(users)
                            break
                return True
        return False

    @staticmethod
    def get(user_id):
        users = load_users()
        if user_id in users:
            user_data = users[user_id]
            # Print debug info
            print(f"Loading user {user_id} with data: {user_data}")
            print(f"Chat history in data: {user_data.get('chat_history', [])}")
            
            # Ensure chat_history exists
            if 'chat_history' not in user_data:
                user_data['chat_history'] = []
                save_users(users)
                
            return User(
                id=user_id,
                username=user_data['username'],
                email=user_data['email'],
                password_hash=user_data['password_hash'],
                chat_history=user_data.get('chat_history', [])
            )
        return None

    @staticmethod
    def get_by_email(email):
        users = load_users()
        for user_id, user_data in users.items():
            if user_data['email'] == email:
                return User(
                    id=user_id,
                    username=user_data['username'],
                    email=user_data['email'],
                    password_hash=user_data['password_hash'],
                    chat_history=user_data.get('chat_history', [])
                )
        return None

    @staticmethod
    def create(username, email, password):
        users = load_users()
        
        # Check if email already exists
        for user_data in users.values():
            if user_data['email'] == email:
                return None, 'Email already registered'
        
        # Generate new user ID
        user_id = str(len(users) + 1)
        password_hash = generate_password_hash(password)
        
        # Create new user
        users[user_id] = {
            'username': username,
            'email': email,
            'password_hash': password_hash,
            'chat_history': []
        }
        
        # Save to file
        save_users(users)
        
        return User(
            id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            chat_history=[]
        ), None
