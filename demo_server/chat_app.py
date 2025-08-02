"""
Chat Interface for Medical Report Processing

This module provides a Flask-based chat interface for interacting with the MCP tools.
It automatically detects which tool to call based on user queries and supports file uploads.
"""

import os
import json
import asyncio
import tempfile
import atexit
import websockets
import uuid
import time
import re
import logging
import base64
import traceback
import sys
from contextlib import asynccontextmanager


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MCP WebSocket client for direct communication with MCP server
class MCPClient:
    def __init__(self, websocket_url="ws://localhost:8765"):
        self.websocket_url = websocket_url
        self.websocket = None
        
    async def connect(self):
        """Connect to the MCP WebSocket server"""
        try:
            self.websocket = await websockets.connect(self.websocket_url)
            logger.info("Connected to MCP WebSocket server")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {str(e)}")
            return False
    
    async def call_tool(self, tool_name, **kwargs):
        """Call a tool on the MCP server"""
        if not self.websocket:
            raise ValueError("Not connected to MCP server")
            
        # Handle file_path specially - read the file and encode as base64
        if 'file_path' in kwargs and os.path.exists(kwargs['file_path']):
            with open(kwargs['file_path'], 'rb') as f:
                file_content = f.read()
                kwargs['file_content_base64'] = base64.b64encode(file_content).decode('utf-8')
                kwargs['file_name'] = os.path.basename(kwargs['file_path'])
        
        # Prepare the request message
        request = {
            "action": "call_tool",
            "tool_name": tool_name,
            "args": kwargs
        }
        
        # Send the request
        await self.websocket.send(json.dumps(request))
        logger.info(f"Sent request to call tool: {tool_name}")
        
        # Wait for the response
        response = await self.websocket.recv()
        response_data = json.loads(response)
        
        if response_data.get('status') == 'error':
            logger.error(f"Error from MCP server: {response_data.get('error')}")
            raise Exception(response_data.get('error', 'Unknown error from MCP server'))
            
        return response_data.get('result')
    
    async def close(self):
        """Close the WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            logger.info("Closed connection to MCP server")

from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_from_directory, Response
from flask_login import LoginManager, current_user, login_required, login_user, logout_user
from db_users import User  # Using PostgreSQL User class
import psycopg2
import psycopg2.extras
from db import get_db_connection  # Database connection utility
from db_results import ResultManager  # Using PostgreSQL Results management
from werkzeug.utils import secure_filename
from contextlib import AsyncExitStack

# MCP client imports
from mcp import ClientSession, StdioServerParameters
import google.generativeai as genai
from google.generativeai import types

# Configure Google API
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyDDBBTclRJjECny3q01Y57TIG9C6ZfVuTY")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini model for chat
CHAT_MODEL = genai.GenerativeModel('gemini-2.0-flash')

# MCP Server configuration
MCP_SERVER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")

MCP_WEBSOCKET_URL = "ws://localhost:8765"

# Celery initialization flag
celery_initialized = False

# Try to initialize Celery
try:
    from tasks import celery_app
    celery_initialized = True
    logger.info("Celery initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Celery: {e}")
    celery_initialized = False

# MCP Client class for the chat app
class MCPClient:
    def __init__(self):
        self.ws    = None
        self.tools = []

    async def connect(self):
        try:
            self.ws = await websockets.connect(MCP_WEBSOCKET_URL)
            greeting = await self.ws.recv()
            info = json.loads(greeting)
            self.tools = info.get("tools", [])
            print(f"[CLIENT] Connected; received {len(self.tools)} tool schemas")
            return True
        except Exception as e:
            print(f"[CLIENT] Connection error: {str(e)}")
            return False

    async def list_tools(self):
        """Fetch the list of tools via WebSocket RPC."""
        # The tools are already provided in the initial connected event
        # No need to send a separate request
        if not self.tools:
            # If tools weren't loaded during connect, try a ping to see if server is alive
            rid = str(uuid.uuid4())
            await self.ws.send(json.dumps({
                "action": "ping",
                "request_id": rid
            }))
            # Wait for any response
            await self.ws.recv()
            
        # Return the tools we already have
        return self.tools

    async def call_tool(self, tool_name, **kwargs):
        """Call an MCP tool over WebSocket and wait for its result."""
        if not self.ws:
            raise RuntimeError("WebSocket is not connected")
        
        # Convert any non-serializable objects to dictionaries
        serializable_args = {}
        for key, value in kwargs.items():
            if hasattr(value, '__dict__'):
                # Try to convert to dict if it's an object
                try:
                    serializable_args[key] = dict(value)
                except Exception:
                    serializable_args[key] = str(value)
            else:
                serializable_args[key] = value
        
        request_id = str(uuid.uuid4())
        payload = {
            "action": "call_tool",
            "tool_name": tool_name,
            "args": serializable_args,
            "request_id": request_id
        }
        print(f"[CLIENT] ▶ {payload}")
        
        try:
            await self.ws.send(json.dumps(payload))
            
            # collect until we see the matching tool_result with timeout
            start_time = time.time()
            timeout = 120  # 30 seconds timeout
            
            while True:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Timeout waiting for tool result after {timeout} seconds")
                    
                raw = await self.ws.recv()
                msg = json.loads(raw)
                
                # Handle tool result
                if msg.get("event") == "tool_result" and msg.get("request_id") == request_id:
                    print(f"[CLIENT] ◀ Tool result received")
                    return msg["result"]
                    
                # Handle tool error
                elif msg.get("event") == "tool_error" and msg.get("request_id") == request_id:
                    print(f"[CLIENT] ◀ Tool error received: {msg.get('error')}")
                    error_msg = msg.get('error', 'Unknown error')
                    raise RuntimeError(f"Tool execution failed: {error_msg}")
                    
                # Handle progress updates
                elif msg.get("event") == "progress":
                    print(f"[CLIENT] Progress: {msg.get('current')}/{msg.get('total')}")
                    # Continue waiting for the final result
        except Exception as e:
            print(f"[CLIENT] Error during tool call: {str(e)}")
            raise
 
    async def close(self):
        """Tear down the WebSocket connection."""
        if self.ws:
            try:
                # Just try to close it without checking closed attribute
                await self.ws.close()
            except Exception as e:
                print(f"Error closing WebSocket: {str(e)}")
            finally:
                self.ws = None

# Initialize Flask application
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)  # For flash messages and session

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chat_uploads')
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chat_results')
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

# Create necessary directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def save_result_to_json(result, filename_base, tool_name, chat_id=None, batch_id=None):
    """Save results to the database and return a unique filename"""
    timestamp = get_timestamp()
    json_filename = f"{filename_base}_{tool_name}_{timestamp}.json"
    
    # Determine processing type based on tool name
    if 'summarize' in tool_name.lower():
        processing_type = 'summary'
    else:
        processing_type = 'process'
    
    # Save to database if user is authenticated and we have a valid chat_id
    if current_user.is_authenticated:
        try:
            # Only save to database if we have a valid chat_id
            if chat_id:
                ResultManager.save_result(
                    user_id=current_user.id,
                    filename=json_filename,
                    original_filename=filename_base,
                    processing_type=processing_type,
                    tool_used=tool_name,
                    result_data=result,
                    chat_id=chat_id
                )
                app.logger.info(f"Saved result to database with chat_id: {chat_id}")
            else:
                app.logger.warning("Skipping database save: No chat_id provided")
        except Exception as e:
            app.logger.error(f"Error saving to database: {str(e)}")
    
    # Always save to file system for backup
    try:
        json_path = os.path.join(RESULTS_FOLDER, json_filename)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        app.logger.info(f"Saved result to file: {json_path}")
    except Exception as e:
        app.logger.error(f"Error saving result to file: {str(e)}")
        raise  # Re-raise the exception to handle it in the calling function
    
    # If this is part of a batch, store in the batch folder as well
    if batch_id:
        try:
            batch_results_folder = os.path.join(app.static_folder, 'results', 'batches', batch_id)
            os.makedirs(batch_results_folder, exist_ok=True)
            batch_json_path = os.path.join(batch_results_folder, json_filename)
            with open(batch_json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            app.logger.info(f"Saved batch result to {batch_json_path}")
        except Exception as e:
            app.logger.error(f"Error saving batch result: {str(e)}")
    
    return json_filename

async def get_tool_with_gemini(message, has_file=False, session=None):
    """
    Use Gemini to determine which MCP tool to call based on the user's message.

    This function:
    1. Connects to the MCP server to dynamically fetch available tools.
    2. Cleans each tool's schema recursively to ensure Gemini compatibility (removing problematic fields).
    3. Constructs a Gemini-compatible function declaration list.
    4. Sends the user message and tool schemas to Gemini with an explicit prompt to encourage function calling.
    5. Returns the selected tool name, confidence, and tool arguments.

    Args:
        message (str): The user's request or query.
        has_file (bool): Whether a file is attached to the request.
        session: Optional session object for advanced usage.
    Returns:
        tuple: (tool_name, confidence, tool_args)
    """
    print("\n[GEMINI] Analyzing user request...")
    try:
        # Step 1: Connect to MCP and fetch available tools
        client = MCPClient()
        await client.connect()
        tools = await client.list_tools()
        try:
                # 2) build Gemini function declarations with schema cleaning
            def clean_schema(schema):
                """Recursively clean schema to remove fields that cause Gemini validation errors"""
                if not isinstance(schema, dict):
                    return schema
                    
                # Remove problematic fields
                problematic_fields = ['default', 'title', 'additionalProperties', '$schema']
                cleaned = {k: v for k, v in schema.items() if k not in problematic_fields}
                
                # Recursively clean nested properties
                if 'properties' in cleaned and isinstance(cleaned['properties'], dict):
                    for prop_name, prop_schema in cleaned['properties'].items():
                        cleaned['properties'][prop_name] = clean_schema(prop_schema)
                        
                return cleaned
            
            fn_decls = [{
                "name": t["name"],
                "description": t["description"],
                "parameters": clean_schema(t["parameters"]),
            } for t in tools]
            gemini_tools = [types.Tool(function_declarations=fn_decls)]
            prompt = f"""
            You are an intelligent assistant for medical report processing.

            You have access to the following tools (functions), each with a name, description, and parameters.
            Your job is to:
            - Carefully read the user's request.
            - Select the single most appropriate tool from the provided list to fulfill the request.
            - Call that tool with the correct parameters, using only the information provided by the user.
            - If any required parameter is missing, make a best effort to infer it from the user's request, or leave it blank/null if truly unavailable.

            Guidelines:
            - Do NOT explain your reasoning or output any text other than the function call.
            - Do NOT summarize, paraphrase, or rephrase the user's request.
            - Do NOT call more than one tool at a time.
            - If the user's request cannot be handled by any available tool, respond with a function call to the tool that is the closest fit.

            TOOLS:
            (The available tools, their descriptions, and parameters are provided below.)

            USER REQUEST:
            '{message}'

            Respond ONLY with the function call in the required format.
            """
            print("[GEMINI] Calling Gemini with tool selection capabilities...")
            model = genai.GenerativeModel(
                "gemini-2.0-flash",
                generation_config=genai.GenerationConfig(
                    temperature=0,
                    top_p=0.95
                )
            )
            # Use a simpler approach without ToolConfig

            response = model.generate_content(
                prompt,
                tools=gemini_tools
            )
            # ... (rest of the function remains the same)
            # Print basic debug info
            print(f"[DEBUG] Response type: {type(response)}")
            
            # Check for a function call using the structure from your friend's code
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    content = candidate.content
                    if hasattr(content, 'parts') and len(content.parts) > 0:
                        part = content.parts[0]
                        
                        # Check if there's a function call
                        if hasattr(part, 'function_call') and part.function_call:
                            function_call = part.function_call
                            print(f"[GEMINI] Function to call: {function_call.name}")
                            print(f"[GEMINI] Arguments: {function_call.args}")
                            
                            tool_name = function_call.name
                            
                            # Extract arguments if any
                            tool_args = {}
                            if hasattr(function_call, 'args') and function_call.args:
                                # Convert MapComposite to regular dict
                                try:
                                    # Try to convert to dict if it's a MapComposite
                                    tool_args = dict(function_call.args)
                                    print(f"[GEMINI] Converted arguments to dict: {tool_args}")
                                except Exception as e:
                                    print(f"[GEMINI] Error converting arguments to dict: {str(e)}")
                                    # Fall back to empty dict
                                    tool_args = {}
                            
                            # Ensure file_path is included
                            if "file_path" not in tool_args:
                                tool_args["file_path"] = ""
                                
                            # Add extraction_method for process_medical_report if not present
                            if tool_name == "process_medical_report" and "extraction_method" not in tool_args:
                                tool_args["extraction_method"] = "auto"
                                
                            return tool_name, 0.95, tool_args
                        
                        # Fallback to text response if no function call
                        elif hasattr(part, 'text'):
                            text = part.text.strip()
                            print(f"[GEMINI] No function call found. Text response: {text}")
                            
                            # Check if the text contains a tool name
                            if "summarize_medical_report" in text:
                                print(f"[GEMINI] Tool selected via text response: summarize_medical_report")
                                return "summarize_medical_report", 0.95, {"file_path": ""}
                            elif "process_medical_report" in text:
                                print(f"[GEMINI] Tool selected via text response: process_medical_report")
                                return "process_medical_report", 0.95, {"file_path": "", "extraction_method": "auto"}
                            else:
                                print(f"[GEMINI] Could not determine tool from text response")
            
            # If we couldn't determine the tool from the response
            print("[GEMINI] Could not determine tool from Gemini response, falling back to classification approach")
        finally:
            # Close the temporary client
            await client.close()
            
        # If we got here, Gemini didn't select a tool via function call, fall back to classification approach
        print("[GEMINI] No tool selected via function call, falling back to classification approach")
        
        # Enhanced prompt for Gemini to classify the request with better handling of unclear cases
        prompt = f"""
        Determine if the user wants to summarize or process a medical report based on their message.
        Message: "{message}"
        
        SUMMARIZE: If the user mentions summarization, summary, brief overview, condensed information, key points, highlights, or wants a concise version.
        PROCESS: If the user wants detailed processing, analysis, extraction of specific information, verification, enhancement, or in-depth examination.
        
        If the message is unclear, analyze the context and user intent:
        - If the message suggests the user wants quick information or is short on time, choose "summarize".
        - If the message suggests the user needs comprehensive information or detailed analysis, choose "process".
        - If the message contains specific questions about the report content, choose "process".
        - If the message is completely ambiguous, choose "process" as the default.
        
        Respond with ONLY one word: "summarize" or "process".
        """
        
        # Call Gemini for classification using the correct method
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0,
                top_p=0.95,
                top_k=0
            )
        )
        
        result = response.text.strip().lower()
        print(f"Gemini classification result: {result}")
        
        # Determine tool based on classification
        if "summarize" in result:
            return "summarize_medical_report", 0.9, {"file_path": ""}
        else:
            return "process_medical_report", 0.8, {"file_path": "", "extraction_method": "auto"}
            
    except Exception as e:
        print(f"[ERROR] Error using Gemini for tool selection: {str(e)}")
        print("[FALLBACK] Using keyword-based tool detection instead")
        
        # Fallback to keyword-based detection
        message = message.lower()
        
        # Check for summarization keywords
        if any(keyword in message for keyword in ["summarize", "summary", "brief", "overview", "short"]):
            return "summarize_medical_report", 0.7, {"file_path": ""}
        
        # Check for processing keywords
        elif any(keyword in message for keyword in ["process", "analyze", "extract", "detail", "full"]):
            return "process_medical_report", 0.7, {"file_path": "", "extraction_method": "auto"}
        
        # Default to process_medical_report
        else:
            return "process_medical_report", 0.5, {"file_path": "", "extraction_method": "auto"}

async def answer_general_question(message):
    """Handle general questions without file uploads"""
    try:
        # Check for batch processing requests with various patterns
        batch_match = re.search(r'extract\s+(?:these\s+)?(?:reports|files|text)\s+(.+)', message, re.IGNORECASE)
        if batch_match:
            # This is a batch processing request, handle it directly here
            directory_path = batch_match.group(1).strip()
            app.logger.info(f"Detected batch processing request in general question handler: {directory_path}")
            
            # Check if the directory exists
            if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
                return f"Sorry, the directory '{directory_path}' does not exist or is not accessible.", None, None
            
            # Get all valid files in the directory
            valid_files = []
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                if os.path.isfile(file_path) and allowed_file(filename):
                    valid_files.append(file_path)
            
            app.logger.info(f"Found {len(valid_files)} valid files in directory: {directory_path}")
            
            if not valid_files:
                return f"No valid files found in the directory '{directory_path}'.", None, None
            
            # Determine the appropriate tool using get_tool_with_gemini
            try:
                # Extract the command part (e.g., "extract text")
                command_part = message[:batch_match.start(1)].strip()
                app.logger.info(f"Command part for tool selection: {command_part}")
                
                # Use get_tool_with_gemini to analyze the command and select the appropriate tool
                tool_name, confidence, tool_args = await get_tool_with_gemini(command_part, has_file=True)
                app.logger.info(f"Selected tool: {tool_name} with confidence {confidence}")
            except Exception as e:
                app.logger.warning(f"Error determining tool with Gemini: {str(e)}")
                # Fall back to default tool if get_tool_with_gemini fails
                tool_name = "process_medical_report"  # Default tool
                tool_args = {}
                app.logger.info(f"Falling back to default tool: {tool_name}")
                
            # If confidence is too low or no tool was selected, use the default
            if not tool_name or confidence < 0.4:
                tool_name = "process_medical_report"
                tool_args = {}
                app.logger.info(f"Using default tool due to low confidence: {tool_name}")
            
            # Generate unique batch ID for tracking
            batch_id = str(uuid.uuid4())
            app.logger.info(f"Generated batch ID: {batch_id}")
            
            # Start batch processing with detailed logging
            app.logger.info(f"Starting batch processing of {len(valid_files)} files with tool: {tool_name}")
            for i, file_path in enumerate(valid_files):
                app.logger.info(f"  File {i+1}: {os.path.basename(file_path)}")
            
            # Generate a batch ID for tracking
            batch_id = str(uuid.uuid4())
            app.logger.info(f"Generated batch ID: {batch_id}")
            
            # First, check if Celery is properly initialized
            if not celery_initialized:
                app.logger.warning("Celery not initialized - falling back to direct processing")
                raise ConnectionError("Celery not initialized")
                
            # Try to use Celery for asynchronous processing
            try:
                app.logger.info("Attempting to use Celery for batch processing...")
                
                # Import tasks directly
                try:
                    import tasks
                    from tasks import process_batch_task
                    app.logger.info("Tasks module imported successfully")
                except ImportError as ie:
                    app.logger.error(f"Failed to import tasks: {str(ie)}")
                    raise ConnectionError(f"Failed to import tasks: {str(ie)}")
                    
                # Skip connection testing since we've verified Redis is running
                app.logger.info("Proceeding with batch task creation")
                
                # We'll catch any Celery connection errors in the task.delay() call below
                # rather than pre-testing with ping
                
                # process_batch_task is already imported above
                
                # Create the batch processing task
                batch_task = process_batch_task.delay(
                    file_paths=valid_files,
                    tool_name=tool_name,
                    tool_args=None if not tool_args else tool_args,  # Handle None case explicitly
                    user_id=current_user.id if current_user.is_authenticated else None,
                    chat_id=None
                )
                
                app.logger.info(f"Celery task created with ID: {batch_task.id}")
                task_id = batch_task.id
                
                # Save basic batch info locally too for tracking
                if not hasattr(app, 'batch_tracking'):
                    app.batch_tracking = {}
                app.batch_tracking[batch_id] = {
                    "id": batch_id,
                    "celery_task_id": batch_task.id,
                    "total_files": len(valid_files),
                    "status": "processing",
                    "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Save batch job to database
                try:
                    with get_db_connection() as conn:
                        cursor = conn.cursor()
                        # First check if the batch_jobs table exists
                        cursor.execute(
                            """SELECT EXISTS (
                               SELECT FROM information_schema.tables 
                               WHERE table_schema = 'public' 
                               AND table_name = 'batch_jobs'
                            )"""
                        )
                        table_exists = cursor.fetchone()[0]
                        
                        if not table_exists:
                            # Create the batch_jobs table if it doesn't exist
                            cursor.execute(
                                """CREATE TABLE IF NOT EXISTS batch_jobs (
                                    id SERIAL PRIMARY KEY,
                                    user_id INTEGER NOT NULL,
                                    status VARCHAR(50) NOT NULL DEFAULT 'processing',
                                    processed INTEGER NOT NULL DEFAULT 0,
                                    failed INTEGER NOT NULL DEFAULT 0,
                                    total_files INTEGER NOT NULL DEFAULT 0,
                                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                                    metadata JSONB
                                )"""
                            )
                            conn.commit()
                        
                        # Insert the batch job record using the existing schema
                        cursor.execute(
                            """INSERT INTO batch_jobs 
                               (user_id, status, total_files, processed, failed, metadata) 
                               VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
                            (current_user.id, "processing", len(valid_files), 0, 0, {'batch_id': batch_id})
                        )
                        conn.commit()
                        app.logger.info(f"Saved batch job {batch_id} to database")
                except Exception as e:
                    app.logger.error(f"Error saving batch job to database: {e}")
                    # Continue processing even if database save fails
                
                # Return early since Celery will handle processing
                response_text = f"Started async batch processing of {len(valid_files)} files. Results will be available soon. Batch ID: {batch_id}"
                
                # Create a proper response object
                response_data = {
                    "response": response_text,
                    "status": "batch_processing",
                    "batch_id": batch_id,
                    "total_files": len(valid_files),
                    "task_id": batch_task.id,
                    "timestamp": datetime.now().isoformat()
                }
                
                return jsonify(response_data)
                
            except Exception as e:
                # Fallback to direct processing if Celery fails
                app.logger.warning(f"Celery batch processing failed: {str(e)}")
                app.logger.info("Falling back to direct processing...")
                
                # Process files in batches to avoid timeout
                # First batch: Process up to 3 files immediately for quick feedback
                first_batch = valid_files[:min(3, len(valid_files))]
                remaining_files = valid_files[min(3, len(valid_files)):]
                
                app.logger.info(f"Processing first batch of {len(first_batch)} files directly")
                results = []
                processed_count = 0
                successful_results = []
                
                # Create a simple structure to track batch processing
                batch_info = {
                    "id": batch_id,
                    "total_files": len(valid_files),
                    "processed": 0,
                    "successful": 0,
                    "failed": 0,
                    "results": [],
                    "status": "processing",
                    "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Process first batch of files directly
                for file_path in first_batch:
                    try:
                        file_name = os.path.basename(file_path)
                        app.logger.info(f"Processing file: {file_name}")
                        
                        # Process the file directly
                        result = await process_file(file_path, tool_name, tool_args, chat_id=None)
                        
                        if result and not isinstance(result, tuple):
                            # Likely an error message string
                            app.logger.warning(f"Unexpected result format: {result}")
                            batch_info["processed"] += 1
                            batch_info["failed"] += 1
                            batch_info["results"].append({
                                "file": file_name,
                                "success": False,
                                "error": str(result) if result else "Unknown error"
                            })
                        elif result and isinstance(result, tuple) and len(result) >= 2 and result[1] is not None:
                            # Successful processing
                            processed_count += 1
                            batch_info["processed"] += 1
                            batch_info["successful"] += 1
                            
                            # Use result[1] for the actual data result
                            batch_info["results"].append({
                                "file": file_name,
                                "success": True
                            })
                        else:
                            # Error during processing
                            batch_info["processed"] += 1
                            batch_info["failed"] += 1
                            batch_info["results"].append({
                                "file": file_name,
                                "success": False,
                                "error": "Failed to process file"
                            })
                            
                    except Exception as process_error:
                        app.logger.error(f"Error processing file {file_path}: {str(process_error)}")
                        import traceback
                        app.logger.error(traceback.format_exc())
                        
                        batch_info["processed"] += 1
                        batch_info["failed"] += 1
                        batch_info["results"].append({
                            "file": os.path.basename(file_path),
                            "success": False,
                            "error": str(process_error)
                        })
                
                # Update status based on processing results
                if batch_info["processed"] == batch_info["total_files"]:
                    batch_info["status"] = "complete"
                else:
                    batch_info["status"] = "partially_complete"
                
                # Store batch info in the app config temporarily
                if not hasattr(app, 'batch_tracking'):
                    app.batch_tracking = {}
                app.batch_tracking[batch_id] = batch_info
                
                # Save batch info to a JSON file for persistence
                batch_dir = os.path.join(app.static_folder, 'results', 'batches')
                os.makedirs(batch_dir, exist_ok=True)
                batch_file = os.path.join(batch_dir, f"{batch_id}.json")
                
                with open(batch_file, 'w') as f:
                    json.dump(batch_info, f, indent=2)
                
                app.logger.info(f"Processed {processed_count} files. Successful: {batch_info['successful']}, Failed: {batch_info['failed']}")
                app.logger.info(f"Batch info saved to {batch_file}")
                
                # Set task ID to batch ID
                task_id = batch_id
            
            # Log information about remaining files
            if remaining_files:
                app.logger.info(f"{len(remaining_files)} files remaining to be processed")
                app.logger.info("User will need to process these files separately or in smaller batches")
            
            # Create a user-friendly response
            response = f"Started batch processing of {len(valid_files)} files from '{directory_path}'."
            response += f"\n\nI'll analyze these files using the {tool_name} tool."
            response += f"\n\nBatch ID: {task_id}"
            response += "\n\nYou can view the results when processing is complete."
            
            # Create a structured result for the frontend
            tool_result = {
                "batch_id": task_id,
                "file_count": len(valid_files),
                "status": "processing",
                "is_batch": True,
                "view_url": f"/batch/results/{task_id}"
            }
            
            return response, tool_result, None
        
        # If not a batch processing request, proceed with normal general question handling
        # Connect to MCP WebSocket server
        async with websockets.connect(MCP_WEBSOCKET_URL) as ws:
            # Receive greeting
            greeting = await ws.recv()
            greeting_data = json.loads(greeting)
            
            # Send message to get tool selection
            await ws.send(json.dumps({
                "action": "select_tool",
                "message": message,
                "request_id": str(uuid.uuid4())
            }))
            
            # Get tool selection response
            tool_selection = await ws.recv()
            tool_data = json.loads(tool_selection)
            
            if tool_data.get("event") == "tool_selected":
                tool_name = tool_data.get("tool_name")
                confidence = tool_data.get("confidence", 0.0)
                
                # For now, just return a generic response
                response = "It looks like you're trying to provide a file path. I can't directly access files on your computer.\n\n"
                response += "However, if you want me to help you understand the content of a medical report, you have a few options:\n\n"
                response += "1. Upload the medical report for processing. This will give you a detailed analysis of the report.\n"
                response += "2. Upload the medical report for summarization. This will give you a quick overview of the report.\n"
                response += "3. Ask me questions about medical terminology or report interpretation.\n\n"
                response += "You can also view past results by clicking on the 'View Results' link."
                
                return response, None, None
            else:
                return "I'm not sure how to help with that. Could you try uploading a medical report?", None, None
    except Exception as e:
        app.logger.error(f"Error in answer_general_question: {str(e)}")
        import traceback
        app.logger.error(traceback.format_exc())
        return f"Sorry, I encountered an error: {str(e)}", None, None

async def process_file(file_path, tool_name, tool_args=None, chat_id=None):
    """Process a file using the specified tool through MCP client"""
    
    # Initialize client as None so finally block doesn't fail
    client = None
    
    try:
        app.logger.info(f"Processing file {os.path.basename(file_path)} with tool {tool_name}")
        
        # Initialize default tool args if None
        if tool_args is None:
            tool_args = {}
        
        # Create MCP client and connect to server
        client = MCPClient()
        connected = await client.connect()
        
        if not connected:
            error_msg = "Failed to connect to MCP server"
            app.logger.error(error_msg)
            return error_msg, None, None
        
        # Call the tool with the file path
        app.logger.info(f"Calling tool {tool_name} with file {os.path.basename(file_path)}")
        
        # Check if file_path is already in tool_args and remove it if present
        tool_args_copy = tool_args.copy() if tool_args else {}
        if 'file_path' in tool_args_copy:
            app.logger.info("Removing duplicate file_path from tool_args")
            del tool_args_copy['file_path']
            
        result = await client.call_tool(tool_name, file_path=file_path, **tool_args_copy)
        app.logger.info(f"Successfully processed file with {tool_name}")
        
        # Generate a user-friendly response
        if tool_name == "summarize_medical_report":
            response = "I've summarized your medical report. "
            if "patient_name" in result and result["patient_name"] and result["patient_name"] != "UNKNOWN":
                response += f"Patient: {result['patient_name']}. "
            if "summary" in result and result["summary"]:
                response += f"\n\n{result['summary']}"
        else:  # process_medical_report
            response = "I've processed your medical report. "
            if "report_type" in result and result["report_type"]:
                response += f"Report type: {result['report_type']}. "
            if "patient_name" in result and result["patient_name"] and result["patient_name"] != "UNKNOWN":
                response += f"Patient: {result['patient_name']}. "
            if "summary" in result and result["summary"]:
                response += f"\n\n{result['summary']}"
        
        # Save the result to JSON
        filename_base = os.path.splitext(os.path.basename(file_path))[0]
        json_filename = save_result_to_json(result, filename_base, tool_name, chat_id)
        return response, result, json_filename
        
    except Exception as e:
        error_msg = f"Error processing file: {str(e)}"
        app.logger.error(error_msg)
        app.logger.error(traceback.format_exc())
        return error_msg, None, None
        
    finally:
        # Always close the WebSocket connection
        if client:
            try:
                await client.close()
                app.logger.info("WebSocket connection closed")
            except Exception as close_error:
                app.logger.error(f"Error closing WebSocket: {str(close_error)}")
                app.logger.error(traceback.format_exc())

# Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Debug
        print(f"Login attempt for email: {email}")
        
        user = User.get_by_email(email)
        if user:
            print(f"Found user: {user.username}, ID: {user.id}")
            print(f"User chat history: {user.chat_history}")
            
        if user and user.verify_password(password):
            login_user(user)
            print(f"Logged in user: {current_user.username}, ID: {current_user.id}")
            print(f"Current user chat history: {current_user.chat_history}")
            
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        
        flash('Invalid email or password')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match')
            return render_template('signup.html')
        
        user, error = User.create(username, email, password)
        if error:
            flash(error)
            return render_template('signup.html')
        
        login_user(user)
        return redirect(url_for('index'))
    
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    # Get current time for the welcome message
    now = datetime.now().strftime('%H:%M')
    return render_template('chat_daisy.html', now=now, current_user=current_user)

@app.route('/results')
@login_required
def view_results():
    try:
        app.logger.debug(f"Fetching results for user {current_user.id}")
        # Get results from database for the current user
        results = ResultManager.get_results_by_user(current_user.id)
        app.logger.debug(f"Retrieved {len(results)} results for user {current_user.id}")
        
        # Format results for display
        formatted_results = []
        for result in results:
            try:
                # Safely handle the result object
                if not hasattr(result, 'id') or not hasattr(result, 'filename'):
                    app.logger.error(f"Invalid result object: {result}")
                    continue
                    
                # Safely format the created timestamp
                created = 'Unknown'
                if hasattr(result, 'created_at') and result.created_at:
                    try:
                        created = result.created_at.strftime('%Y-%m-%d %H:%M:%S')
                    except Exception as e:
                        app.logger.error(f"Error formatting created_at: {e}")
                
                # Calculate file size
                size = 0
                try:
                    if hasattr(result, 'result_data') and result.result_data:
                        # Estimate size based on JSON data
                        size = len(str(result.result_data))
                except Exception as e:
                    app.logger.error(f"Error calculating size: {e}")
                
                formatted_results.append({
                    'id': result.id,
                    'filename': result.filename,
                    'type': getattr(result, 'processing_type', 'unknown'),
                    'created': created,
                    'original_filename': getattr(result, 'original_filename', 'unknown'),
                    'size': size
                })
                
            except Exception as e:
                app.logger.error(f"Error formatting result {getattr(result, 'id', 'unknown')}: {e}", exc_info=True)
                continue
        
        # Get batch jobs for the current user
        batches = []
        try:
            # Query database for batch jobs
            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                # First, check if the batch_jobs table exists
                cursor.execute(
                    """SELECT EXISTS (
                       SELECT FROM information_schema.tables 
                       WHERE table_schema = 'public' 
                       AND table_name = 'batch_jobs'
                    )"""
                )
                table_exists = cursor.fetchone()[0]
                
                if table_exists:
                    # Query the batch_jobs table with the actual schema
                    cursor.execute(
                        """SELECT 
                              id, 
                              metadata->>'batch_id' as name, 
                              status, 
                              created_at, 
                              created_at as updated_at, 
                              total_files, 
                              processed, 
                              failed,
                              FALSE as is_deleted
                           FROM batch_jobs 
                           WHERE user_id = %s 
                           ORDER BY created_at DESC""", 
                        (current_user.id,)
                    )
                else:
                    # If table doesn't exist, create it
                    app.logger.warning("batch_jobs table does not exist. Creating it now.")
                    cursor.execute(
                        """CREATE TABLE IF NOT EXISTS batch_jobs (
                            id SERIAL PRIMARY KEY,
                            batch_id TEXT NOT NULL,
                            user_id INTEGER NOT NULL,
                            status VARCHAR(50) NOT NULL DEFAULT 'processing',
                            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                            total_files INTEGER NOT NULL DEFAULT 0,
                            processed INTEGER NOT NULL DEFAULT 0,
                            failed INTEGER NOT NULL DEFAULT 0,
                            is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
                            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                        )"""
                    )
                    conn.commit()
                    # Return empty result set
                    cursor.execute("SELECT 1 WHERE FALSE")
                
                batch_rows = cursor.fetchall()
                
                # Format batch jobs for display
                for batch in batch_rows:
                    created_at = 'Unknown'
                    if batch['created_at']:
                        try:
                            created_at = batch['created_at'].strftime('%Y-%m-%d %H:%M:%S')
                        except Exception as e:
                            app.logger.error(f"Error formatting batch created_at: {e}")
                    
                    batches.append({
                        'id': batch['id'],
                        'name': batch['name'],
                        'status': batch['status'],
                        'created_at': created_at,
                        'total_files': batch['total_files'],
                        'processed': batch['processed'],
                        'failed': batch['failed']
                    })
        except Exception as e:
            app.logger.error(f"Error fetching batch jobs: {e}", exc_info=True)
        
        app.logger.debug(f"Rendering template with {len(formatted_results)} formatted results and {len(batches)} batch jobs")
        return render_template('results.html', results=formatted_results, batches=batches)
        
    except Exception as e:
        app.logger.error(f"Error in view_results: {e}", exc_info=True)
        flash("An error occurred while loading your results. Please try again.")
        return render_template('results.html', results=[], batches=[])

@app.route('/view_result/<filename>')
@login_required
def view_result(filename):
    # Get result from database
    result_obj = ResultManager.get_result_by_filename(filename, user_id=current_user.id)
    
    if not result_obj:
        # Try fallback to file system for backward compatibility
        file_path = os.path.join(RESULTS_FOLDER, filename)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
                return render_template('view_result.html', result=result_data, filename=filename)
        
        flash("Result not found")
        return redirect(url_for('view_results'))
    
    # Check if user has permission to view this result
    if result_obj.user_id != current_user.id and current_user.id != 1:  # User 1 is admin
        flash("You don't have permission to view this result")
        return redirect(url_for('view_results'))
    
    # Use the result data from the database
    result_data = result_obj.result_data
    
    # For now, use the old template - we'll create a DaisyUI version later if needed
    return render_template('view_result.html', result=result_data, filename=filename, result_id=result_obj.id)

@app.route('/download_result/<filename>')
def download_result(filename):
    return send_from_directory(RESULTS_FOLDER, filename, as_attachment=True)

@app.route('/delete_result/<filename>', methods=['POST'])
@login_required
def delete_result(filename):
    """Delete a result from the database"""
    # First try to get the result from the database to get its ID
    result_obj = ResultManager.get_result_by_filename(filename)
    
    if result_obj:
        # Delete from database using dot notation to access the id attribute
        success = ResultManager.delete_result(result_obj.id, current_user.id)
        if success:
            # Also try to delete the file for backward compatibility
            try:
                file_path = os.path.join(RESULTS_FOLDER, filename)
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                # Ignore file system errors as the database is the source of truth
                pass
            return jsonify({"success": True}), 200
        else:
            return jsonify({"error": "Failed to delete result"}), 500
    else:
        # Try fallback to file system
        file_path = os.path.join(RESULTS_FOLDER, filename)
        if not os.path.exists(file_path):
            return jsonify({"error": "Result not found"}), 404
        
        try:
            os.remove(file_path)
            return jsonify({"success": True}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    # Check content type and get data accordingly
    if request.is_json:
        data = request.json
        message = data.get('message', '')
        chat_id = data.get('chat_id', '')
        has_file = False
        file_path = None
    else:
        # Handle form data
        message = request.form.get('message', '')
        chat_id = request.form.get('chat_id', '')
        has_file = 'file' in request.files and request.files['file'].filename
    
    # Check if the message is empty
    if not message.strip():
        return jsonify({'error': 'Message cannot be empty'}), 400
    
    # Add the message to the chat
    current_user.add_message_to_chat(chat_id, 'user', message)
    
    # Check if the message is a batch processing command
    # Pattern: "<command> from <directory_path>"
    # Example: "extract text from C:\Users\Documents\Reports"
    batch_dir_pattern = re.compile(r'(.+)\s+from\s+(["\w:\\\s\/\.\-_]+)', re.IGNORECASE)
    batch_match = batch_dir_pattern.match(message.strip())
    
    # Handle file upload for form data
    if not request.is_json and has_file:
        file = request.files['file']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
        else:
            has_file = False
            file_path = None
    
    # Process the request
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Check if the message contains a directory path
        directory_path = None
        
        # Look for a path-like pattern in the message
        path_match = re.search(r'([A-Za-z]:[\\/](?:[^\\/]+[\\/])*[^\\/]+|/[^\s\\/]+(?:/[^\s\\/]+)*)', message)
        if path_match:
            directory_path = path_match.group(1).strip('"\'')
            # Verify if it's a directory
            if not os.path.isdir(directory_path):
                directory_path = None
        
        # Always use Gemini to determine the tool (summarize or extract)
        tool_name, confidence, tool_args = loop.run_until_complete(get_tool_with_gemini(message, has_file or bool(directory_path)))
        app.logger.info(f"Gemini selected tool: {tool_name} with confidence {confidence}")
        
        # If a directory path was found, process as batch
        if directory_path:
            # Store the tool name in the message for batch processing
            if not has_file:  # Only modify message if no file was uploaded
                message = f"{tool_name} {message}"
                
            # Process the directory as a batch
            batch_result = process_batch_from_directory(directory_path, message, chat_id)
            
            # Extract response data from the batch result
            response = batch_result.get('response', f"Started batch processing from '{directory_path}'")
            tool_result = {
                "batch_id": batch_result.get('batch_id'),
                "file_count": batch_result.get('total_files', 0),
                "status": batch_result.get('status', 'processing'),
                "is_batch": True
            }
            json_filename = None
            tool_name = batch_result.get('tool_name', tool_name)
        # Process single file if uploaded
        elif has_file:
            # Use Gemini to determine which tool to call
            tool_name, confidence, tool_args = loop.run_until_complete(get_tool_with_gemini(message, has_file))
            app.logger.info(f"Gemini selected tool: {tool_name} with confidence {confidence}")
            
            # Process the file with the selected tool
            response, tool_result, json_filename = loop.run_until_complete(
                process_file(file_path, tool_name, tool_args, chat_id)
            )
        else:
            # Handle general questions
            answer_result = loop.run_until_complete(answer_general_question(message))
            
            # Check if the response is already a complete Response object from jsonify
            if isinstance(answer_result, Response):
                # Just return the response directly
                return answer_result
            
            # Otherwise, unpack the response tuple
            if isinstance(answer_result, tuple) and len(answer_result) == 3:
                response, tool_result, json_filename = answer_result
            else:
                # Handle unexpected response format
                response = str(answer_result)
                tool_result = None
                json_filename = None
            
            tool_name = "general_question"
            confidence = 0.8
    except Exception as e:
        response = f"Sorry, there was an error processing your request: {str(e)}"
        tool_result = ""
        json_filename = None
        tool_name = "error"
        confidence = 0
    finally:
        loop.close()
    
    # Prepare metadata for the assistant response
    metadata = {
        "tool_used": tool_name,
        "confidence": confidence
    }
    
    # Add result file information if available
    if json_filename:
        metadata["view_url"] = f"/view_result/{json_filename}"
    
    # Add batch processing information if available
    if isinstance(tool_result, dict) and "batch_id" in tool_result:
        metadata["batch_id"] = tool_result["batch_id"]
        metadata["file_count"] = tool_result.get("file_count", 0)
        metadata["status"] = tool_result.get("status", "processing")
        metadata["view_batch_url"] = f"/batch/results/{tool_result['batch_id']}"
    
    # Add result file info to metadata if available
    if json_filename:
        metadata["result_file"] = json_filename
        metadata["view_url"] = f"/view_result/{json_filename}"
    
    # Store the assistant response in chat history with metadata
    current_user.add_message_to_chat(chat_id, response, is_user=False, metadata=metadata)
    
    result_data = {
        "response": response,
        "tool_used": tool_name,
        "confidence": confidence,
        "chat_id": chat_id
    }
    
    # Add result file info if available
    if json_filename:
        result_data["result_file"] = json_filename
        result_data["view_url"] = f"/view_result/{json_filename}"
    
    return jsonify(result_data)

@app.route('/view_chat/<chat_id>')
@login_required
def view_chat(chat_id):
    # Find the chat in user's history
    chat = None
    for item in current_user.chat_history:
        if item['id'] == chat_id:
            chat = item
            break
    
    if not chat:
        flash("Chat not found")
        return redirect(url_for('index'))
    
    # Get chat messages
    messages = current_user.get_chat_messages(chat_id)
    
    # Get current time for the welcome message
    now = datetime.now().strftime('%H:%M')
    
    return render_template('chat_daisy.html', 
                           now=now, 
                           current_user=current_user, 
                           chat_id=chat_id, 
                           chat_title=chat['title'],
                           messages=messages)

@app.route('/debug/chat_history')
@login_required
def debug_chat_history():
    return jsonify({
        'username': current_user.username,
        'user_id': current_user.id,
        'chat_history': current_user.chat_history
    })

@app.route('/delete_chat/<chat_id>', methods=['POST'])
@login_required
def delete_chat(chat_id):
    success = current_user.delete_chat(chat_id)
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({
            'success': success,
            'message': 'Chat deleted successfully' if success else 'Failed to delete chat'
        })
    else:
        # Fallback for non-AJAX requests
        if success:
            flash('Chat deleted successfully', 'success')
        else:
            flash('Failed to delete chat', 'error')
        return redirect(url_for('index'))

@app.route('/batch/upload', methods=['POST'])
@login_required
def batch_upload():
    """Handle batch upload of multiple files with enhanced error handling and progress tracking"""
    try:
        app.logger.info("=== BATCH UPLOAD REQUEST RECEIVED ===")
        
        # Check if request has files
        if not request.files:
            return jsonify({
                'status': 'error',
                'message': 'No files found in the request',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Get files from request
        files = []
        for key in request.files:
            files.extend(request.files.getlist(key))
        
        # Filter out empty files
        files = [f for f in files if f and f.filename and f.filename.strip()]
        
        if not files:
            return jsonify({
                'status': 'error',
                'message': 'No valid files found in the request',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Get form data
        tool_name = request.form.get('tool_name', 'process_medical_report')
        chat_id = request.form.get('chat_id')
        extraction_method = request.form.get('extraction_method')
        
        # Save and validate files
        valid_files = []
        invalid_files = []
        
        for file in files:
            try:
                if allowed_file(file.filename):
                    # Generate a unique filename
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    unique_id = str(uuid.uuid4())[:8]
                    original_filename = secure_filename(file.filename)
                    filename = f"{timestamp}_{unique_id}_{original_filename}"
                    
                    # Ensure upload directory exists
                    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                    
                    # Save the file
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                    valid_files.append(file_path)
                    app.logger.info(f"Saved file: {file_path}")
                else:
                    invalid_files.append(file.filename)
                    app.logger.warning(f"File type not allowed: {file.filename}")
            except Exception as e:
                app.logger.error(f"Error processing file {file.filename}: {str(e)}", exc_info=True)
                invalid_files.append(file.filename)
        
        if not valid_files:
            return jsonify({
                'status': 'error',
                'message': 'No valid files could be processed',
                'invalid_files': invalid_files,
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Prepare tool arguments
        tool_args = {}
        if extraction_method:
            tool_args['extraction_method'] = extraction_method
        
        # Start batch processing task
        from tasks import process_batch_task
        
        batch_task = process_batch_task.delay(
            file_paths=valid_files,
            tool_name=tool_name,
            tool_args=tool_args,
            user_id=current_user.id,
            chat_id=chat_id
        )
        
        app.logger.info(f"Started batch task {batch_task.id} for {len(valid_files)} files")
        
        # Return initial response
        response = {
            'status': 'processing',
            'batch_id': batch_task.id,
            'total_files': len(valid_files),
            'invalid_files': invalid_files,
            'message': f'Batch processing started for {len(valid_files)} files',
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 202
        
    except Exception as e:
        app.logger.error(f"Error in batch upload: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Error processing batch upload: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/batch/status/<batch_id>')
@login_required
def batch_status(batch_id):
    """Get the status of a batch processing job with detailed progress information"""
    try:
        from tasks import check_batch_status
        
        # Get the most up-to-date status from Celery
        task_status = check_batch_status.delay(batch_id, current_user.id).get(timeout=10)
        
        # Get batch results from the database
        results = ResultManager.get_results_by_batch(batch_id, current_user.id)
        
        # Format the response
        response = {
            'batch_id': batch_id,
            'status': task_status.get('status', 'unknown'),
            'progress': task_status.get('progress', 0),
            'processed': task_status.get('processed', 0),
            'total': task_status.get('total', 0),
            'success': task_status.get('success', 0),
            'failed': task_status.get('failed', 0),
            'start_time': task_status.get('start_time'),
            'end_time': task_status.get('end_time'),
            'tool_name': task_status.get('tool_name', 'Unknown'),
            'results': [],
            'file_results': task_status.get('file_results', {}),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add result details if available
        if results:
            response['results'] = [{
                'id': r.id,
                'filename': r.original_filename,
                'status': r.status,
                'created_at': r.created_at.isoformat() if r.created_at else None,
                'result_url': url_for('view_result', filename=r.filename) if r.filename else None
            } for r in results]
        
        # Add error details if batch failed
        if 'error' in task_status:
            response['error'] = task_status['error']
        
        return jsonify(response)
        
    except Exception as e:
        app.logger.error(f"Error getting batch status: {str(e)}", exc_info=True)
        return jsonify({
            'batch_id': batch_id,
            'status': 'error',
            'message': f'Error getting batch status: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/batch/results/<batch_id>')
@login_required
def batch_results(batch_id):
    """View detailed results for a batch processing job"""
    try:
        # Get batch status first to ensure it's up-to-date
        from tasks import check_batch_status
        task_status = check_batch_status.delay(batch_id, current_user.id).get(timeout=5)
        
        # Get batch results from database
        results = ResultManager.get_results_by_batch(batch_id, current_user.id)
        
        # Format results for display
        formatted_results = []
        for result in results:
            formatted_results.append({
                'id': result.id,
                'filename': result.original_filename or 'Unknown',
                'status': result.status,
                'created_at': result.created_at.strftime('%Y-%m-%d %H:%M:%S') if result.created_at else 'N/A',
                'result_url': url_for('view_result', filename=result.filename) if result.filename else None,
                'is_success': result.status.lower() in ('completed', 'success')
            })
        
        # Prepare summary data
        summary = {
            'batch_id': batch_id,
            'total_files': len(results),
            'success_count': sum(1 for r in results if r.status.lower() in ('completed', 'success')),
            'failed_count': sum(1 for r in results if r.status.lower() in ('failed', 'error')),
            'in_progress_count': sum(1 for r in results if r.status.lower() in ('processing', 'pending')),
            'start_time': task_status.get('start_time'),
            'end_time': task_status.get('end_time'),
            'tool_name': task_status.get('tool_name', 'Unknown'),
            'status': task_status.get('status', 'unknown'),
            'progress': task_status.get('progress', 0)
        }
        
        # Add error details if batch failed
        if 'error' in task_status:
            summary['error'] = task_status['error']
        
        return render_template(
            'batch_results.html',
            results=formatted_results,
            summary=summary,
            batch_id=batch_id,
            now=datetime.now()
        )
        
    except Exception as e:
        app.logger.error(f"Error getting batch results: {str(e)}", exc_info=True)
        flash(f'Error loading batch results: {str(e)}', 'error')
        return redirect(url_for('view_results'))

@app.route('/batch/delete/<batch_id>', methods=['POST'])
@login_required
def delete_batch(batch_id):
    """
    Delete a batch and all its associated results

    Supports both AJAX and regular form submissions
    """
    try:
        # First, try to revoke any running Celery tasks
        from tasks import celery_app
        from celery.exceptions import TimeoutError

        try:
            # Get task ID from the batch
            task_id = f"batch_{batch_id}"
            celery_app.control.revoke(task_id, terminate=True, timeout=1.0)
        except Exception as e:
            app.logger.warning(f"Could not revoke Celery task for batch {batch_id}: {str(e)}")

        # Mark batch as deleted in the database
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                # Verify batch belongs to current user
                cursor.execute(
                    "SELECT id FROM batch_jobs WHERE id = %s AND user_id = %s AND is_deleted = FALSE",
                    (batch_id, current_user.id)
                )
                if not cursor.fetchone():
                    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                        return jsonify({
                            'success': False,
                            'error': "Batch not found or you don't have permission to delete it."
                        }), 403
                    flash("Batch not found or you don't have permission to delete it.", 'error')
                    return redirect(url_for('view_results'))

                # Mark batch as deleted
                cursor.execute(
                    "UPDATE batch_jobs SET is_deleted = TRUE, updated_at = NOW() WHERE id = %s",
                    (batch_id,)
                )
                
                # Mark all batch results as deleted
                cursor.execute(
                    "UPDATE processing_results SET is_deleted = TRUE, updated_at = NOW() WHERE batch_id = %s",
                    (batch_id,)
                )
                
                # Get the count of deleted results
                cursor.execute(
                    "SELECT COUNT(*) FROM processing_results WHERE batch_id = %s",
                    (batch_id,)
                )
                deleted_count = cursor.fetchone()[0]
                
                conn.commit()
        finally:
            conn.close()

        # Prepare response
        response_data = {
            'success': True,
            'deleted_count': deleted_count,
            'batch_id': batch_id,
            'message': f'Successfully deleted batch {batch_id} and all its results',
            'timestamp': datetime.now().isoformat()
        }

        # Handle AJAX requests
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify(response_data)

        # Handle regular form submissions
        flash(response_data['message'], 'success')
        return redirect(url_for('view_results'))

    except Exception as e:
        error_msg = f'Error deleting batch {batch_id}: {str(e)}'
        app.logger.error(error_msg, exc_info=True)

        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
            
        flash(error_msg, 'error')
        return redirect(url_for('view_results'))

def process_batch_from_directory(directory_path, message, chat_id=None):
    """Process all files in a directory as a batch.
    
    Args:
        directory_path (str): Path to the directory containing files to process
        message (str): Original user message that triggered the batch processing
        chat_id (str, optional): ID of the chat where the batch was initiated
        
    Returns:
        dict: Response data for the frontend
    """
    app.logger.info(f"Processing batch from directory: {directory_path}")
    
    # Check if the directory exists
    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        return {
            'response': f"Sorry, the directory '{directory_path}' does not exist or is not accessible.",
            'status': 'error'
        }
    
    # Get all valid files in the directory
    valid_files = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path) and allowed_file(filename):
            valid_files.append(file_path)
    
    app.logger.info(f"Found {len(valid_files)} valid files in directory: {directory_path}")
    
    if not valid_files:
        return {
            'response': f"No valid files found in the directory '{directory_path}'.",
            'status': 'error'
        }
    
    # Determine the appropriate tool using get_tool_with_gemini
    try:
        # Extract the command part (e.g., "extract text")
        command_part = re.sub(r'\s+from\s+.+$', '', message, flags=re.IGNORECASE).strip()
        app.logger.info(f"Command part for tool selection: {command_part}")
        
        # Use get_tool_with_gemini to analyze the command and select the appropriate tool
        tool_name, confidence, tool_args = asyncio.run(get_tool_with_gemini(command_part, has_file=True))
        app.logger.info(f"Selected tool: {tool_name} with confidence {confidence}")
    except Exception as e:
        app.logger.warning(f"Error determining tool with Gemini: {str(e)}")
        # Fall back to default tool if get_tool_with_gemini fails
        tool_name = "process_medical_report"  # Default tool
        tool_args = {"extraction_method": "auto"}
        confidence = 0.5
        app.logger.info(f"Falling back to default tool: {tool_name}")
    
    # If confidence is too low or no tool was selected, use the default
    if not tool_name or confidence < 0.4:
        tool_name = "process_medical_report"
        tool_args = {"extraction_method": "auto"}
        app.logger.info(f"Using default tool due to low confidence: {tool_name}")
    
    # Generate unique batch ID for tracking
    batch_id = str(uuid.uuid4())
    app.logger.info(f"Generated batch ID: {batch_id}")
    
    # First, check if Celery is properly initialized
    celery_initialized = True  # This should be set based on your actual Celery initialization check
    
    # Try to use Celery for asynchronous processing
    try:
        app.logger.info("Attempting to use Celery for batch processing...")
        
        # Import tasks directly
        try:
            import tasks
            from tasks import process_batch_task
            app.logger.info("Tasks module imported successfully")
        except ImportError as ie:
            app.logger.error(f"Failed to import tasks: {str(ie)}")
            raise ConnectionError(f"Failed to import tasks: {str(ie)}")
        
        # Only pass chat_id if it's not None and not an empty string
        task_kwargs = {
            'file_paths': valid_files,
            'tool_name': tool_name,
            'tool_args': tool_args,
            'user_id': current_user.id if current_user.is_authenticated else None
        }
        
        # Only add chat_id to the task kwargs if it's not None and not an empty string
        if chat_id:
            task_kwargs['chat_id'] = chat_id
        
        # Create the batch processing task
        batch_task = process_batch_task.delay(**task_kwargs)
        
        app.logger.info(f"Celery task created with ID: {batch_task.id}")
        task_id = batch_task.id
        
        # Save basic batch info locally too for tracking
        if not hasattr(app, 'batch_tracking'):
            app.batch_tracking = {}
        app.batch_tracking[batch_id] = {
            "id": batch_id,
            "celery_task_id": batch_task.id,
            "total_files": len(valid_files),
            "status": "processing",
            "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Create a user-friendly response
        response_text = f"I'll process all {len(valid_files)} files from '{directory_path}' using the {tool_name} tool. This will be done in the background, and I'll notify you when it's complete."
        
        # Create a proper response object
        response_data = {
            "response": response_text,
            "status": "batch_processing",
            "batch_id": batch_id,
            "total_files": len(valid_files),
            "task_id": batch_task.id,
            "timestamp": datetime.now().isoformat(),
            "is_batch": True,
            "view_url": f"/batch/results/{batch_id}"
        }
        
        return response_data
        
    except Exception as e:
        # Fallback to direct processing if Celery fails
        app.logger.warning(f"Celery batch processing failed: {str(e)}")
        app.logger.info("Falling back to direct processing...")
        
        # Process files in batches to avoid timeout
        # First batch: Process up to 3 files immediately for quick feedback
        first_batch = valid_files[:min(3, len(valid_files))]
        remaining_files = valid_files[min(3, len(valid_files)):]
        
        app.logger.info(f"Processing first batch of {len(first_batch)} files directly")
        
        # Create a simple structure to track batch processing
        batch_info = {
            "id": batch_id,
            "total_files": len(valid_files),
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "results": [],
            "status": "processing",
            "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Process first batch of files directly
        for file_path in first_batch:
            try:
                file_name = os.path.basename(file_path)
                app.logger.info(f"Processing file: {file_name}")
                
                # Process the file directly
                result = asyncio.run(process_file(file_path, tool_name, tool_args, chat_id=chat_id))
                
                # Update batch info based on result
                batch_info["processed"] += 1
                
                if result and isinstance(result, tuple) and len(result) >= 2 and result[1] is not None:
                    # Successful processing
                    batch_info["successful"] += 1
                    batch_info["results"].append({
                        "file": file_name,
                        "success": True,
                        "result_id": result[0] if isinstance(result, tuple) and len(result) > 0 else None
                    })
                else:
                    # Error during processing
                    batch_info["failed"] += 1
                    batch_info["results"].append({
                        "file": file_name,
                        "success": False,
                        "error": str(result) if result else "Unknown error"
                    })
                    
            except Exception as process_error:
                app.logger.error(f"Error processing file {file_path}: {str(process_error)}")
                
                batch_info["processed"] += 1
                batch_info["failed"] += 1
                batch_info["results"].append({
                    "file": os.path.basename(file_path),
                    "success": False,
                    "error": str(process_error)
                })
        
        # Update status based on processing results
        if batch_info["processed"] == batch_info["total_files"]:
            batch_info["status"] = "complete"
        else:
            batch_info["status"] = "partially_complete"
        
        # Store batch info in the app config temporarily
        if not hasattr(app, 'batch_tracking'):
            app.batch_tracking = {}
        app.batch_tracking[batch_id] = batch_info
        
        # Save batch info to a JSON file for persistence
        batch_dir = os.path.join(app.static_folder, 'results', 'batches')
        os.makedirs(batch_dir, exist_ok=True)
        batch_file = os.path.join(batch_dir, f"{batch_id}.json")
        
        with open(batch_file, 'w') as f:
            json.dump(batch_info, f, indent=2)
        
        app.logger.info(f"Processed {batch_info['processed']} files. Successful: {batch_info['successful']}, Failed: {batch_info['failed']}")
        app.logger.info(f"Batch info saved to {batch_file}")
        
        # Create a user-friendly response
        response_text = f"I've started processing {len(valid_files)} files from '{directory_path}' using the {tool_name} tool."
        
        if remaining_files:
            response_text += f" I've processed the first {len(first_batch)} files, and {len(remaining_files)} more will be processed in the background."
        else:
            response_text += f" All files have been processed: {batch_info['successful']} succeeded, {batch_info['failed']} failed."
        
        # Create a proper response object
        response_data = {
            "response": response_text,
            "status": "batch_processing",
            "batch_id": batch_id,
            "total_files": len(valid_files),
            "processed": batch_info["processed"],
            "successful": batch_info["successful"],
            "failed": batch_info["failed"],
            "timestamp": datetime.now().isoformat(),
            "is_batch": True,
            "view_url": f"/batch/results/{batch_id}"
        }
        
        return response_data

@app.route('/api/batch/upload', methods=['POST'])
@login_required
def api_batch_upload():
    """Handle batch file uploads via API"""
    try:
        # Check if files were uploaded
        if 'files[]' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
            
        files = request.files.getlist('files[]')
        if not files or len(files) == 0:
            return jsonify({'error': 'No files selected'}), 400
            
        # Get chat ID from request
        chat_id = request.form.get('chat_id')
        message = request.form.get('message', 'Process these files')
        
        # Filter valid files
        valid_files = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(file_path)
                valid_files.append(file_path)
        
        if not valid_files:
            return jsonify({'error': 'No valid files uploaded'}), 400
            
        app.logger.info(f"Received batch upload with {len(valid_files)} valid files")
        
        # Determine the appropriate tool using get_tool_with_gemini
        try:
            tool_name, confidence, tool_args = asyncio.run(get_tool_with_gemini(message, has_file=True))
            app.logger.info(f"Selected tool: {tool_name} with confidence {confidence}")
        except Exception as e:
            app.logger.warning(f"Error determining tool with Gemini: {str(e)}")
            # Fall back to default tool if get_tool_with_gemini fails
            tool_name = "process_medical_report"  # Default tool
            tool_args = {"extraction_method": "auto"}
            app.logger.info(f"Falling back to default tool: {tool_name}")
        
        # Generate unique batch ID for tracking
        batch_id = str(uuid.uuid4())
        
        # Try to use Celery for asynchronous processing
        try:
            # Import tasks directly
            import tasks
            from tasks import process_batch_task
            
            # Create the batch processing task
            batch_task = process_batch_task.delay(
                file_paths=valid_files,
                tool_name=tool_name,
                tool_args=tool_args,
                user_id=current_user.id,
                chat_id=chat_id
            )
            
            app.logger.info(f"Celery task created with ID: {batch_task.id}")
            
            # Save basic batch info locally too for tracking
            if not hasattr(app, 'batch_tracking'):
                app.batch_tracking = {}
            app.batch_tracking[batch_id] = {
                "id": batch_id,
                "celery_task_id": batch_task.id,
                "total_files": len(valid_files),
                "status": "processing",
                "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Add a message to the chat
            if chat_id:
                response_text = f"I'll process all {len(valid_files)} files using the {tool_name} tool. This will be done in the background, and I'll notify you when it's complete."
                current_user.add_message_to_chat(chat_id, 'assistant', response_text)
            
            # Create a proper response object
            return jsonify({
                "status": "success",
                "message": f"Started batch processing of {len(valid_files)} files",
                "batch_id": batch_id,
                "total_files": len(valid_files),
                "task_id": batch_task.id,
                "is_batch": True,
                "view_url": f"/batch/results/{batch_id}"
            })
            
        except Exception as e:
            app.logger.warning(f"Celery batch processing failed: {str(e)}")
            app.logger.info("Falling back to direct processing...")
            
            # Process files directly (just the first few)
            first_batch = valid_files[:min(3, len(valid_files))]
            
            # Create a simple structure to track batch processing
            batch_info = {
                "id": batch_id,
                "total_files": len(valid_files),
                "processed": 0,
                "successful": 0,
                "failed": 0,
                "results": [],
                "status": "processing",
                "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Process first batch of files directly
            for file_path in first_batch:
                try:
                    # Process the file directly
                    result = asyncio.run(process_file(file_path, tool_name, tool_args, chat_id=chat_id))
                    
                    # Update batch info based on result
                    batch_info["processed"] += 1
                    
                    if result and isinstance(result, tuple) and len(result) >= 2 and result[1] is not None:
                        # Successful processing
                        batch_info["successful"] += 1
                    else:
                        # Error during processing
                        batch_info["failed"] += 1
                        
                except Exception as process_error:
                    app.logger.error(f"Error processing file {file_path}: {str(process_error)}")
                    batch_info["processed"] += 1
                    batch_info["failed"] += 1
            
            # Add a message to the chat
            if chat_id:
                response_text = f"I've started processing {len(valid_files)} files using the {tool_name} tool."
                current_user.add_message_to_chat(chat_id, 'assistant', response_text)
            
            # Return response
            return jsonify({
                "status": "success",
                "message": f"Started batch processing of {len(valid_files)} files",
                "batch_id": batch_id,
                "total_files": len(valid_files),
                "processed": batch_info["processed"],
                "successful": batch_info["successful"],
                "failed": batch_info["failed"],
                "is_batch": True,
                "view_url": f"/batch/results/{batch_id}"
            })
            
    except Exception as e:
        app.logger.error(f"Error processing batch upload: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/notifications', methods=['GET'])
@login_required
def get_notifications():
    """Get notifications for the current user"""
    try:
        # Get notifications from user object
        if not hasattr(current_user, 'notifications'):
            current_user.notifications = []
            
        # Return notifications
        return jsonify({
            "status": "success",
            "notifications": current_user.notifications
        })
        
    except Exception as e:
        app.logger.error(f"Error getting notifications: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/notifications/mark-read', methods=['POST'])
@login_required
def mark_notifications_read():
    """Mark notifications as read"""
    try:
        data = request.json
        notification_ids = data.get('notification_ids', [])
        
        # Mark notifications as read
        if hasattr(current_user, 'notifications'):
            for notification in current_user.notifications:
                if notification['id'] in notification_ids:
                    notification['read'] = True
            
            # Save user object
            current_user.save()
            
        # Return updated notifications
        return jsonify({
            "status": "success",
            "message": f"Marked {len(notification_ids)} notifications as read",
            "notifications": current_user.notifications if hasattr(current_user, 'notifications') else []
        })
        
    except Exception as e:
        app.logger.error(f"Error marking notifications as read: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch/results/<batch_id>')
@login_required
def view_batch_results(batch_id):
    """View results for a specific batch processing job"""
    try:
        # Get batch information from database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get batch metadata
        cursor.execute(
            "SELECT id, user_id, status, total_files, processed, successful, failed, "
            "created_at, updated_at FROM batch_jobs WHERE id = %s AND user_id = %s",
            (batch_id, current_user.id)
        )
        batch_row = cursor.fetchone()
        
        if not batch_row:
            flash("Batch not found or you don't have permission to view it.")
            return redirect(url_for('view_results'))
        
        # Convert to dictionary
        batch = {
            'id': batch_row[0],
            'user_id': batch_row[1],
            'status': batch_row[2],
            'total_files': batch_row[3],
            'processed': batch_row[4],
            'successful': batch_row[5],
            'failed': batch_row[6],
            'started_at': batch_row[7],
            'completed_at': batch_row[8]
        }
        
        # Get batch results
        cursor.execute(
            "SELECT id, filename, original_filename, processing_type, tool_used, status, "
            "result_data, error_message, created_at FROM processing_results "
            "WHERE batch_id = %s AND user_id = %s ORDER BY created_at DESC",
            (batch_id, current_user.id)
        )
        result_rows = cursor.fetchall()
        
        # Convert to list of dictionaries
        batch['results'] = []
        for row in result_rows:
            result = {
                'result_id': row[0],
                'file': row[2] or row[1],  # Use original filename if available
                'processing_type': row[3],
                'tool_used': row[4],
                'success': row[5] == 'completed',
                'error': row[7],
                'created_at': row[8]
            }
            batch['results'].append(result)
            
        cursor.close()
        conn.close()
        
        return render_template('batch/results.html', batch=batch)
        
    except Exception as e:
        logger.error(f"Error viewing batch results: {e}", exc_info=True)
        flash("An error occurred while retrieving batch results.")
        return redirect(url_for('view_results'))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)  # Note: Using port 5001 to avoid conflict with web_app.py
