"""
Celery tasks for batch processing medical reports.
"""

import os
import json
import asyncio
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from psycopg2.extras import Json

import websockets
from celery import shared_task, group, chain
from celery.utils.log import get_task_logger
from celery.result import allow_join_result

# Import celery configuration
from celery_config import celery_app
from db_results import ResultManager
from db_users import User
from db import get_db_connection

# Ensure the current directory is in the path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import BatchManager for batch operations
try:
    from db_batch_helper import BatchManager
except ImportError:
    logger.error("Could not import BatchManager. Make sure db_batch_helper.py is in the correct location.")
    # Define a fallback BatchManager class to prevent errors
    class BatchManager:
        @staticmethod
        def create_batch_job(*args, **kwargs):
            logger.error("Using fallback BatchManager.create_batch_job")
            return str(uuid.uuid4())
            
        @staticmethod
        def update_batch_status(*args, **kwargs):
            logger.error("Using fallback BatchManager.update_batch_status")
            return True
            
        @staticmethod
        def get_batch_status(*args, **kwargs):
            logger.error("Using fallback BatchManager.get_batch_status")
            return {"processed": 0, "failed": 0, "total_files": 0, "status": "unknown"}

# Configure logging
logger = get_task_logger(__name__)

# MCP WebSocket URL
MCP_WEBSOCKET_URL = "ws://localhost:8765"

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chat_uploads')
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chat_results')

# Create necessary directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

@celery_app.task(name='tasks.process_batch_task', bind=True)
def process_batch_task(self, file_paths, tool_name, tool_args=None, user_id=None, chat_id=None):
    """
    Process a batch of files asynchronously using Celery.
    
    Args:
        file_paths: List of file paths to process
        tool_name: Name of the MCP tool to use for processing
        tool_args: Additional arguments for the tool
        user_id: User ID for database storage
        chat_id: Chat ID for database storage
    
    Returns:
        Dictionary with batch processing results
    """
    logger.info(f"Starting batch processing of {len(file_paths)} files with tool: {tool_name}")
    
    # Initialize counters
    total_files = len(file_paths)
    processed = 0
    failed = 0
    results = []
    
    # Import BatchManager and create a batch job in the database
    try:
        from db_batch_helper import BatchManager
        
        # Get folder path if available
        folder_path = os.path.dirname(file_paths[0]) if file_paths else ""
        
        # Create batch job in database
        batch_id = BatchManager.create_batch_job(
            user_id=user_id,
            tool_name=tool_name,
            total_files=total_files,
            folder_path=folder_path
        )
        
        if not batch_id:
            error_msg = "Failed to create batch job in database"
            logger.error(error_msg)
            self.update_state(state="FAILURE", meta={"error": error_msg})
            return {"success": False, "error": error_msg, "processed": 0, "failed": 0}
            
        logger.info(f"Created batch job with ID: {batch_id}")
    except Exception as e:
        # Fall back to UUID generation if BatchManager fails
        batch_id = str(uuid.uuid4())
        logger.warning(f"Using fallback batch_id generation due to error: {e}")
        logger.info(f"Generated batch ID: {batch_id}")
    
    # Process each file
    for i, file_path in enumerate(file_paths):
        try:
            logger.info(f"Processing file {i+1}/{total_files}: {os.path.basename(file_path)}")
            
            # Create event loop for async processing
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Process the file
            result = loop.run_until_complete(
                process_file_async(file_path, tool_name, tool_args, user_id, chat_id, batch_id)
            )
            
            # Close the loop
            loop.close()
            
            # Track results
            results.append(result)
            if result.get('success', False):
                processed += 1
            else:
                failed += 1
                
            # Update batch status in database
            try:
                from db_batch_helper import BatchManager
                BatchManager.update_batch_status(
                    batch_id=batch_id,
                    processed=processed,
                    failed=failed,
                    status="processing"
                )
            except Exception as e:
                logger.error(f"Error updating batch status: {e}")
                
            # Report progress to Celery
            self.update_state(
                state='PROGRESS',
                meta={
                    'batch_id': batch_id,
                    'current': i + 1,
                    'total': total_files,
                    'processed': processed,
                    'failed': failed,
                    'status': 'processing'
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            failed += 1
            results.append({
                'status': 'error',
                'message': str(e),
                'file_path': file_path,
                'batch_id': batch_id,
                'success': False
            })
    
    # Update final batch status
    status = "completed" if failed == 0 else "completed_with_errors" if processed > 0 else "failed"
    try:
        from db_batch_helper import BatchManager
        BatchManager.update_batch_status(
            batch_id=batch_id,
            processed=processed,
            failed=failed,
            status=status
        )
        logger.info(f"Batch processing complete. Status: {status}, Processed: {processed}, Failed: {failed}")
    except Exception as e:
        logger.error(f"Error updating final batch status: {e}")
    
    return {
        'batch_id': batch_id,
        'total': total_files,
        'processed': processed,
        'failed': failed,
        'status': status,
        'results': results
    }

# Helper functions
def get_timestamp():
    """Get current timestamp in readable format."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def save_result_to_json(result, filename_base, tool_name, user_id, chat_id=None):
    """Save results to the database and return a unique filename."""
    timestamp = get_timestamp()
    json_filename = f"{filename_base}_{tool_name}_{timestamp}.json"
    
    # Determine processing type based on tool name
    if 'summarize' in tool_name.lower():
        processing_type = 'summary'
    else:
        processing_type = 'process'
    
    # Save to database
    ResultManager.save_result(
        user_id=user_id,
        filename=json_filename,
        original_filename=filename_base,
        processing_type=processing_type,
        tool_used=tool_name,
        result_data=result,
        chat_id=chat_id,
        batch_id=result.get('batch_id')
    )
    
    # For backward compatibility, also save to file system
    json_path = os.path.join(RESULTS_FOLDER, json_filename)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    
    return json_filename

class MCPAsyncClient:
    """Async client for MCP WebSocket communication."""
    
    def __init__(self):
        self.ws = None
        self.tools = []

    async def connect(self):
        """Connect to the MCP WebSocket server."""
        try:
            self.ws = await websockets.connect(MCP_WEBSOCKET_URL)
            greeting = await self.ws.recv()
            info = json.loads(greeting)
            self.tools = info.get("tools", [])
            logger.info(f"Connected to MCP server; received {len(self.tools)} tool schemas")
            return True
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            return False

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
        logger.info(f"Calling tool: {tool_name}")
        
        try:
            await self.ws.send(json.dumps(payload))
            
            # collect until we see the matching tool_result with timeout
            start_time = time.time()
            timeout = 300  # 5 minutes timeout for batch processing
            
            while True:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Timeout waiting for tool result after {timeout} seconds")
                    
                raw = await self.ws.recv()
                msg = json.loads(raw)
                
                # Handle tool result
                if msg.get("event") == "tool_result" and msg.get("request_id") == request_id:
                    logger.info(f"Tool result received for {tool_name}")
                    return msg["result"]
                    
                # Handle tool error
                elif msg.get("event") == "tool_error" and msg.get("request_id") == request_id:
                    error_msg = msg.get('error', 'Unknown error')
                    logger.error(f"Tool error received: {error_msg}")
                    raise RuntimeError(f"Tool execution failed: {error_msg}")
                    
                # Handle progress updates
                elif msg.get("event") == "progress":
                    logger.debug(f"Progress: {msg.get('current')}/{msg.get('total')}")
                    # Continue waiting for the final result
        except Exception as e:
            logger.error(f"Error during tool call: {str(e)}")
            raise
 
    async def close(self):
        """Tear down the WebSocket connection."""
        if self.ws:
            try:
                await self.ws.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {str(e)}")
            finally:
                self.ws = None

async def process_file_async(file_path, tool_name=None, tool_args=None, user_id=None, chat_id=None, batch_id=None):
    """Process a file using the specified tool through MCP client (async version).
    If tool_name is None, it will be determined using Gemini."""
    """Process a file using the specified tool through MCP client (async version)."""
    result = {
        'status': 'error',
        'message': 'Unknown error occurred',
        'file_path': file_path,
        'batch_id': batch_id,
        'success': False
    }
    
    client = MCPAsyncClient()
    try:
        # Connect to the MCP server
        connected = await client.connect()
        if not connected:
            result.update({
                'status': 'error',
                'message': 'Could not connect to MCP server. Is it running?',
                'success': False
            })
            return result
        
        # Prepare arguments
        if tool_args is None:
            tool_args = {}
        
        # Always set the file_path in the arguments
        tool_args['file_path'] = file_path
        
        # If tool_name is not provided, use Gemini to determine it
        if tool_name is None:
            try:
                # Get file content for Gemini to analyze
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                    try:
                        # Try to decode as text
                        file_content_text = file_content.decode('utf-8', errors='replace')
                    except:
                        # If decoding fails, just use the filename
                        file_content_text = f"File: {os.path.basename(file_path)}"
                
                # Create a message for Gemini to analyze
                message = f"Process this medical report: {os.path.basename(file_path)}"
                
                # Import the function from chat_app to avoid circular imports
                from chat_app import get_tool_with_gemini
                
                # Get the tool name and arguments from Gemini
                tool_name, confidence, gemini_tool_args = await get_tool_with_gemini(message, has_file=True)
                logger.info(f"Gemini selected tool: {tool_name} with confidence {confidence}")
                
                # Merge the provided tool_args with the ones from Gemini
                if tool_args is None:
                    tool_args = {}
                tool_args.update(gemini_tool_args)
                
            except Exception as e:
                logger.error(f"Error determining tool with Gemini: {str(e)}")
                # Default to process_medical_report if Gemini fails
                tool_name = "process_medical_report"
                if tool_args is None:
                    tool_args = {}
                tool_args.update({"extraction_method": "auto"})
        
        # Always set the file_path in the arguments
        tool_args['file_path'] = file_path
        
        # Call the tool
        logger.info(f"Calling MCP tool: {tool_name} with args: {tool_args}")
        
        # Call the tool and get the response
        response = await client.call_tool(tool_name, **tool_args)
        
        # Process the response
        if not response:
            result.update({
                'status': 'error',
                'message': 'No response from the tool',
                'success': False
            })
            return result
            
        # If response is a string, try to parse it as JSON
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                response = {'content': response}
        
        # If response is a dictionary, update the result
        if isinstance(response, dict):
            result.update(response)
            result['success'] = 'error' not in response or not response['error']
        else:
            result.update({
                'content': str(response),
                'success': True
            })
        
        # Add batch ID to result if provided
        if batch_id and isinstance(result, dict):
            result['batch_id'] = batch_id
        
        # Save the result to JSON and database if successful
        if result.get('success'):
            try:
                filename_base = os.path.splitext(os.path.basename(file_path))[0]
                json_filename = save_result_to_json(result, filename_base, tool_name, user_id, chat_id)
                result['filename'] = json_filename
            except Exception as save_error:
                logger.error(f"Error saving result: {str(save_error)}")
                result.update({
                    'status': 'warning',
                    'message': f'Processed but failed to save result: {str(save_error)}',
                    'success': False
                })
        
        return result
        
    except Exception as e:
        error_msg = f"Error processing file {file_path}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        result.update({
            'status': 'error',
            'message': error_msg,
            'success': False
        })
        return result
        
    finally:
        # Always close the WebSocket connection
        try:
            await client.close()
            logger.info("WebSocket connection closed")
        except Exception as close_error:
            logger.error(f"Error closing WebSocket: {str(close_error)}")

def process_file_with_mcp(file_path, tool_name=None, tool_args=None, timeout=300):
    """Process a file using the MCP WebSocket server with enhanced error handling and Windows compatibility."""
    logger.info(f"Processing file {file_path} with tool {tool_name}")
    
    # Get the file content
    try:
        with open(file_path, 'rb') as f:
            file_content = f.read().decode('utf-8', errors='replace')
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        return {
            'success': False,
            'error': f"Error reading file: {str(e)}",
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    # If tool_name is None, we'll need to determine it using Gemini
    # But we need to do this in the async function below, so just prepare the request
    # with the file content for now
    request = {
        'content': file_content
    }
    
    # Add tool_name if it's provided
    if tool_name:
        request['tool_name'] = tool_name
    
    # Add tool arguments if provided
    if tool_args:
        request.update(tool_args)
    
    # Convert to JSON
    request_json = json.dumps(request)
    
    # Windows-friendly approach to handle asyncio
    import platform
    is_windows = platform.system().lower() == 'windows'
    
    # Create a new event loop for this task
    try:
        # Close any existing event loop first to avoid resource leaks
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running() or loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            # No event loop exists yet
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Define async function to communicate with MCP
        async def communicate():
            # Add retry logic
            max_retries = 3
            retry_delay = 2  # seconds
            
            for attempt in range(max_retries):
                try:
                    # Use a timeout for the connection attempt
                    connect_timeout = 30  # seconds
                    conn_kwargs = {
                        'max_size': None,
                        'close_timeout': 10,
                        'max_queue': 32
                    }
                    
                    # Windows-specific adjustments
                    if is_windows:
                        conn_kwargs['ping_interval'] = None  # Disable ping on Windows
                        conn_kwargs['ping_timeout'] = None
                    
                    async with asyncio.timeout(connect_timeout):
                        async with websockets.connect(MCP_WEBSOCKET_URL, **conn_kwargs) as websocket:
                            # If tool_name is None, determine it using Gemini
                            if 'tool_name' not in request:
                                try:
                                    # Import the function from chat_app to avoid circular imports
                                    # We need to import it here to avoid circular imports
                                    import importlib.util
                                    spec = importlib.util.spec_from_file_location(
                                        "chat_app", 
                                        os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_app.py")
                                    )
                                    chat_app = importlib.util.module_from_spec(spec)
                                    spec.loader.exec_module(chat_app)
                                    
                                    # Create a message for Gemini to analyze
                                    filename = os.path.basename(file_path)
                                    message = f"Process this medical report: {filename}"
                                    
                                    # Get the tool name and arguments from Gemini
                                    tool_name, confidence, gemini_tool_args = await chat_app.get_tool_with_gemini(message, has_file=True)
                                    logger.info(f"Gemini selected tool: {tool_name} with confidence {confidence}")
                                    
                                    # Add the tool name to the request
                                    request['tool_name'] = tool_name
                                    
                                    # Merge the provided tool_args with the ones from Gemini
                                    if tool_args is None:
                                        tool_args = {}
                                    tool_args.update(gemini_tool_args)
                                    
                                    # Update the request with the merged tool_args
                                    request.update(tool_args)
                                    
                                    # Re-encode the request
                                    request_json = json.dumps(request)
                                    
                                except Exception as e:
                                    logger.error(f"Error determining tool with Gemini: {str(e)}")
                                    # Default to process_medical_report if Gemini fails
                                    request['tool_name'] = "process_medical_report"
                                    if 'extraction_method' not in request:
                                        request['extraction_method'] = "auto"
                                    # Re-encode the request
                                    request_json = json.dumps(request)
                            
                            # Send the request
                            await websocket.send(request_json)
                            
                            # Wait for the response with timeout
                            response = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                            
                            # Parse the response
                            return json.loads(response)
                            
                except asyncio.TimeoutError:
                    logger.warning(f"Attempt {attempt+1}/{max_retries}: Timeout waiting for response from MCP server")
                    if attempt == max_retries - 1:  # Last attempt
                        return {
                            'success': False,
                            'error': f"Timeout waiting for response from MCP server after {timeout} seconds",
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                except websockets.exceptions.ConnectionClosed as e:
                    logger.warning(f"Attempt {attempt+1}/{max_retries}: WebSocket connection closed: {str(e)}")
                    if attempt == max_retries - 1:  # Last attempt
                        return {
                            'success': False,
                            'error': f"WebSocket connection closed: {str(e)}",
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                except Exception as e:
                    logger.warning(f"Attempt {attempt+1}/{max_retries}: Error communicating with MCP server: {str(e)}")
                    if attempt == max_retries - 1:  # Last attempt
                        return {
                            'success': False,
                            'error': f"Error communicating with MCP server: {str(e)}",
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                
                # Wait before retrying
                if attempt < max_retries - 1:  # Don't sleep after the last attempt
                    await asyncio.sleep(retry_delay * (attempt + 1))  # Exponential backoff
        
        # Run the async function
        result = loop.run_until_complete(communicate())
        return result
    except Exception as e:
        logger.error(f"Error in event loop: {str(e)}", exc_info=True)
        return {
            'success': False,
            'error': f"Error in event loop: {str(e)}",
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    finally:
        # Clean up the event loop
        try:
            # Cancel all running tasks
            for task in asyncio.all_tasks(loop) if hasattr(asyncio, 'all_tasks') else asyncio.Task.all_tasks(loop):
                task.cancel()
            
            # Run the event loop until all tasks are cancelled
            if not loop.is_closed():
                loop.run_until_complete(asyncio.sleep(0.1))
                loop.close()
        except Exception as e:
            logger.error(f"Error cleaning up event loop: {str(e)}", exc_info=True)

@celery_app.task(bind=True, name='tasks.process_file_task')
def process_file_task(self, file_path, tool_name=None, tool_args=None, user_id=None, chat_id=None, batch_id=None):
    """Celery task to process a single file."""
    logger.info(f"Processing file: {file_path} with tool: {tool_name}")
    
    # Initialize result with default values
    result = {
        'status': 'processing',
        'message': 'Processing started',
        'file_path': file_path,
        'tool_name': tool_name,
        'batch_id': batch_id,
        'success': False
    }
    
    # Update task state
    self.update_state(
        state='PROGRESS',
        meta={
            'status': 'processing',
            'progress': 0,
            **result
        }
    )
    
    try:
        # Process the file using our enhanced MCP function
        # If tool_name is None, it will be determined using Gemini in process_file_with_mcp
        mcp_result = process_file_with_mcp(file_path, tool_name, tool_args)
        
        # Ensure we have a dictionary result
        if not isinstance(mcp_result, dict):
            mcp_result = {
                'status': 'error',
                'message': f'Unexpected result type: {type(mcp_result).__name__}',
                'success': False
            }
        
        # Update the result with the MCP operation results
        result.update(mcp_result)
        
        # Save the result to the database if user_id is provided
        if user_id and batch_id:
            try:
                filename = os.path.basename(file_path)
                original_filename = filename
                
                # Extract original filename from the processed filename if possible
                parts = filename.split('_', 2)
                if len(parts) >= 3:
                    original_filename = parts[2]
                
                from db_results import ResultManager
                ResultManager.save_result(
                    user_id=user_id,
                    filename=filename,
                    original_filename=original_filename,
                    processing_type=tool_name,
                    tool_used=tool_name,
                    result_data=result,
                    status='completed' if result.get('success') else 'failed',
                    error_message=result.get('error'),
                    chat_id=chat_id,
                    batch_id=batch_id
                )
            except Exception as db_error:
                logger.error(f"Error saving result to database: {str(db_error)}", exc_info=True)
        
        # Set the final status
        if result.get('success'):
            result['status'] = 'completed'
            result['progress'] = 100
            state = 'SUCCESS'
        else:
            result['status'] = 'failed'
            result['progress'] = 0
            state = 'FAILURE'
        
        # Update task state with final result
        self.update_state(
            state=state,
            meta={
                'status': result['status'],
                'progress': result.get('progress', 0),
                **result
            }
        )
        
        return result
        
    except Exception as e:
        error_msg = f"Error processing file {file_path}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Update result with error information
        result.update({
            'status': 'error',
            'message': error_msg,
            'success': False,
            'error': str(e)
        })
        
        # Update task state with error
        self.update_state(
            state='FAILURE',
            meta={
                'status': 'error',
                'progress': 0,
                **result
            }
        )
        
        return result

@celery_app.task(bind=True, name='tasks.process_batch_task')
def process_batch_task(self, file_paths, tool_name=None, tool_args=None, user_id=None, chat_id=None):
    """Celery task to process a batch of files with progress tracking."""
    logger.info(f"Processing batch of {len(file_paths)} files with tool: {tool_name}")
    
    # Generate a unique batch ID
    batch_id = str(uuid.uuid4())
    
    # Initialize counters
    total_files = len(file_paths)
    processed = 0
    failed = 0
    
    # Create batch job record in database
    try:
        batch_id = BatchManager.create_batch_job(
            user_id=user_id,
            tool_name=tool_name,
            total_files=len(file_paths),
            folder_path=os.path.dirname(file_paths[0]) if file_paths else None,
            chat_id=chat_id
        )
        
        if not batch_id:
            logger.error("Failed to create batch job using BatchManager")
            return {
                'status': 'error',
                'message': 'Failed to create batch job',
                'batch_id': None
            }
            
        logger.info(f"Created new batch job record with ID: {batch_id}")
    except Exception as e:
        logger.error(f"Error creating batch job record: {e}", exc_info=True)
        raise
    
    try:
        # More robust approach for Windows - process files sequentially to avoid multiprocessing issues
        import platform
        is_windows = platform.system().lower() == 'windows'
        
        if is_windows:
            logger.info("Running on Windows - using sequential processing")
            # Process files sequentially on Windows to avoid multiprocessing issues
            for i, file_path in enumerate(file_paths):
                try:
                    logger.info(f"Processing file {i+1}/{len(file_paths)}: {os.path.basename(file_path)}")
                    # Process the file directly
                    result = process_file_task(file_path=file_path,
                                               tool_name=tool_name,
                                               tool_args=tool_args,
                                               user_id=user_id,
                                               chat_id=chat_id,
                                               batch_id=batch_id)
                    
                    # Update batch status based on result
                    if result.get('success', False):
                        processed += 1
                    else:
                        failed += 1
                    
                    # Update batch job in database
                    with get_db_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            """UPDATE batch_jobs 
                               SET processed = %s, failed = %s, 
                                   status = %s, 
                                   updated_at = CURRENT_TIMESTAMP
                               WHERE metadata->>'batch_id' = %s""",
                            (processed, failed, 
                             "processing" if processed + failed < total_files else 
                             ("completed" if failed == 0 else "partial_success"), 
                             batch_id)
                        )
                        conn.commit()
                    
                    # Update progress
                    self.update_state(state='PROGRESS',
                                      meta={'current': i + 1, 'total': len(file_paths), 'file': os.path.basename(file_path)})
                    
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}", exc_info=True)
                    failed += 1
                    # Continue processing other files even if one fails
                    continue
        else:
            # On non-Windows systems, use the more efficient chunked approach
            logger.info("Running on non-Windows system - using chunked processing")
            chunk_size = 5  # Process 5 files at a time
            
            for i in range(0, len(file_paths), chunk_size):
                chunk = file_paths[i:i + chunk_size]
                
                # Create tasks for the current chunk
                tasks = []
                for file_path in chunk:
                    # Create a chain: process_file_task -> update_batch_after_process
                    task_chain = chain(
                        process_file_task.s(
                            file_path=file_path,
                            tool_name=tool_name,
                            tool_args=tool_args,
                            user_id=user_id,
                            chat_id=chat_id,
                            batch_id=batch_id
                        ),
                        update_batch_after_process.s(
                            batch_id=batch_id,
                            user_id=user_id,
                            file_path=file_path
                        )
                    )
                    tasks.append(task_chain)
                
                # Execute the chunk and wait for completion with better error handling
                try:
                    with allow_join_result():
                        chunk_results = group(tasks).apply_async().get(disable_sync_subtasks=False, timeout=600)
                    
                    # Check if any task in the chunk failed
                    if any(isinstance(result, Exception) for result in chunk_results):
                        logger.warning(f"Some tasks in chunk {i//chunk_size + 1} failed")
                except Exception as e:
                    logger.error(f"Error processing chunk {i//chunk_size + 1}: {str(e)}", exc_info=True)
                
                # Update progress
                progress = min(i + chunk_size, len(file_paths))
                self.update_state(state='PROGRESS',
                                  meta={'current': progress, 'total': len(file_paths)})
                
                # Get current processed/failed counts from database
                with get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """SELECT processed, failed FROM batch_jobs WHERE metadata->>'batch_id' = %s""",
                        (batch_id,)
                    )
                    row = cursor.fetchone()
                    if row:
                        processed, failed = row
        
        # Final update with completion status
        status = "completed" if failed == 0 else "partial_success"
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE batch_jobs 
                   SET status = %s, 
                       metadata = jsonb_set(metadata, '{end_time}', %s),
                       updated_at = CURRENT_TIMESTAMP
                   WHERE metadata->>'batch_id' = %s""",
                (status, Json(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), batch_id)
            )
            conn.commit()
        
        return {
            'batch_id': batch_id,
            'status': status,
            'processed': processed,
            'failed': failed,
            'total': total_files,
            'message': f'Batch processing completed. {processed} succeeded, {failed} failed.'
        }
        
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}", exc_info=True)
        
        # Update batch status to failed
        try:
            # Use BatchManager to update batch status to failed
            BatchManager.update_batch_status(
                batch_id=batch_id,
                status='failed',
                error_message=str(e)
            )
            logger.info(f"Updated batch {batch_id} status to failed due to error")
        except Exception as db_error:
            logger.error(f"Error updating batch status: {db_error}", exc_info=True)
        
        # Re-raise the exception to mark the task as failed
        raise

@celery_app.task(bind=True, name='tasks.update_batch_after_process')
def update_batch_after_process(self, file_result, batch_id, user_id, file_path):
    """Update batch status after processing a single file."""
    try:
        filename = os.path.basename(file_path)
        
        # Import BatchManager
        from db_batch_helper import BatchManager
        
        # Get current batch status from database
        batch_status = BatchManager.get_batch_status(batch_id)
        
        if not batch_status:
            logger.error(f"Batch job not found for batch_id={batch_id}")
            return file_result
            
        # Extract current counts
        processed = batch_status.get('processed', 0)
        failed = batch_status.get('failed', 0)
        total_files = batch_status.get('total_files', 0)
        
        # Update processed/failed counts based on result
        if file_result.get('success', False):
            processed += 1
        else:
            failed += 1
            
        # Determine status
        status = "processing"
        if processed + failed >= total_files:
            status = "completed" if failed == 0 else "partial_success"
            
        # Update batch job in database
        BatchManager.update_batch_status(
            batch_id=batch_id,
            processed=processed,
            failed=failed,
            status=status
        )
        
        logger.info(f"Updated batch {batch_id}: processed={processed}, failed={failed}, status={status}")
        
        return file_result
        
    except Exception as e:
        logger.error(f"Error updating batch after processing file {file_path}: {str(e)}", exc_info=True)
        return file_result

def update_batch_status(batch_id, status_updates, user_id):
    """Update batch status in the database."""
    try:
        # Import BatchManager
        from db_batch_helper import BatchManager
        
        # Verify batch job exists
        batch_status = BatchManager.get_batch_status(batch_id)
        
        if not batch_status:
            logger.error(f"Batch job not found for batch_id={batch_id}, user_id={user_id}")
            return False
        
        # Extract update parameters
        processed = status_updates.get('processed')
        failed = status_updates.get('failed')
        status = status_updates.get('status')
        total_files = status_updates.get('total_files')
        
        # Update batch job in database
        success = BatchManager.update_batch_status(
            batch_id=batch_id,
            processed=processed,
            failed=failed,
            status=status,
            total_files=total_files
        )
        
        if success:
            logger.debug(f"Updated batch job {batch_id} successfully")
            
            # Send a notification if the batch is completed or failed
            if status in ['completed', 'failed', 'partial_success']:
                try:
                    # Create a notification message
                    if status == 'completed':
                        message = f"Batch processing completed successfully. {batch_status.get('success_count', 0)} files processed."
                        notification_type = 'success'
                    elif status == 'partial_success':
                        success_count = batch_status.get('success_count', 0)
                        failed_count = batch_status.get('failed_count', 0)
                        message = f"Batch processing completed with some failures. {success_count} succeeded, {failed_count} failed."
                        notification_type = 'warning'
                    else:  # failed
                        message = "Batch processing failed. Please check the logs for details."
                        notification_type = 'error'
                    
                    # Send the notification asynchronously
                    send_notification.delay(
                        user_id=user_id,
                        message=message,
                        notification_type=notification_type,
                        batch_id=batch_id
                    )
                except Exception as e:
                    logger.error(f"Error sending batch completion notification: {str(e)}")
            
            return True
        else:
            logger.warning(f"Failed to update batch job {batch_id}")
            return False
            
    except Exception as e:
        logger.error(f"Error updating batch status: {str(e)}", exc_info=True)
        return False

# Simple ping task to test Celery connectivity
@celery_app.task(bind=True, name='tasks.ping')
def ping(self):
    """Simple ping task to test Celery connectivity"""
    return {'status': 'success', 'message': 'pong'}

@celery_app.task(bind=True, name='tasks.process_directory_task')
def process_directory_task(self, directory_path, tool_name=None, tool_args=None, user_id=None, chat_id=None):
    """Celery task to process all valid files in a directory.
    
    Args:
        directory_path (str): Path to the directory containing files to process
        tool_name (str, optional): Name of the tool to use. If None, will be determined using Gemini.
        tool_args (dict, optional): Arguments for the tool
        user_id (int, optional): ID of the user who initiated the task
        chat_id (str, optional): ID of the chat where the task was initiated
        
    Returns:
        dict: Information about the batch processing job
    """
    logger.info(f"Processing directory: {directory_path}")
    
    # Check if the directory exists
    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        return {
            'status': 'error',
            'message': f"Directory '{directory_path}' does not exist or is not accessible"
        }
    
    # Define allowed file extensions
    allowed_extensions = {'pdf', 'png', 'jpg', 'jpeg'}
    
    # Function to check if a file has an allowed extension
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions
    
    # Get all valid files in the directory
    valid_files = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path) and allowed_file(filename):
            valid_files.append(file_path)
    
    logger.info(f"Found {len(valid_files)} valid files in directory: {directory_path}")
    
    if not valid_files:
        return {
            'status': 'error',
            'message': f"No valid files found in the directory '{directory_path}'"
        }
    
    # If tool_name is None and we have at least one file, use the first file to determine the tool
    if tool_name is None and valid_files:
        try:
            # Create a message for Gemini to analyze based on the directory name
            dir_name = os.path.basename(directory_path)
            message = f"Process medical reports from directory: {dir_name}"
            
            # Import the function from chat_app to avoid circular imports
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "chat_app", 
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_app.py")
            )
            chat_app = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(chat_app)
            
            # Create an event loop for the async function
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Get the tool name and arguments from Gemini
            tool_name, confidence, gemini_tool_args = loop.run_until_complete(
                chat_app.get_tool_with_gemini(message, has_file=True)
            )
            logger.info(f"Gemini selected tool: {tool_name} with confidence {confidence}")
            
            # Merge the provided tool_args with the ones from Gemini
            if tool_args is None:
                tool_args = {}
            tool_args.update(gemini_tool_args)
            
            # Clean up the event loop
            loop.close()
            
        except Exception as e:
            logger.error(f"Error determining tool with Gemini: {str(e)}")
            # Default to process_medical_report if Gemini fails
            tool_name = "process_medical_report"
            if tool_args is None:
                tool_args = {}
            tool_args.update({"extraction_method": "auto"})
    
    # Generate a unique batch ID
    batch_id = str(uuid.uuid4())
    
    # Create a batch processing task
    batch_task = process_batch_task.delay(
        file_paths=valid_files,
        tool_name=tool_name,
        tool_args=tool_args,
        user_id=user_id,
        chat_id=chat_id
    )
    
    logger.info(f"Created batch processing task with ID: {batch_task.id} for batch: {batch_id}")
    
    return {
        'status': 'processing',
        'message': f"Started batch processing of {len(valid_files)} files",
        'batch_id': batch_id,
        'task_id': batch_task.id,
        'file_count': len(valid_files)
    }

@celery_app.task(name='tasks.cleanup_expired_results')
def cleanup_expired_results(days=30):
    """Clean up expired results from the database.
    
    Args:
        days (int): Number of days to keep results for
        
    Returns:
        dict: Information about the cleanup operation
    """
    from datetime import datetime, timedelta
    import psycopg2
    from db import get_db_connection
    
    logger.info(f"Cleaning up results older than {days} days")
    
    try:
        # Calculate the cutoff date
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Connect to the database
        conn = get_db_connection()
        with conn.cursor() as cur:
            # Mark old results as deleted
            cur.execute(
                "UPDATE results SET is_deleted = TRUE WHERE created_at < %s AND is_deleted = FALSE",
                (cutoff_date,)
            )
            deleted_count = cur.rowcount
            conn.commit()
        
        logger.info(f"Marked {deleted_count} results as deleted")
        
        return {
            'status': 'success',
            'message': f"Cleaned up {deleted_count} results older than {days} days",
            'deleted_count': deleted_count
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up expired results: {str(e)}")
        return {
            'status': 'error',
            'message': f"Error cleaning up expired results: {str(e)}"
        }

@celery_app.task(bind=True, name='tasks.send_notification')
def send_notification(self, user_id, message, notification_type='info', batch_id=None, result_id=None):
    """Send a notification to a user.
    
    Args:
        user_id (int): ID of the user to notify
        message (str): Notification message
        notification_type (str): Type of notification (info, success, warning, error)
        batch_id (str, optional): ID of the batch related to this notification
        result_id (int, optional): ID of the result related to this notification
        
    Returns:
        dict: Information about the notification
    """
    logger.info(f"Sending notification to user {user_id}: {message}")
    
    try:
        from db_users import User
        
        # Get the user
        user = User.get(user_id)
        if not user:
            logger.error(f"User {user_id} not found")
            return {
                'status': 'error',
                'message': f"User {user_id} not found"
            }
        
        # Create the notification
        notification = {
            'id': str(uuid.uuid4()),
            'user_id': user_id,
            'message': message,
            'type': notification_type,
            'batch_id': batch_id,
            'result_id': result_id,
            'created_at': datetime.now().isoformat(),
            'read': False
        }
        
        # Add the notification to the user's notifications
        if not hasattr(user, 'notifications'):
            user.notifications = []
        user.notifications.append(notification)
        user.save()
        
        logger.info(f"Notification sent to user {user_id}")
        
        return {
            'status': 'success',
            'message': 'Notification sent',
            'notification': notification
        }
        
    except Exception as e:
        logger.error(f"Error sending notification: {str(e)}")
        return {
            'status': 'error',
            'message': f"Error sending notification: {str(e)}"
        }

# Add a task to check batch status
@celery_app.task(bind=True, name='tasks.check_batch_status')
def check_batch_status(self, batch_id, user_id=None):
    """Check the status of a batch processing job.
    
    Args:
        batch_id (str): The ID of the batch to check
        user_id (str, optional): The ID of the user who owns the batch
        
    Returns:
        dict: Batch information including status, progress, and results
    """
    try:
        if not user_id:
            logger.warning("No user_id provided for batch status check")
            return {
                'batch_id': batch_id,
                'status': 'error',
                'message': 'User ID is required',
                'timestamp': datetime.now().isoformat()
            }

        # Get the user and batch info
        user = User.get(user_id)
        if not user or not hasattr(user, 'batch_jobs'):
            return {
                'batch_id': batch_id,
                'status': 'not_found',
                'message': 'Batch job not found',
                'timestamp': datetime.now().isoformat()
            }

        batch_info = user.batch_jobs.get(batch_id)
        if not batch_info:
            return {
                'batch_id': batch_id,
                'status': 'not_found',
                'message': 'Batch job not found',
                'timestamp': datetime.now().isoformat()
            }
        
        # Calculate progress
        total = batch_info.get('file_count', 0)
        processed = batch_info.get('processed_count', 0)
        success = batch_info.get('success_count', 0)
        failed = batch_info.get('failed_count', 0)
        
        # Calculate progress percentage
        if total > 0:
            progress = min(100, int((processed / total) * 100))
        else:
            progress = 0
            
        # Determine status if not explicitly set
        current_status = batch_info.get('status', 'unknown')
        if current_status == 'unknown':
            if processed >= total > 0:
                current_status = 'completed' if failed == 0 else 'partial_success'
            else:
                current_status = 'processing'
        
        # Prepare response
        response = {
            'batch_id': batch_id,
            'status': current_status,
            'progress': progress,
            'processed': processed,
            'total': total,
            'success': success,
            'failed': failed,
            'start_time': batch_info.get('start_time'),
            'end_time': batch_info.get('end_time'),
            'tool_name': batch_info.get('tool_name'),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add file results if available
        if 'file_results' in batch_info:
            response['file_results'] = batch_info['file_results']
            
        # Add error details if batch failed
        if current_status == 'failed' and 'error' in batch_info:
            response['error'] = batch_info['error']
        
        return response
        
    except Exception as e:
        error_msg = f"Error checking batch status: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            'batch_id': batch_id,
            'status': 'error',
            'message': error_msg,
            'timestamp': datetime.now().isoformat()
        }
