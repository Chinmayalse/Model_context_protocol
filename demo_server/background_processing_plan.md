# Background Processing Implementation Plan for Medical Reports

## Overview

This document outlines the plan for implementing background processing capabilities for the medical report processing application to reduce user wait times and enable parallel processing of multiple reports.

## Current System Analysis

The current implementation processes medical reports synchronously within the HTTP request lifecycle:
1. User uploads a file and submits a message
2. The server processes the file using the MCP tools during the HTTP request
3. The user must wait for processing to complete before receiving a response
4. Only one report can be processed at a time per user session

This approach has several limitations:
- Long wait times for users, especially with complex reports
- Potential for request timeouts with large files
- Inability to process multiple reports in parallel
- Poor user experience as the interface appears frozen during processing

## Implementation Goals

1. **Reduce User Wait Times**: Implement asynchronous processing to allow users to continue interacting with the application while reports are being processed
2. **Enable Parallel Processing**: Process multiple reports simultaneously to improve throughput
3. **Provide Processing Status Updates**: Keep users informed about the progress of their report processing
4. **Improve Error Handling**: Better manage and communicate processing errors
5. **Maintain Data Consistency**: Ensure all processing results are properly stored and associated with the correct user and chat

## Technical Solution

### 1. Task Queue System

We will implement a task queue system using Celery with Redis as the message broker:

- **Celery**: A distributed task queue that will manage background processing jobs
- **Redis**: An in-memory data store that will serve as the message broker between the web application and worker processes

### 2. Database Schema Updates

Add a new table to track processing jobs:

```sql
CREATE TABLE processing_jobs (
    id VARCHAR(100) PRIMARY KEY, -- UUID for the job
    user_id INTEGER NOT NULL,
    chat_id VARCHAR(100),
    file_path VARCHAR(255) NOT NULL,
    tool_name VARCHAR(100) NOT NULL,
    tool_args JSON,
    status VARCHAR(50) NOT NULL, -- 'pending', 'processing', 'completed', 'failed'
    result_filename VARCHAR(255),
    error_message TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE SET NULL
);
```

### 3. Implementation Components

#### 3.1. Backend Changes

1. **Task Queue Integration**:
   - Install and configure Celery and Redis
   - Create Celery worker configuration
   - Define task functions for processing reports

2. **API Endpoints**:
   - Modify the `/chat` endpoint to queue processing tasks instead of processing synchronously
   - Add a new `/job_status/<job_id>` endpoint to check processing status
   - Add a new `/jobs` endpoint to list all pending and recent jobs for a user

3. **Job Management**:
   - Create a `JobManager` class to handle job creation, status updates, and result retrieval
   - Implement job status tracking and notification

#### 3.2. Frontend Changes

1. **UI Updates**:
   - Add a job status indicator for pending and in-progress jobs
   - Implement automatic polling for job status updates
   - Add a jobs list view to see all pending and recent jobs

2. **Real-time Updates**:
   - Implement WebSocket connection for real-time job status updates
   - Update chat interface to show processing status

### 4. Implementation Steps

#### Phase 1: Core Infrastructure

1. Install and configure dependencies:
   - Add Celery, Redis, and WebSocket libraries to requirements.txt
   - Set up Redis server
   - Configure Celery workers

2. Create database schema for job tracking:
   - Add the processing_jobs table
   - Create JobManager class for database interactions

3. Refactor the file processing logic:
   - Extract the processing logic from the HTTP request handlers
   - Create Celery task functions for background processing

#### Phase 2: API and Backend Integration

1. Modify the chat endpoint:
   - Queue processing tasks instead of processing synchronously
   - Return job ID immediately to the client

2. Create job status endpoints:
   - Implement `/job_status/<job_id>` endpoint
   - Implement `/jobs` endpoint for listing jobs

3. Implement WebSocket notifications:
   - Set up WebSocket server
   - Implement job status change notifications

#### Phase 3: Frontend Updates

1. Update the chat interface:
   - Add job status indicators
   - Implement polling for job status updates
   - Display processing status messages

2. Create a jobs dashboard:
   - List all pending and recent jobs
   - Allow cancellation of pending jobs
   - Show detailed job information

### 5. Technical Details

#### 5.1. Celery Task Definition

```python
@celery.task(bind=True)
def process_file_task(self, file_path, tool_name, tool_args, user_id, chat_id, job_id):
    """Process a file using the specified tool through MCP client as a background task"""
    # Update job status to 'processing'
    JobManager.update_job_status(job_id, 'processing')
    
    try:
        # Run the processing logic (similar to the current process_file function)
        result, _, json_filename = run_processing(file_path, tool_name, tool_args, chat_id)
        
        # Update job status to 'completed'
        JobManager.update_job_status(job_id, 'completed', result_filename=json_filename)
        
        # Add the result to the chat history
        user = User.get(user_id)
        if user and chat_id:
            metadata = {
                "tool_used": tool_name,
                "result_file": json_filename,
                "view_url": f"/view_result/{json_filename}",
                "job_id": job_id
            }
            user.add_message_to_chat(chat_id, result, is_user=False, metadata=metadata)
        
        return json_filename
    except Exception as e:
        # Update job status to 'failed'
        error_message = str(e)
        JobManager.update_job_status(job_id, 'failed', error_message=error_message)
        
        # Add error message to chat
        if user and chat_id:
            user.add_message_to_chat(
                chat_id, 
                f"Error processing file: {error_message}", 
                is_user=False, 
                metadata={"tool_used": "error", "job_id": job_id}
            )
        
        raise
```

#### 5.2. Modified Chat Endpoint

```python
@app.route('/chat', methods=['POST'])
@login_required
def chat():
    message = request.form.get('message', '')
    chat_id = request.form.get('chat_id', '')
    
    # If no chat_id provided, create a new one
    if not chat_id:
        chat_id = str(uuid.uuid4())
        title = message[:30] + "..." if len(message) > 30 else message
        current_user.add_chat(chat_id, title, datetime.now().strftime("%Y-%m-%d %H:%M"))
    
    # Store the user message in chat history
    current_user.add_message_to_chat(chat_id, message, is_user=True)
    
    # Check if a file was uploaded
    has_file = False
    file_path = None
    
    if 'file' in request.files and request.files['file'].filename:
        file = request.files['file']
        
        if file and allowed_file(file.filename):
            has_file = True
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
    
    try:
        if has_file:
            # Use Gemini to determine which tool to call
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                tool_name, confidence, tool_args = loop.run_until_complete(
                    get_tool_with_gemini(message, has_file)
                )
            finally:
                loop.close()
            
            # Create a job ID
            job_id = str(uuid.uuid4())
            
            # Create a job record
            JobManager.create_job(
                job_id=job_id,
                user_id=current_user.id,
                chat_id=chat_id,
                file_path=file_path,
                tool_name=tool_name,
                tool_args=tool_args
            )
            
            # Queue the processing task
            process_file_task.delay(
                file_path=file_path,
                tool_name=tool_name,
                tool_args=tool_args,
                user_id=current_user.id,
                chat_id=chat_id,
                job_id=job_id
            )
            
            # Send an immediate response with job information
            response = f"Your file is being processed. You'll be notified when it's complete."
            
            # Add a placeholder message in the chat
            metadata = {
                "tool_used": tool_name,
                "confidence": confidence,
                "job_id": job_id,
                "status": "processing"
            }
            current_user.add_message_to_chat(chat_id, response, is_user=False, metadata=metadata)
            
            return jsonify({
                "response": response,
                "tool_used": tool_name,
                "confidence": confidence,
                "chat_id": chat_id,
                "job_id": job_id,
                "status": "processing"
            })
        else:
            # Handle general questions (no changes needed)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response, tool_result, json_filename = loop.run_until_complete(
                    answer_general_question(message)
                )
                tool_name = "general_question"
                confidence = 0.8
            finally:
                loop.close()
                
            # Store the assistant response in chat history with metadata
            metadata = {"tool_used": tool_name, "confidence": confidence}
            if json_filename:
                metadata["result_file"] = json_filename
                metadata["view_url"] = f"/view_result/{json_filename}"
                
            current_user.add_message_to_chat(chat_id, response, is_user=False, metadata=metadata)
            
            result_data = {
                "response": response,
                "tool_used": tool_name,
                "confidence": confidence,
                "chat_id": chat_id
            }
            
            if json_filename:
                result_data["result_file"] = json_filename
                result_data["view_url"] = f"/view_result/{json_filename}"
                
            return jsonify(result_data)
    except Exception as e:
        response = f"Sorry, there was an error processing your request: {str(e)}"
        
        # Store the error response in chat history
        metadata = {"tool_used": "error", "confidence": 0}
        current_user.add_message_to_chat(chat_id, response, is_user=False, metadata=metadata)
        
        return jsonify({
            "response": response,
            "tool_used": "error",
            "confidence": 0,
            "chat_id": chat_id
        })
```

#### 5.3. WebSocket Integration

```python
# In app initialization
socketio = SocketIO(app, cors_allowed_origins="*")

# Event handler for job status updates
@socketio.on('connect')
def handle_connect():
    if current_user.is_authenticated:
        join_room(f"user_{current_user.id}")

# Function to emit job status updates
def emit_job_update(user_id, job_id, status, result=None, error=None):
    data = {
        "job_id": job_id,
        "status": status
    }
    
    if result:
        data["result"] = result
    
    if error:
        data["error"] = error
        
    socketio.emit('job_update', data, room=f"user_{user_id}")
```

## 6. Deployment Considerations

1. **Redis Server**:
   - Set up a Redis server for development and production
   - Configure appropriate security settings
   - Consider using a managed Redis service in production

2. **Celery Workers**:
   - Configure appropriate number of worker processes based on server capacity
   - Set up monitoring for worker processes
   - Implement proper error handling and retry mechanisms

3. **Scaling**:
   - Design the system to allow horizontal scaling of worker processes
   - Consider containerization for easier deployment and scaling

## 7. Testing Plan

1. **Unit Tests**:
   - Test job creation and status updates
   - Test task execution and error handling
   - Test WebSocket notifications

2. **Integration Tests**:
   - Test end-to-end flow from file upload to result display
   - Test parallel processing of multiple files
   - Test error scenarios and recovery

3. **Load Tests**:
   - Test system performance under load
   - Measure processing throughput and latency
   - Identify bottlenecks and optimize

## 8. Timeline and Milestones

1. **Week 1: Infrastructure Setup**
   - Install and configure Celery and Redis
   - Create database schema for job tracking
   - Set up basic task definitions

2. **Week 2: Backend Implementation**
   - Implement job management system
   - Modify API endpoints for asynchronous processing
   - Implement WebSocket notifications

3. **Week 3: Frontend Updates**
   - Update chat interface to show job status
   - Implement job status polling
   - Create jobs dashboard

4. **Week 4: Testing and Optimization**
   - Conduct thorough testing
   - Fix bugs and optimize performance
   - Document the implementation

## 9. WebSocket Connection Management

### 9.1. Current Issues

The current implementation has several issues with WebSocket connection management:

1. **On-demand Connections**: WebSocket connections to the MCP server are created on-demand only when a file is uploaded, leading to connection overhead and potential failures.
2. **Connection Errors**: The application experiences errors like `'ClientConnection' object has no attribute 'closed'` due to improper connection handling.
3. **Context Handling**: The MCP server's Context object is not properly integrated with the WebSocket handler, causing errors like `ValueError: Context is not available outside of a request`.
4. **Data Type Conversion**: Issues with converting between different data types (e.g., MapComposite objects) when passing data between client and server.

### 9.2. Proposed Improvements

#### 9.2.1. Persistent WebSocket Connection

Implement a persistent WebSocket connection that is maintained throughout the user's session:

```python
# In app initialization
class WebSocketManager:
    def __init__(self):
        self.connections = {}
        
    async def get_connection(self, user_id):
        if user_id not in self.connections or self.connections[user_id].closed:
            self.connections[user_id] = await websockets.connect(MCP_WEBSOCKET_URL)
            # Initialize connection
            greeting = await self.connections[user_id].recv()
            print(f"[CLIENT] Connected for user {user_id}")
        return self.connections[user_id]
        
    async def close_all(self):
        for user_id, conn in self.connections.items():
            try:
                await conn.close()
            except Exception as e:
                print(f"Error closing connection for user {user_id}: {str(e)}")
```

#### 9.2.2. Custom Context Implementation

Implement a custom WebSocketContext that properly works with the WebSocket handler:

```python
class WebSocketContext:
    """A simplified context for WebSocket handlers that mimics the FastMCP Context."""
    
    def __init__(self, websocket=None):
        self.websocket = websocket
        self.messages = []
    
    async def report_progress(self, current, total):
        """Report progress to the client."""
        if self.websocket:
            try:
                await self.websocket.send(json.dumps({
                    'event': 'progress',
                    'current': current,
                    'total': total
                }))
            except Exception as e:
                print(f"Error reporting progress: {str(e)}")
```

#### 9.2.3. Robust Data Type Handling

Implement robust data type conversion between client and server:

```python
def convert_to_serializable(obj):
    """Convert any object to a JSON-serializable format."""
    if hasattr(obj, '__dict__'):
        return {k: convert_to_serializable(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()
    return obj
```

## 10. Future Enhancements

1. **Job Prioritization**: Implement a priority system for processing jobs
2. **Batch Processing**: Allow users to upload multiple files for batch processing
3. **Processing Quotas**: Implement user-based quotas for processing jobs
4. **Advanced Monitoring**: Add detailed monitoring and analytics for processing jobs
5. **Scheduled Processing**: Allow users to schedule processing jobs for later execution
6. **Connection Pooling**: Implement a connection pool for WebSocket connections to improve performance and reliability

## Conclusion

Implementing background processing for medical reports will significantly improve the user experience by reducing wait times and enabling parallel processing. The proposed solution using Celery and Redis provides a robust and scalable architecture that can be extended to support additional features in the future.
