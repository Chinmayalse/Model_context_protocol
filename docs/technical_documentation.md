# Medical Report Processing Application with MCP: Technical Documentation

This document provides a comprehensive technical explanation of the Medical Report Processing application, focusing on the Model Context Protocol (MCP) implementation and the end-to-end flow from user interaction to result display.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [MCP Server Implementation](#mcp-server-implementation)
3. [MCP Client Implementation](#mcp-client-implementation)
4. [Web Application](#web-application)
5. [End-to-End Flow](#end-to-end-flow)
6. [Key Files and Components](#key-files-and-components)

## System Architecture

The application follows a client-server architecture using the Model Context Protocol (MCP):

```
┌─────────────────┐                          ┌─────────────────┐
│                 │                          │                 │
│  Web Interface  │─────────────────────────►│   MCP Server    │
│  (chat_app.py)  │                          │  (mcp_chat.py)  │
│                 │                          │                 │
└─────────────────┘                          └─────────────────┘
                                                    ▲
┌─────────────────┐                                 │
│                 │                                 │
│  CLI Interface  │─────────────────────────────────┘
│ (mcp_client.py) │
│                 │
└─────────────────┘
```

- **Web Interface (chat_app.py)**: A Flask-based web application that provides a user interface and connects directly to the MCP server
- **CLI Interface (mcp_client.py)**: A separate command-line interface that also connects to the MCP server
- **MCP Server (mcp_chat.py)**: Registers and executes tools for processing medical reports

## MCP Server Implementation

The MCP server is implemented in `mcp_chat.py` and serves as the core processing engine.

### Server Initialization

```python
# Initialize FastMCP with proper configuration
mcp = FastMCP(
    name="Medical Report Processing",
    description="Tools for processing and summarizing medical reports",
    host="0.0.0.0",  # Listen on all interfaces
    port=8050      # Port to use for the server
)
```

### Tool Registration

The server registers two main tools using the `@mcp.tool` decorator:

1. **process_medical_report**: Extracts and classifies text from medical reports
2. **summarize_medical_report**: Generates a structured summary of medical reports

```python
@mcp.tool
def process_medical_report(file_path: str, ctx: Context, extraction_method: str = "auto"):
    """
    Process a medical report file - extract text and classify it in a single operation.
    """
    # Implementation details...
```

```python
@mcp.tool
def summarize_medical_report(file_path: str, ctx: Context):
    """
    Summarize a medical report file - extract text and provide a structured summary.
    """
    # Implementation details...
```

### Server Startup

The server can be started in different ways:

1. **Direct execution**: When run directly with `python mcp_chat.py`
2. **MCP CLI**: When run through the MCP CLI with `mcp dev mcp_chat.py`
3. **Transport options**: Supports stdio, HTTP, or WebSocket transports

```python
# When run directly (not through mcp dev)
if not os.environ.get("MCP_MANAGED", False):
    # Check if we should use stdio transport directly
    if "--transport=stdio" in sys.argv or ("--transport" not in " ".join(sys.argv) and len(sys.argv) == 1):
        # Direct stdio transport doesn't need asyncio wrapping
        mcp.run(transport="stdio")
    else:
        # For HTTP/WebSocket transport, use asyncio
        import asyncio
        asyncio.run(main())
```

## MCP Client Implementations

The application has two separate client implementations that connect to the MCP server:

1. **Web Application Client** (in `chat_app.py`)
2. **Command-Line Interface Client** (in `mcp_client.py`)

### Web Application Client (chat_app.py)

The web application has its own embedded MCP client implementation that connects directly to the MCP server:

```python
class MCPClient:
    def __init__(self):
        self.session = None
        self.exit_stack = AsyncExitStack()
        self.stdio = None
        self.write = None
        self.logs = []
    
    async def connect(self):
        """Connect to the MCP server"""
        print("\n[CLIENT] Connecting to MCP server...")
        # Server configuration
        server_params = StdioServerParameters(
            command="python",
            args=[MCP_SERVER_SCRIPT, "--transport=stdio"],
        )
        
        # Connect to the server
        print("[CLIENT] Establishing stdio transport...")
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        print("[CLIENT] Creating client session...")
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )
        
        # Initialize the connection
        print("[CLIENT] Initializing connection...")
        await self.session.initialize()
        print("[CLIENT] Successfully connected to MCP server!")
        return True
```

### Command-Line Interface Client (mcp_client.py)

The `mcp_client.py` file contains a separate, standalone client implementation with an interactive chat interface:

```python
class MCPGeminiClient:
    """Client for interacting with Gemini models using MCP tools with chat interface."""

    def __init__(self, model: str = "gemini-2.0-flash"):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.model = model
        self.available_resources = []
        self.stdio: Optional[Any] = None
        self.write: Optional[Any] = None
        self.conversation_history: List[Dict[str, str]] = []
        self.tools = []
        
        # Initialize Gemini model
        self.genai_model = genai.GenerativeModel(model)

    async def connect_to_server(self, server_script_path: str = "mcp_server.py", use_mcp_dev: bool = False):
        # Server configuration
        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path, "--transport=stdio"],
        )

        # Connect to the server
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        # Initialize the connection
        await self.session.initialize()
```

Both client implementations connect to the same MCP server but serve different purposes:
- The web application client handles HTTP requests and provides a web interface
- The command-line client provides an interactive chat experience in the terminal

## Web Application

The web application is implemented in `chat_app.py` and provides the user interface for interacting with the MCP tools.

### Flask Application Setup

```python
# Initialize Flask application
app = Flask(__name__)
app.secret_key = os.urandom(24)  # For flash messages and session

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chat_uploads')
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chat_results')
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

# Create necessary directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
```

### Tool Detection

The application automatically detects which tool to use based on the user's message:

```python
def detect_tool(message, has_file=False):
    """
    Detect which tool to use based on the user's message
    Returns: tool_name, confidence
    """
    message = message.lower()
    
    # Default to process_medical_report if a file is uploaded without specific instructions
    if has_file and not message:
        return "process_medical_report", 0.9
    
    # Keywords for summarization
    summarize_keywords = ["summarize", "summary", "summarization", "brief", "overview"]
    
    # Keywords for processing
    process_keywords = ["process", "extract", "analyze", "details", "information"]
    
    # Count keyword matches
    summarize_count = sum(1 for keyword in summarize_keywords if keyword in message)
    process_count = sum(1 for keyword in process_keywords if keyword in message)
    
    # Determine which tool to use based on keyword matches
    if summarize_count > process_count:
        return "summarize_medical_report", 0.8
    elif process_count > 0 or has_file:
        return "process_medical_report", 0.8
    else:
        # No clear tool match, handle as a general question
        return "general_question", 0.5
```

### File Processing

The application processes files using the MCP client:

```python
async def process_file(file_path, tool_name):
    """Process a file using the specified tool through MCP client"""
    client = MCPClient()
    
    try:
        # Connect to the MCP server
        await client.connect()
        
        # Call the appropriate tool
        if tool_name == "process_medical_report":
            result = await client.call_tool("process_medical_report", file_path=file_path)
        elif tool_name == "summarize_medical_report":
            result = await client.call_tool("summarize_medical_report", file_path=file_path)
        else:
            return "Unknown tool specified", "", None
        
        # Parse the result
        try:
            result_dict = json.loads(result)
            
            # Save the result to a JSON file
            filename_base = os.path.splitext(os.path.basename(file_path))[0]
            json_filename = save_result_to_json(result_dict, filename_base, tool_name)
            
            # Generate a simple completion message without any details
            if tool_name == "summarize_medical_report":
                response = "Process completed. Medical report has been summarized successfully."
            else:
                response = "Process completed. Medical report has been processed successfully."
                
            return response, result, json_filename
            
        except json.JSONDecodeError:
            return f"Error: Could not parse the result as JSON", result, None
    finally:
        await client.close()
```

## End-to-End Flow

Here's the complete flow from user interaction to result display:

1. **User Uploads File**: 
   - User uploads a medical report file through the web interface
   - User can optionally specify whether to process or summarize the report

2. **Web App Receives Request**:
   - Flask route `/chat` handles the POST request
   - Saves the uploaded file to the `UPLOAD_FOLDER`
   - Uses Gemini to determine which tool to use based on the user's message

3. **Direct MCP Connection**:
   - Web app creates an instance of its embedded `MCPClient` class
   - Client connects directly to the MCP server using stdio transport
   - Server is started with `python mcp_chat.py --transport=stdio`

4. **Tool Execution**:
   - Client calls the appropriate tool on the server (summarize_medical_report or process_medical_report)
   - Server executes the tool with the file path as input
   - Tool processes the file (extracts text, classifies, enhances)
   - Results are returned to the client

5. **Result Processing**:
   - Web app receives the results from the MCP server
   - Results are saved to a JSON file in the `RESULTS_FOLDER`
   - A simple "Process completed" message is returned to the user
   - A link to view the detailed results is provided

6. **Result Viewing**:
   - User clicks on the "View Result" link
   - Flask route `/view_result/<filename>` renders the results
   - Results are displayed in a formatted view using the `view_result.html` template

### Alternative CLI Flow

The application also provides a command-line interface through `mcp_client.py`:

1. **User Starts CLI Client**:
   - User runs `python mcp_client.py`
   - CLI client initializes and connects to the MCP server

2. **Interactive Chat**:
   - User enters queries or commands in the terminal
   - CLI client processes the input and determines which MCP tool to call

3. **Tool Execution**:
   - CLI client calls the appropriate tool on the MCP server
   - Results are displayed directly in the terminal

## Key Files and Components

### Core Components

- **mcp_chat.py**: MCP server implementation with tool registration for medical report processing
- **chat_app.py**: Flask web application with embedded MCP client for web interface
- **mcp_client.py**: Standalone CLI client with interactive chat interface

### Web Application Files

- **templates/chat.html**: Main chat interface template
- **templates/view_result.html**: Template for viewing detailed results
- **templates/chat_results.html**: Template for listing all results

### Directories

- **chat_uploads/**: Temporary storage for uploaded files
- **chat_results/**: Storage for processed results in JSON format
- **System_prompts/**: Contains system prompts for different report types

## Conclusion

The Medical Report Processing application leverages the Model Context Protocol (MCP) to create a flexible, modular system for processing and summarizing medical reports. The client-server architecture allows for clean separation of concerns, with the web interface handling user interactions, the MCP client managing communication, and the MCP server executing the processing tools.

This architecture makes it easy to add new tools and features without modifying the core application logic, providing a solid foundation for future enhancements.
