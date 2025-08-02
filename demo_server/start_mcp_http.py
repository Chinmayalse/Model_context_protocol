"""
HTTP MCP Server Launcher

This script starts the MCP server with HTTP transport.
"""

import os
import subprocess
import sys
import time

def start_mcp_http_server():
    """Start the MCP server with HTTP transport"""
    print("Starting MCP HTTP Server...")
    
    # Get the path to the MCP server script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mcp_server_script = os.path.join(script_dir, "mcp_server.py")
    
    # Start the server with WebSocket transport
    try:
        # Use subprocess to start the server in a new process
        process = subprocess.Popen(
            [sys.executable, mcp_server_script, "--host=0.0.0.0", "--port=8050"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Print the server output
        print("MCP HTTP Server started. Press Ctrl+C to stop.")
        print("Server output:")
        print("-" * 50)
        
        # Monitor the server output
        while True:
            output = process.stdout.readline()
            if output:
                print(output.strip())
            
            # Check if the process has terminated
            if process.poll() is not None:
                break
                
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping MCP HTTP Server...")
        process.terminate()
        process.wait()
        print("MCP HTTP Server stopped.")
    except Exception as e:
        print(f"Error starting MCP HTTP Server: {e}")

if __name__ == "__main__":
    start_mcp_http_server()
