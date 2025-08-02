import asyncio
import json
import os
import sys
import time
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional, Union

import nest_asyncio
from dotenv import load_dotenv
from google import generativeai as genai
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import ListResourcesRequest, ReadResourceRequest
from termcolor import colored

# Apply nest_asyncio to allow nested event loops (needed for Jupyter/IPython)
nest_asyncio.apply()

# Load environment variables
try:
    load_dotenv()
except ImportError:
    print("dotenv not installed. Using environment variables directly.")

# Configure Google API
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyDDBBTclRJjECny3q01Y57TIG9C6ZfVuTY")
genai.configure(api_key=GOOGLE_API_KEY)


class MCPGeminiClient:
    """Client for interacting with Gemini models using MCP tools with chat interface."""

    def __init__(self, model: str = "gemini-2.0-flash"):
        """Initialize the Gemini MCP client with chat capabilities.

        Args:
            model: The Gemini model to use.
        """
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
        """Connect to an MCP server.

        Args:
            server_script_path: Path to the server script.
            use_mcp_dev: Whether to use 'mcp dev' command (recommended) or direct Python execution.
        """
        # Server configuration
        if use_mcp_dev:
            server_params = StdioServerParameters(
                command="mcp",
                args=["dev", server_script_path],
            )
        else:
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

        # List available tools
        tools_result = await self.session.list_tools()
        print("\nConnected to server with tools:")
        for tool in tools_result.tools:
            print(f"  - {tool.name}: {tool.description}")
            
        return tools_result.tools

    async def get_mcp_tools(self) -> List[Dict[str, Any]]:
        """Get available tools from the MCP server in OpenAI format.

        Returns:
            A list of tools in OpenAI format.
        """
        if not self.session:
            return []
            
        tools_result = await self.session.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in tools_result.tools
        ]

    async def process_query(self, query: str) -> str:
        """Process a query using Gemini and available MCP tools.

        Args:
            query: The user query.

        Returns:
            The response from Gemini.
        """
        if not self.session:
            return "Error: Not connected to MCP server"
            
        # Get available tools
        tools = await self.get_mcp_tools()
        
        # Convert tools to Gemini format
        gemini_tools = []
        for tool in tools:
            function_info = tool["function"]
            gemini_tools.append({
                "function_declarations": [{
                    "name": function_info["name"],
                    "description": function_info["description"],
                    "parameters": function_info["parameters"]
                }]
            })
        
        # Prepare system message
        system_message = """You are a helpful medical assistant with access to tools for processing and summarizing medical reports.
        When a user asks you to process or summarize a medical report, use the appropriate tool.
        For processing reports, use the process_medical_report tool.
        For summarizing reports, use the summarize_medical_report tool.
        Always provide the full file path to the report when calling these tools."""
        
        # Initial Gemini API call
        chat = self.genai_model.start_chat(history=[])
        response = chat.send_message(
            query,
            tools=gemini_tools
        )

        # Check if tool calls are present
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and candidate.content:
                parts = candidate.content.parts
                for part in parts:
                    if hasattr(part, "function_call") and part.function_call:
                        # Extract tool call information
                        tool_name = part.function_call.name
                        tool_args = json.loads(part.function_call.args)
                        
                        print(f"\nCalling tool: {tool_name} with args: {tool_args}")
                        
                        # Execute tool call
                        try:
                            result = await self.session.call_tool(
                                tool_name,
                                arguments=tool_args,
                            )
                            
                            # Extract content from result
                            if hasattr(result, "content") and result.content:
                                tool_result = result.content[0].text
                            else:
                                tool_result = str(result)
                                
                            # Truncate if too long
                            if len(tool_result) > 12000:
                                tool_result = tool_result[:12000] + "... [truncated]"
                                
                            # Send tool result back to Gemini
                            response = chat.send_message(
                                f"Tool {tool_name} returned: {tool_result}"
                            )
                            
                            return response.text
                        except Exception as e:
                            return f"Error executing tool {tool_name}: {str(e)}"
        
        # No tool calls, just return the direct response
        return response.text

    def _print_with_typing_effect(self, text: str, color: str = 'green'):
        """Print text with a typing effect."""
        for char in text:
            print(colored(char, color), end='', flush=True)
            time.sleep(0.01)  # Adjust typing speed
        print()

    async def chat_loop(self):
        """Run the interactive chat loop."""
        print(colored("\n=== Medical Report Processing Chat Assistant ===", "green", attrs=["bold"]))
        print(colored("Type 'exit' or 'quit' to end the session", "yellow"))
        print(colored("Type 'tools' to see available tools", "yellow"))

        while True:
            try:
                # Get user input with colored prompt
                user_input = input(colored("\nYou: ", "blue", attrs=["bold"]))
                
                # Check for exit commands
                if user_input.lower() in ("exit", "quit"):
                    print(colored("\nGoodbye!", "green"))
                    break
                    
                # Check for tools command
                if user_input.lower() == "tools":
                    tools = await self.get_mcp_tools()
                    print(colored("\nAvailable tools:", "yellow"))
                    for tool in tools:
                        print(colored(f"  - {tool['function']['name']}: {tool['function']['description']}", "cyan"))
                    continue
                
                if not user_input.strip():
                    continue
                
                # Show typing indicator
                print(colored("\nAssistant: ", "green", attrs=["bold"]), end="", flush=True)
                
                # Process the query and get response
                response = await self.process_query(user_input)
                
                # Print the response with typing effect
                self._print_with_typing_effect(response, "green")
                
            except KeyboardInterrupt:
                print(colored("\n\nInterrupted. Type 'exit' to quit.", "yellow"))
                continue
            except Exception as e:
                print(colored(f"\nError: {str(e)}", "red"))
                continue

    async def close(self):
        """Clean up resources."""
        try:
            await self.exit_stack.aclose()
        except Exception as e:
            print(colored(f"Error during cleanup: {e}", "red"))


async def main():
    """Main entry point for the chat client."""
    import argparse
    parser = argparse.ArgumentParser(description="MCP Medical Report Processing Client")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash", 
                      help="Gemini model to use (default: gemini-2.0-flash)")
    parser.add_argument("--server", type=str, default="mcp_chat.py",
                      help="Path to the server script (default: mcp_chat.py)")
    
    args = parser.parse_args()
    
    client = MCPGeminiClient(model=args.model)
    
    try:
        # Connect to the server
        print(colored("Connecting to MCP server...", "yellow"))
        await client.connect_to_server(server_script_path=args.server)
        
        # Start the chat loop
        await client.chat_loop()
        
    except Exception as e:
        print(colored(f"An error occurred: {e}", "red"))
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
