import asyncio
import json
import os
import sys
import time
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional, Union

import nest_asyncio
from dotenv import load_dotenv
from groq import Groq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import ListResourcesRequest, ReadResourceRequest
from termcolor import colored

# Apply nest_asyncio to allow nested event loops (needed for Jupyter/IPython)
nest_asyncio.apply()

# Load environment variables
load_dotenv("../.env")

# Configure Groq API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is required")


class MCPGroqClient:
    """Client for interacting with Groq models using MCP tools with chat interface."""

    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        """Initialize the Groq MCP client with chat capabilities.

        Args:
            model: The Groq model to use.
        """
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        self.model = model
        self.available_resources = []
        self.stdio: Optional[Any] = None
        self.write: Optional[Any] = None
        self.conversation_history: List[Dict[str, str]] = []
        self.tools = []

    async def connect_to_server(self, server_script_path: str = "server.py"):
        """Connect to an MCP server.

        Args:
            server_script_path: Path to the server script.
        """
        # Server configuration
        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path],
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

    async def get_mcp_tools(self) -> List[Dict[str, Any]]:
        """Get available tools from the MCP server in OpenAI format.

        Returns:
            A list of tools in OpenAI format.
        """
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

    async def list_resources(self) -> list:
        """List all available resources from the MCP server.
        
        Returns:
            List of available resource URIs with descriptions
        """
        if not self.available_resources and self.session:
            try:
                # List all available resources using the list_resources tool
                resources_result = await self.session.call_tool("list_resources", {})
                if resources_result and hasattr(resources_result, 'content'):
                    self.available_resources = [{"uri": "schema://main", "description": "Database schema for the ecommerce application"}]
                else:
                    self.available_resources = []
            except Exception as e:
                print(f"Error listing resources: {e}")
                return []
        return self.available_resources or []

    async def read_resource(self, resource_uri: str) -> str:
        """Read the content of a specific resource.
        
        Args:
            resource_uri: URI of the resource to read (e.g., 'schema://main')
            
        Returns:
            The content of the resource as a string
        """
        if not self.session:
            return "Error: Not connected to MCP server"
            
        try:
            # Use the get_resource tool to fetch the resource content
            result = await self.session.call_tool(
                "get_resource",
                {"resource_uri": resource_uri}
            )

            print("Result:", result)
            
            if hasattr(result, 'content') and result.content:
                # Return the first content item's text
                return result.content[0].text or ""
            return "No content available"
            
        except Exception as e:
            return f"Error reading resource {resource_uri}: {str(e)}"

    async def _select_relevant_resource(self, query: str, resources: list) -> dict:
        """Select the most relevant resource for the given query.
        
        Args:
            query: The user query
            resources: List of available resources
            
        Returns:
            Selected resource or None if no relevant resource found
        """
        if not resources:
            return None
            
        # If only one resource, no need to select
        if len(resources) == 1:
            return resources[0]
            
        # Ask LLM to select the most relevant resource
        resource_descriptions = "\n".join(
            f"{i+1}. {r['uri']}: {r['description']}" 
            for i, r in enumerate(resources)
        )
        
        selection_prompt = f"""
        Based on the user query, select the most relevant resource number.
        Only respond with the number, nothing else.
        
        Query: {query}
        
        Available Resources:
        {resource_descriptions}
        """
        
        try:
            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a resource selection assistant. Select the most relevant resource for the given query."},
                    {"role": "user", "content": selection_prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            selected_idx = int(response.choices[0].message.content.strip()) - 1
            if 0 <= selected_idx < len(resources):
                return resources[selected_idx]
        except (ValueError, IndexError, AttributeError) as e:
            print(f"Error selecting resource: {e}")
            
        # Fallback to first resource if selection fails
        return resources[0]

    async def process_query(self, query: str) -> str:
        """Process a query using Groq and available MCP tools and resources.

        Args:
            query: The user query.

        Returns:
            The response from Groq.
        """
        # Get available tools and resources
        tools = await self.get_mcp_tools()
        resources = await self.list_resources()
        
        # Select the most relevant resource for the query
        selected_resource = await self._select_relevant_resource(query, resources)
        resource_content = ""
        
        if selected_resource:
            print(f"Selected resource: {selected_resource['uri']}")
            resource_content = await self.read_resource(selected_resource['uri'])
            print(f"Resource content: {resource_content[:200]}...")  # Print first 200 chars for debugging

        # Prepare system message with resource information
        system_message = """You are a helpful assistant with access to tools and resources.
        Available resources:"""
        for resource in resources:
            system_message += f"\n- {resource['uri']}: {resource['description']}"
            
        # Add the selected resource content to the context
        if resource_content:
            system_message += f"\n\nSelected resource content for reference:\n{resource_content}"
            
        # Initial Groq API call
        response = self.groq_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": query}
            ],
            tools=tools,
            tool_choice="auto"
        )

        # Get assistant's response
        assistant_message = response.choices[0].message

        # Initialize conversation with user query and assistant response
        messages = [
            {"role": "user", "content": query},
            assistant_message,
        ]

        # Handle tool calls if present
        if assistant_message.tool_calls:
            # Process each tool call
            for tool_call in assistant_message.tool_calls:
                # Execute tool call
                result = await self.session.call_tool(
                    tool_call.function.name,
                    arguments=json.loads(tool_call.function.arguments),
                )

                # Add tool response to conversation
                # Truncate content to 12,000 characters to stay within token limits
                truncated_content = result.content[0].text[:12000]
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": truncated_content,
                    }
                )

            print("Messages:", messages)

            # Get final response from Groq with tool results
            final_response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="none"  # Don't allow more tool calls
            )
            
            # Extract the final response content
            final_content = final_response.choices[0].message.content
            
            # Include tool call results in the response
            tool_results = []
            for msg in messages:
                if hasattr(msg, 'role') and msg.role == "tool":
                    if hasattr(msg, 'content'):
                        tool_results.append(msg.content)
                elif isinstance(msg, dict) and msg.get("role") == "tool":
                    if "content" in msg:
                        tool_results.append(msg["content"])
            
            if tool_results:
                results_text = "\n\nQuery Results:\n" + "\n".join(tool_results)
                final_content = f"{final_content}{results_text}"
            
            print("Final response:", final_content)
            return final_content

        # No tool calls, just return the direct response
        return assistant_message.content

    def _print_with_typing_effect(self, text: str, color: str = 'green'):
        """Print text with a typing effect."""
        for char in text:
            print(colored(char, color), end='', flush=True)
            time.sleep(0.02)  # Adjust typing speed
        print()

    async def chat_loop(self):
        """Run the interactive chat loop."""
        print(colored("\n=== MCP Chat Assistant ===", "green", attrs=["bold"]))
        print(colored("Type 'exit' or 'quit' to end the session\n", "yellow"))

        while True:
            try:
                # Get user input with colored prompt
                user_input = input(colored("\nYou: ", "blue", attrs=["bold"]))
                
                # Check for exit commands
                if user_input.lower() in ("exit", "quit"):
                    print(colored("\nGoodbye!", "green"))
                    break
                
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
    client = MCPGroqClient()
    
    try:
        # Connect to the server
        print(colored("Connecting to MCP server...", "yellow"))
        await client.connect_to_server()
        
        # Load available tools
        client.tools = await client.get_mcp_tools()
        print(colored("Connected! Type your message below.", "green"))
        
        # Start the chat loop
        await client.chat_loop()
        
    except Exception as e:
        print(colored(f"An error occurred: {e}", "red"))
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())