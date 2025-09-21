"""
Core MCP client implementation for connecting to multiple MCP servers
"""

import asyncio
import json
import time
from contextlib import AsyncExitStack
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from anthropic import Anthropic
import openai
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

from .config import Config, MCPServerConfig
from .exceptions import (
    ServerConnectionError,
    ServerInitializationError,
    ToolExecutionError,
    AIProviderError,
    ServerNotFoundError,
    ToolNotFoundError,
    TimeoutError,
)
from .logger import LoggerMixin


class MCPServerConnection:
    """Represents a connection to a single MCP server"""

    def __init__(self, name: str, config: MCPServerConfig, session: ClientSession):
        self.name = name
        self.config = config
        self.session = session
        self.tools: List[Dict[str, Any]] = []
        self.connected_at = time.time()
        self.last_health_check = time.time()
        self.is_healthy = True
        self.tool_call_count = 0
        self.error_count = 0


class GlobalMCPClient(LoggerMixin):
    """
    Production-ready MCP client that can connect to multiple MCP servers
    and orchestrate tool calls through AI models
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the Global MCP Client

        Args:
            config: Configuration instance. If None, will load from default location
        """
        self.config = config or Config()
        self.connections: Dict[str, MCPServerConnection] = {}
        self.tool_to_server: Dict[str, str] = {}
        self.available_tools: List[Dict[str, Any]] = []
        self.exit_stack = AsyncExitStack()
        self.is_initialized = False
        self._ai_client = None

        # Validate configuration
        issues = self.config.validate()
        if issues:
            error_msg = "Configuration validation failed:\n" + "\n".join(
                f"- {issue}" for issue in issues
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.info("Global MCP Client initialized successfully")

    @property
    def ai_client(self) -> Union[Anthropic, openai.OpenAI, genai.GenerativeModel]:
        """Get the AI client instance"""
        if self._ai_client is None:
            if self.config.anthropic_api_key:
                self._ai_client = Anthropic(api_key=self.config.anthropic_api_key)
                self.logger.info("Using Anthropic AI client")
            elif self.config.openai_api_key:
                self._ai_client = openai.OpenAI(api_key=self.config.openai_api_key)
                self.logger.info("Using OpenAI AI client")
            elif self.config.gemini_api_key:
                genai.configure(api_key=self.config.gemini_api_key)
                self._ai_client = genai.GenerativeModel('gemini-1.5-flash')
                self.logger.info("Using Google Gemini AI client")
            else:
                raise AIProviderError("No AI API key configured")

        return self._ai_client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    async def connect_to_server(
        self, name: str, config: MCPServerConfig
    ) -> MCPServerConnection:
        """
        Connect to a single MCP server with retry logic

        Args:
            name: Server name
            config: Server configuration

        Returns:
            MCPServerConnection instance

        Raises:
            ServerConnectionError: If connection fails
            ServerInitializationError: If server initialization fails
        """
        self.logger.info(f"Connecting to MCP server: {name}")

        try:
            # Prepare server parameters
            server_params = StdioServerParameters(
                command=config.command,
                args=config.args,
                env=config.env if config.env else None,
            )

            # Set working directory if specified
            if config.cwd:
                server_params.cwd = config.cwd

            # Create connection
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport

            # Create session with timeout
            session = await asyncio.wait_for(
                self.exit_stack.enter_async_context(ClientSession(read, write)),
                timeout=self.config.global_settings.connection_timeout,
            )

            # Initialize session
            await asyncio.wait_for(
                session.initialize(),
                timeout=self.config.global_settings.initialization_timeout,
            )

            # List available tools
            response = await session.list_tools()
            tools = response.tools

            # Create connection object
            connection = MCPServerConnection(name, config, session)
            connection.tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
                for tool in tools
            ]

            self.logger.info(
                f"Successfully connected to {name}",
                extra={
                    "server": name,
                    "tools": [tool["name"] for tool in connection.tools],
                    "tool_count": len(connection.tools),
                },
            )

            return connection

        except asyncio.TimeoutError:
            error_msg = f"Timeout connecting to server {name}"
            self.logger.error(error_msg)
            raise TimeoutError(error_msg)
        except Exception as e:
            error_msg = f"Failed to connect to server {name}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise ServerConnectionError(error_msg) from e

    async def connect_to_all_servers(self) -> None:
        """Connect to all enabled MCP servers"""
        enabled_servers = self.config.get_enabled_servers()

        if not enabled_servers:
            self.logger.warning("No enabled servers found in configuration")
            return

        self.logger.info(f"Connecting to {len(enabled_servers)} servers...")

        # Connect to servers concurrently
        connection_tasks = []
        for name, config in enabled_servers.items():
            task = asyncio.create_task(
                self.connect_to_server(name, config), name=f"connect_{name}"
            )
            connection_tasks.append((name, task))

        # Wait for all connections with individual error handling
        for name, task in connection_tasks:
            try:
                connection = await task
                self.connections[name] = connection

                # Map tools to servers
                for tool in connection.tools:
                    tool_name = tool["name"]
                    if tool_name in self.tool_to_server:
                        self.logger.warning(
                            f"Tool '{tool_name}' is provided by multiple servers. "
                            f"Using server '{self.tool_to_server[tool_name]}'"
                        )
                    else:
                        self.tool_to_server[tool_name] = name
                        self.available_tools.append(tool)

            except Exception as e:
                self.logger.error(f"Failed to connect to server {name}: {e}")
                # Continue with other servers
                continue

        if not self.connections:
            raise ServerConnectionError("Failed to connect to any MCP servers")

        self.logger.info(
            f"Connected to {len(self.connections)} servers with {len(self.available_tools)} tools total"
        )
        self.is_initialized = True

    async def disconnect_from_server(self, name: str) -> bool:
        """
        Disconnect from a specific server

        Args:
            name: Server name

        Returns:
            True if successfully disconnected, False if server was not connected
        """
        if name not in self.connections:
            return False

        connection = self.connections[name]

        # Remove tools from available tools
        tools_to_remove = [tool["name"] for tool in connection.tools]
        self.available_tools = [
            tool for tool in self.available_tools if tool["name"] not in tools_to_remove
        ]

        # Remove from tool mapping
        for tool_name in tools_to_remove:
            if tool_name in self.tool_to_server:
                del self.tool_to_server[tool_name]

        # Remove connection
        del self.connections[name]

        self.logger.info(f"Disconnected from server: {name}")
        return True

    async def call_tool(
        self, tool_name: str, arguments: Dict[str, Any], timeout: Optional[int] = None
    ) -> Any:
        """
        Call a tool on the appropriate MCP server

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            timeout: Optional timeout override

        Returns:
            Tool execution result

        Raises:
            ToolNotFoundError: If tool is not found
            ServerNotFoundError: If server is not connected
            ToolExecutionError: If tool execution fails
        """
        if tool_name not in self.tool_to_server:
            raise ToolNotFoundError(f"Tool '{tool_name}' not found")

        server_name = self.tool_to_server[tool_name]
        if server_name not in self.connections:
            raise ServerNotFoundError(f"Server '{server_name}' not connected")

        connection = self.connections[server_name]
        timeout = timeout or self.config.global_settings.tool_call_timeout

        self.logger.debug(
            f"Calling tool '{tool_name}' on server '{server_name}'",
            extra={"tool": tool_name, "server": server_name, "arguments": arguments},
        )

        try:
            # Execute tool with timeout
            result = await asyncio.wait_for(
                connection.session.call_tool(tool_name, arguments=arguments),
                timeout=timeout,
            )

            connection.tool_call_count += 1

            self.logger.debug(
                f"Tool '{tool_name}' executed successfully",
                extra={"tool": tool_name, "server": server_name},
            )

            return result.content

        except asyncio.TimeoutError:
            error_msg = f"Tool '{tool_name}' timed out after {timeout} seconds"
            self.logger.error(error_msg)
            connection.error_count += 1
            raise TimeoutError(error_msg)
        except Exception as e:
            error_msg = f"Tool '{tool_name}' execution failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            connection.error_count += 1
            raise ToolExecutionError(error_msg) from e

    async def process_query_with_anthropic(self, query: str) -> str:
        """
        Process a query using Anthropic's Claude

        Args:
            query: User query

        Returns:
            AI response
        """
        messages = [{"role": "user", "content": query}]

        try:
            response = self.ai_client.messages.create(
                max_tokens=self.config.max_tokens,
                model=self.config.default_model,
                tools=self.available_tools,
                messages=messages,
                temperature=self.config.temperature,
            )

            process_query = True
            final_response = ""

            while process_query:
                assistant_content = []

                for content in response.content:
                    if content.type == "text":
                        final_response += content.text
                        assistant_content.append(content)
                        if len(response.content) == 1:
                            process_query = False

                    elif content.type == "tool_use":
                        assistant_content.append(content)
                        messages.append(
                            {"role": "assistant", "content": assistant_content}
                        )

                        tool_id = content.id
                        tool_args = content.input
                        tool_name = content.name

                        self.logger.info(
                            f"AI requesting tool: {tool_name} with args: {tool_args}"
                        )

                        # Call the tool
                        try:
                            result = await self.call_tool(tool_name, tool_args)

                            messages.append(
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "tool_result",
                                            "tool_use_id": tool_id,
                                            "content": result,
                                        }
                                    ],
                                }
                            )

                            # Get next response from AI
                            response = self.ai_client.messages.create(
                                max_tokens=self.config.max_tokens,
                                model=self.config.default_model,
                                tools=self.available_tools,
                                messages=messages,
                                temperature=self.config.temperature,
                            )

                            if (
                                len(response.content) == 1
                                and response.content[0].type == "text"
                            ):
                                final_response += response.content[0].text
                                process_query = False

                        except Exception as e:
                            # Handle tool execution error
                            error_result = f"Error executing tool {tool_name}: {str(e)}"
                            messages.append(
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "tool_result",
                                            "tool_use_id": tool_id,
                                            "content": error_result,
                                            "is_error": True,
                                        }
                                    ],
                                }
                            )

                            # Let AI handle the error
                            response = self.ai_client.messages.create(
                                max_tokens=self.config.max_tokens,
                                model=self.config.default_model,
                                tools=self.available_tools,
                                messages=messages,
                                temperature=self.config.temperature,
                            )

            return final_response

        except Exception as e:
            error_msg = f"AI query processing failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise AIProviderError(error_msg) from e

    async def process_query_with_openai(self, query: str) -> str:
        """
        Process a query using OpenAI's GPT

        Args:
            query: User query

        Returns:
            AI response
        """
        # Convert MCP tools to OpenAI format
        openai_tools = []
        for tool in self.available_tools:
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"],
                    },
                }
            )

        messages = [{"role": "user", "content": query}]

        try:
            response = self.ai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                tools=openai_tools,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            final_response = ""

            while True:
                message = response.choices[0].message

                if message.content:
                    final_response += message.content

                if not message.tool_calls:
                    break

                # Process tool calls
                messages.append(message)

                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    self.logger.info(
                        f"AI requesting tool: {tool_name} with args: {tool_args}"
                    )

                    try:
                        result = await self.call_tool(tool_name, tool_args)

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": str(result),
                            }
                        )

                    except Exception as e:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"Error: {str(e)}",
                            }
                        )

                # Get next response
                response = self.ai_client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=messages,
                    tools=openai_tools,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )

            return final_response

        except Exception as e:
            error_msg = f"AI query processing failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise AIProviderError(error_msg) from e

    def _clean_schema_for_gemini(self, schema: dict) -> dict:
        """
        Clean MCP schema to be compatible with Gemini function calling

        Args:
            schema: MCP tool input schema

        Returns:
            Cleaned schema compatible with Gemini
        """
        if not isinstance(schema, dict):
            return {"type": "object"}

        cleaned = {}

        # Copy allowed fields for Gemini schemas
        allowed_fields = {
            "type", "properties", "required", "description",
            "enum", "items", "minimum", "maximum", "format"
        }

        # Type mapping for Gemini compatibility
        type_mapping = {
            "object": "object",
            "string": "string",
            "number": "number",
            "integer": "integer",
            "boolean": "boolean",
            "array": "array",
            "null": "string"  # Gemini doesn't support null, map to string
        }

        for key, value in schema.items():
            if key in allowed_fields:
                if key == "type":
                    # Map types to Gemini-compatible types
                    if isinstance(value, str):
                        cleaned[key] = type_mapping.get(value, "string")
                    elif isinstance(value, list):
                        # Handle multiple types by picking the first supported one
                        for type_val in value:
                            mapped_type = type_mapping.get(type_val)
                            if mapped_type:
                                cleaned[key] = mapped_type
                                break
                        else:
                            cleaned[key] = "string"  # Default fallback
                    else:
                        cleaned[key] = "string"  # Default fallback
                elif key == "properties" and isinstance(value, dict):
                    # Recursively clean properties
                    cleaned[key] = {
                        prop_name: self._clean_schema_for_gemini(prop_schema)
                        for prop_name, prop_schema in value.items()
                    }
                elif key == "items" and isinstance(value, dict):
                    # Clean array item schemas
                    cleaned[key] = self._clean_schema_for_gemini(value)
                else:
                    cleaned[key] = value

        # Ensure we always have a type
        if "type" not in cleaned:
            cleaned["type"] = "object"

        return cleaned

    async def process_query_with_gemini(self, query: str) -> str:
        """
        Process a query using Google Gemini (following Anthropic/OpenAI pattern)

        Args:
            query: User query

        Returns:
            AI response
        """
        try:
            model = self.ai_client

            # Convert MCP tools to Gemini function calling format
            function_declarations = []
            for tool in self.available_tools:
                # Clean the schema to remove unsupported fields
                clean_schema = self._clean_schema_for_gemini(tool["input_schema"])

                function_declarations.append({
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": clean_schema
                })

            # Start a chat session for multi-turn conversation
            if function_declarations:
                tools = [{"function_declarations": function_declarations}]
                chat = model.start_chat()

                # First response from Gemini
                response = chat.send_message(
                    query,
                    tools=tools,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.config.temperature,
                        max_output_tokens=self.config.max_tokens,
                    )
                )

                final_response = ""

                # Loop for tool execution (following OpenAI/Anthropic pattern)
                while True:
                    # Collect text content
                    if response.candidates and response.candidates[0].content.parts:
                        for part in response.candidates[0].content.parts:
                            if hasattr(part, 'text') and part.text:
                                final_response += part.text

                    # Check for function calls
                    function_calls = []
                    if response.candidates and response.candidates[0].content.parts:
                        for part in response.candidates[0].content.parts:
                            if hasattr(part, 'function_call') and part.function_call:
                                function_calls.append(part.function_call)

                    if not function_calls:
                        # No more function calls, we're done
                        break

                    # Execute all function calls
                    function_responses = []
                    for function_call in function_calls:
                        tool_name = function_call.name
                        tool_args = dict(function_call.args) if function_call.args else {}

                        self.logger.info(f"Gemini calling tool: {tool_name} with args: {tool_args}")

                        try:
                            # Execute the tool
                            tool_result = await self.call_tool(tool_name, tool_args)
                            self.logger.info(f"Tool {tool_name} executed successfully")

                            function_responses.append({
                                "function_response": {
                                    "name": tool_name,
                                    "response": {"result": str(tool_result)}
                                }
                            })

                        except Exception as tool_error:
                            self.logger.error(f"Tool execution failed for {tool_name}: {tool_error}")
                            function_responses.append({
                                "function_response": {
                                    "name": tool_name,
                                    "response": {"error": str(tool_error)}
                                }
                            })

                    # Send function results back to Gemini for next iteration
                    response = chat.send_message(
                        function_responses,
                        generation_config=genai.types.GenerationConfig(
                            temperature=self.config.temperature,
                            max_output_tokens=self.config.max_tokens,
                        )
                    )

                return final_response

            else:
                # No tools available, regular response
                response = model.generate_content(
                    query,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.config.temperature,
                        max_output_tokens=self.config.max_tokens,
                    )
                )
                return response.text

        except Exception as e:
            error_msg = f"AI query processing failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise AIProviderError(error_msg) from e

    async def process_query(self, query: str) -> str:
        """
        Process a user query using the configured AI provider

        Args:
            query: User query

        Returns:
            AI response
        """
        if not self.is_initialized:
            raise ServerConnectionError(
                "Client not initialized. Call connect_to_all_servers() first."
            )

        self.logger.info(f"Processing query: {query}")

        if self.config.anthropic_api_key:
            return await self.process_query_with_anthropic(query)
        elif self.config.openai_api_key:
            return await self.process_query_with_openai(query)
        elif self.config.gemini_api_key:
            return await self.process_query_with_gemini(query)
        else:
            raise AIProviderError("No AI provider configured")

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all connected servers

        Returns:
            Health status of all servers
        """
        health_status = {
            "timestamp": time.time(),
            "servers": {},
            "total_servers": len(self.connections),
            "healthy_servers": 0,
            "total_tools": len(self.available_tools),
        }

        for name, connection in self.connections.items():
            try:
                # Try to list tools as a health check
                await asyncio.wait_for(connection.session.list_tools(), timeout=5.0)

                connection.is_healthy = True
                connection.last_health_check = time.time()
                health_status["healthy_servers"] += 1

                health_status["servers"][name] = {
                    "status": "healthy",
                    "uptime": time.time() - connection.connected_at,
                    "tool_calls": connection.tool_call_count,
                    "errors": connection.error_count,
                    "last_check": connection.last_health_check,
                }

            except Exception as e:
                connection.is_healthy = False

                health_status["servers"][name] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "uptime": time.time() - connection.connected_at,
                    "tool_calls": connection.tool_call_count,
                    "errors": connection.error_count,
                    "last_check": time.time(),
                }

        return health_status

    def get_server_info(self) -> Dict[str, Any]:
        """Get information about connected servers and available tools"""
        return {
            "connected_servers": list(self.connections.keys()),
            "total_servers": len(self.connections),
            "available_tools": [
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "server": self.tool_to_server[tool["name"]],
                }
                for tool in self.available_tools
            ],
            "total_tools": len(self.available_tools),
            "tool_to_server_mapping": self.tool_to_server.copy(),
        }

    async def cleanup(self) -> None:
        """Clean up all resources"""
        self.logger.info("Cleaning up Global MCP Client...")

        # Close all connections
        await self.exit_stack.aclose()

        # Clear state
        self.connections.clear()
        self.tool_to_server.clear()
        self.available_tools.clear()
        self.is_initialized = False

        self.logger.info("Cleanup completed")

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect_to_all_servers()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()
