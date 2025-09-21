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
        self._gemini_chat = None  # Store Gemini chat session for conversation history
        self._conversation_context = []  # Store conversation history

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
        
        Gemini has strict requirements:
        - Only supports: type, properties, required, description, enum, items
        - No additionalProperties, $ref, anyOf, oneOf, etc.
        - All properties must have explicit types
        - Schemas must be simple and well-structured
        
        Args:
            schema: MCP tool input schema

        Returns:
            Cleaned schema compatible with Gemini
        """
        if not isinstance(schema, dict):
            return {"type": "object", "properties": {}, "required": []}

        cleaned = {}

        # Only keep fields that Gemini supports
        allowed_fields = {"type", "properties", "required", "description", "enum", "items"}
        
        # Type mapping for Gemini compatibility - be more conservative
        type_mapping = {
            "object": "object",
            "string": "string", 
            "number": "number",
            "integer": "integer", 
            "boolean": "boolean",
            "array": "array",
            "null": "string",  # Gemini doesn't support null
            # Map edge cases to safe types
            "any": "string",
            "mixed": "string",
            "unknown": "string"
        }

        # Handle type field
        if "type" in schema:
            schema_type = schema["type"]
            if isinstance(schema_type, str):
                cleaned["type"] = type_mapping.get(schema_type, "string")
            elif isinstance(schema_type, list):
                # Pick first supported type
                for t in schema_type:
                    if t in type_mapping:
                        cleaned["type"] = type_mapping[t]
                        break
                else:
                    cleaned["type"] = "string"
            else:
                cleaned["type"] = "string"
        else:
            cleaned["type"] = "object"

        # Handle properties with extra validation
        if "properties" in schema and isinstance(schema["properties"], dict):
            cleaned["properties"] = {}
            for prop_name, prop_schema in schema["properties"].items():
                # Ensure property name is valid
                if not isinstance(prop_name, str) or not prop_name.strip():
                    continue
                    
                if isinstance(prop_schema, dict):
                    cleaned_prop = self._clean_schema_for_gemini(prop_schema)
                    # Ensure every property has a type
                    if "type" not in cleaned_prop:
                        cleaned_prop["type"] = "string"
                    
                    # Limit nested complexity for Gemini
                    if cleaned_prop.get("type") == "object" and "properties" in cleaned_prop:
                        # Limit nested object properties to avoid complexity
                        nested_props = cleaned_prop["properties"]
                        if len(nested_props) > 10:  # Limit nested properties
                            # Keep only the first 10 properties
                            limited_props = dict(list(nested_props.items())[:10])
                            cleaned_prop["properties"] = limited_props
                    
                    cleaned["properties"][prop_name] = cleaned_prop
                else:
                    # Simple property, just assign string type
                    cleaned["properties"][prop_name] = {"type": "string"}
        elif cleaned["type"] == "object":
            # Object type must have properties
            cleaned["properties"] = {}

        # Handle required fields
        if "required" in schema and isinstance(schema["required"], list):
            cleaned["required"] = [r for r in schema["required"] if isinstance(r, str)]
        elif cleaned["type"] == "object":
            cleaned["required"] = []

        # Handle description
        if "description" in schema and isinstance(schema["description"], str):
            cleaned["description"] = schema["description"][:200]  # Limit length

        # Handle enum
        if "enum" in schema and isinstance(schema["enum"], list):
            cleaned["enum"] = schema["enum"][:10]  # Limit enum values

        # Handle array items
        if "items" in schema and cleaned["type"] == "array":
            if isinstance(schema["items"], dict):
                cleaned["items"] = self._clean_schema_for_gemini(schema["items"])
            else:
                cleaned["items"] = {"type": "string"}

        return cleaned
    
    def _validate_gemini_schema(self, schema: dict) -> bool:
        """
        Validate that a schema is compatible with Gemini function calling
        
        Args:
            schema: Cleaned schema to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(schema, dict):
            return False
            
        # Must have type
        if "type" not in schema:
            return False
            
        # Check for supported types only
        valid_types = {"object", "string", "number", "integer", "boolean", "array"}
        if schema["type"] not in valid_types:
            return False
            
        # If object type, must have properties
        if schema["type"] == "object":
            if "properties" not in schema or not isinstance(schema["properties"], dict):
                return False
                
            # Validate each property recursively
            for prop_name, prop_schema in schema["properties"].items():
                if not isinstance(prop_schema, dict) or "type" not in prop_schema:
                    return False
                if prop_schema["type"] not in valid_types:
                    return False
                    
        # If array type, should have items
        if schema["type"] == "array":
            if "items" not in schema or not isinstance(schema["items"], dict):
                return False
            # Validate array items
            if "type" not in schema["items"] or schema["items"]["type"] not in valid_types:
                return False
                
        # Check for any disallowed fields that might cause issues
        disallowed_fields = {"$ref", "anyOf", "oneOf", "allOf", "additionalProperties", "patternProperties"}
        if any(field in schema for field in disallowed_fields):
            return False
                
        return True
    
    def reset_conversation(self) -> None:
        """Reset the conversation history for Gemini"""
        self._gemini_chat = None
        self._conversation_context.clear()
        self.logger.info("Conversation history reset")
    
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

            # Convert MCP tools to Gemini function calling format with filtering
            function_declarations = []
            
            # Filter tools to avoid overwhelming Gemini (max 50 tools)
            tools_to_use = self.available_tools[:50] if len(self.available_tools) > 50 else self.available_tools
            
            for tool in tools_to_use:
                try:
                    # Clean the schema to remove unsupported fields
                    clean_schema = self._clean_schema_for_gemini(tool["input_schema"])
                    
                    # Validate the cleaned schema
                    if self._validate_gemini_schema(clean_schema):
                        function_declarations.append({
                            "name": tool["name"],
                            "description": tool["description"][:500],  # Limit description length
                            "parameters": clean_schema
                        })
                    else:
                        self.logger.debug(f"Skipping tool {tool['name']} due to invalid schema")
                except Exception as e:
                    self.logger.debug(f"Failed to process tool {tool['name']}: {e}")

            # Build comprehensive context about the MCP environment
            server_info = self.get_server_info()
            system_context = f"""
You are an advanced intelligent AI assistant with access to a powerful Model Context Protocol (MCP) client that connects to multiple specialized servers. You are designed to be proactive, intelligent, and automatically discover what users need without asking for clarification.

CURRENT ENVIRONMENT:
- Connected to {server_info['total_servers']} MCP servers: {', '.join(server_info['connected_servers'])}
- Total available tools: {server_info['total_tools']}

SERVER CAPABILITIES:
"""
            
            # Add server-specific context
            for server_name, connection in self.connections.items():
                tool_names = [tool['name'] for tool in connection.tools]
                system_context += f"\n• {server_name}: {len(connection.tools)} tools - {', '.join(tool_names[:5])}{'...' if len(tool_names) > 5 else ''}"
            
            system_context += f"""

YOUR INTELLIGENT BEHAVIOR:
1. 🧠 BE PROACTIVE: When users ask for analysis or dashboards, automatically discover schemas, tables, and data structure
2. 🔍 AUTO-DISCOVER: Use get_all_tables and analyze_table_structure to understand databases before asking questions
3. 📊 COMPREHENSIVE: For analysis requests, automatically run multiple queries and provide complete insights
4. 🎯 INTELLIGENT: Understand user intent - "analyze X schema" means discover structure + provide insights + create visualizations
5. 🚀 ACTION-ORIENTED: Don't ask for clarification - take intelligent action based on available tools

SPECIFIC INTELLIGENCE FOR DATABASE ANALYSIS:
- When users mention a schema (like "C##loan_schema"), immediately use get_all_tables to discover all tables
- Automatically analyze table structures with analyze_table_structure for key tables
- Run sample queries to understand data patterns and relationships
- Generate comprehensive dashboards with multiple visualizations
- Provide business insights based on discovered data patterns

TOOL USAGE INTELLIGENCE:
- Oracle DB tools: Use for database discovery, analysis, and dashboard generation
- Filesystem tools: Use for file operations and data processing
- Memory tools: Use for storing analysis results and insights

IMPORTANT INSTRUCTIONS:
- NEVER ask for more details when you can discover them yourself using available tools
- ALWAYS be comprehensive - run multiple queries to provide complete analysis
- AUTOMATICALLY discover schemas, tables, and data structure when doing analysis
- SHOW complete tool results - users want to see actual data, not summaries
- GENERATE multiple visualizations when creating dashboards (charts, tables, metrics)
- BE INTELLIGENT about data relationships and business insights
- TAKE ACTION IMMEDIATELY - be proactive, comprehensive, and solution-oriented
"""

            # Start or continue chat session for multi-turn conversation
            if function_declarations:
                tools = [{"function_declarations": function_declarations}]
                
                # Initialize chat session if it doesn't exist
                if self._gemini_chat is None:
                    self._gemini_chat = model.start_chat()
                    # Send system context as first message
                    system_message = f"{system_context}\n\nPlease acknowledge that you understand your role and capabilities."
                    try:
                        self._gemini_chat.send_message(
                            system_message,
                            tools=tools,
                            generation_config=genai.types.GenerationConfig(
                                temperature=self.config.temperature,
                                max_output_tokens=self.config.max_tokens,
                            )
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to send system message: {e}")
                
                try:
                    # Send user query with conversation context
                    response = self._gemini_chat.send_message(
                        query,
                        tools=tools,
                        generation_config=genai.types.GenerationConfig(
                            temperature=self.config.temperature,
                            max_output_tokens=self.config.max_tokens,
                        )
                    )
                except Exception as e:
                    error_msg = str(e)
                    self.logger.warning(f"Function calling failed, falling back to text-only: {error_msg}")
                    
                    # Check if it's a malformed function call - if so, reset and retry with simpler tools
                    if "MALFORMED_FUNCTION_CALL" in error_msg:
                        self.logger.info("Detected malformed function call, resetting session with simpler tools")
                        self._gemini_chat = None
                        
                        # Retry with a smaller, simpler set of tools
                        simple_tools = [{"function_declarations": function_declarations[:10]}]  # Use only first 10 tools
                        try:
                            self._gemini_chat = model.start_chat()
                            response = self._gemini_chat.send_message(
                                f"{system_context}\n\nUser Query: {query}",
                                tools=simple_tools,
                                generation_config=genai.types.GenerationConfig(
                                    temperature=self.config.temperature,
                                    max_output_tokens=self.config.max_tokens,
                                )
                            )
                            # Continue with the normal processing loop
                        except Exception as retry_error:
                            self.logger.warning(f"Retry with simple tools also failed: {retry_error}")
                            # Fall back to text-only
                            self._gemini_chat = None
                            context_query = f"{system_context}\n\nUser Query: {query}"
                            response = model.generate_content(
                                context_query,
                                generation_config=genai.types.GenerationConfig(
                                    temperature=self.config.temperature,
                                    max_output_tokens=self.config.max_tokens,
                                )
                            )
                            return response.text
                    else:
                        # For other errors, fall back to text-only immediately
                        self._gemini_chat = None
                        context_query = f"{system_context}\n\nUser Query: {query}"
                        response = model.generate_content(
                            context_query,
                            generation_config=genai.types.GenerationConfig(
                                temperature=self.config.temperature,
                                max_output_tokens=self.config.max_tokens,
                            )
                        )
                        return response.text

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
                        
                        # Convert Gemini function arguments to MCP-compatible format
                        tool_args = {}
                        if function_call.args:
                            for key, value in function_call.args.items():
                                # Handle different argument types
                                if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                                    # Convert lists/arrays to strings or handle appropriately
                                    if isinstance(value, (list, tuple)):
                                        # Join list elements with commas for database columns
                                        tool_args[key] = ','.join(str(v) for v in value)
                                    else:
                                        tool_args[key] = str(value)
                                else:
                                    tool_args[key] = value

                        self.logger.info(f"Gemini calling tool: {tool_name} with args: {tool_args}")

                        try:
                            # Execute the tool
                            tool_result = await self.call_tool(tool_name, tool_args)
                            self.logger.info(f"Tool {tool_name} executed successfully")

                            # Format the tool result for better display
                            result_str = str(tool_result)
                            
                            # Add the raw tool result to final response for user visibility
                            final_response += f"\n\n**Tool Result ({tool_name}):**\n```json\n{result_str}\n```\n"

                            function_responses.append({
                                "function_response": {
                                    "name": tool_name,
                                    "response": {"result": result_str}
                                }
                            })

                        except Exception as tool_error:
                            self.logger.error(f"Tool execution failed for {tool_name}: {tool_error}")
                            error_msg = str(tool_error)
                            final_response += f"\n\n**Tool Error ({tool_name}):**\n{error_msg}\n"
                            
                            function_responses.append({
                                "function_response": {
                                    "name": tool_name,
                                    "response": {"error": error_msg}
                                }
                            })

                    # Send function results back to Gemini for next iteration
                    response = self._gemini_chat.send_message(
                        function_responses,
                        generation_config=genai.types.GenerationConfig(
                            temperature=self.config.temperature,
                            max_output_tokens=self.config.max_tokens,
                        )
                    )

                return final_response

            else:
                # No tools available, but still provide context
                # For no tools case, we'll still use a simple chat session for consistency
                if self._gemini_chat is None:
                    self._gemini_chat = model.start_chat()
                    server_info = self.get_server_info()
                    system_message = f"""
You are an AI assistant integrated with a Model Context Protocol (MCP) client.
Currently connected to {server_info['total_servers']} MCP servers: {', '.join(server_info['connected_servers'])}
No tools are currently available, but you can still help answer questions about the connected servers.

Please acknowledge that you understand your role.
"""
                    try:
                        self._gemini_chat.send_message(
                            system_message,
                            generation_config=genai.types.GenerationConfig(
                                temperature=self.config.temperature,
                                max_output_tokens=self.config.max_tokens,
                            )
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to send system message: {e}")
                
                try:
                    response = self._gemini_chat.send_message(
                        query,
                        generation_config=genai.types.GenerationConfig(
                            temperature=self.config.temperature,
                            max_output_tokens=self.config.max_tokens,
                        )
                    )
                    return response.text
                except Exception as e:
                    self.logger.warning(f"Chat session failed, using direct generation: {e}")
                    # Final fallback
                    server_info = self.get_server_info()
                    context_query = f"""
You are an AI assistant integrated with a Model Context Protocol (MCP) client.
Currently connected to {server_info['total_servers']} MCP servers: {', '.join(server_info['connected_servers'])}
No tools are currently available.

User Query: {query}
"""
                    response = model.generate_content(
                        context_query,
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

        try:
            # Close all connections with proper error handling
            if hasattr(self, 'exit_stack') and self.exit_stack:
                await self.exit_stack.aclose()
        except Exception as e:
            # Log cleanup errors but don't crash
            self.logger.warning(f"Error during connection cleanup: {e}")
            
            # Try individual connection cleanup as fallback
            for name, connection in self.connections.items():
                try:
                    if hasattr(connection.session, 'close'):
                        await connection.session.close()
                except Exception as conn_error:
                    self.logger.warning(f"Error closing connection {name}: {conn_error}")

        # Clear state
        self.connections.clear()
        self.tool_to_server.clear()
        self.available_tools.clear()
        self.is_initialized = False
        
        # Reset conversation state
        self._gemini_chat = None
        if hasattr(self, '_conversation_context'):
            self._conversation_context.clear()

        self.logger.info("Cleanup completed")

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect_to_all_servers()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()
