"""
Core MCP client implementation for connecting to multiple MCP servers
"""

import asyncio
import json
import time
import warnings
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

# Suppress the specific protobuf warning for unrecognized FinishReason enum values
warnings.filterwarnings("ignore", message="Unrecognized FinishReason enum value.*", category=UserWarning)

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

# Enhanced AI capabilities imports
try:
    from global_mcp_client.enhancements.context.context_manager import IntelligentContextManager
    from global_mcp_client.enhancements.reasoning.cot_engine import ChainOfThoughtEngine
    from global_mcp_client.enhancements.quality.response_optimizer import ResponseQualityOptimizer
    from global_mcp_client.enhancements.monitoring.performance_tracker import PerformanceTracker
    ENHANCEMENTS_AVAILABLE = True
except ImportError as e:
    # Graceful degradation if enhancements are not available
    IntelligentContextManager = None
    ChainOfThoughtEngine = None
    ResponseQualityOptimizer = None
    PerformanceTracker = None
    ENHANCEMENTS_AVAILABLE = False


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

        # Initialize Enhancement Components
        self._initialize_enhancements()

        self.logger.info("Global MCP Client initialized successfully")

    def _initialize_enhancements(self) -> None:
        """Initialize enhancement components based on configuration"""
        self.enhancement_components = {}

        if not ENHANCEMENTS_AVAILABLE:
            self.logger.warning("AI enhancements are not available - running in basic mode")
            return

        # Initialize Intelligent Context Manager
        if self.config.enable_intelligent_context and IntelligentContextManager:
            try:
                self.enhancement_components['context_manager'] = IntelligentContextManager(self.config)
                self.logger.info("IntelligentContextManager initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize IntelligentContextManager: {e}")

        # Initialize Chain-of-Thought Engine
        if self.config.enable_chain_of_thought and ChainOfThoughtEngine:
            try:
                self.enhancement_components['cot_engine'] = ChainOfThoughtEngine(self.config)
                self.logger.info("ChainOfThoughtEngine initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize ChainOfThoughtEngine: {e}")

        # Initialize Response Quality Optimizer
        if self.config.enable_quality_optimization and ResponseQualityOptimizer:
            try:
                self.enhancement_components['quality_optimizer'] = ResponseQualityOptimizer(self.config)
                self.logger.info("ResponseQualityOptimizer initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize ResponseQualityOptimizer: {e}")

        # Initialize Performance Tracker
        if self.config.enable_performance_tracking and PerformanceTracker:
            try:
                self.enhancement_components['performance_tracker'] = PerformanceTracker(self.config)
                self.logger.info("PerformanceTracker initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize PerformanceTracker: {e}")

        self.logger.info(f"Enhancement components initialized: {list(self.enhancement_components.keys())}")

    async def _enhance_query_processing(self, query: str, provider: str) -> Tuple[str, Dict[str, Any]]:
        """
        Enhanced query processing with context optimization and reasoning

        Args:
            query: User query
            provider: AI provider being used

        Returns:
            Tuple of (enhanced_query, processing_context)
        """
        processing_context = {
            "original_query": query,
            "provider": provider,
            "available_tools": self.available_tools,
            "enhancement_used": False,
            "reasoning_result": None,
            "context_optimization": None
        }

        enhanced_query = query

        # Step 1: Context Management Enhancement
        context_manager = self.enhancement_components.get('context_manager')
        if context_manager:
            context_start_time = time.time()
            try:
                # Add current message to context
                context_manager.add_message("user", query)

                # Optimize context for this query
                context_optimization = context_manager.optimize_context_for_query(query, self.available_tools)
                processing_context["context_optimization"] = context_optimization
                processing_context["enhancement_used"] = True

                # Update available tools based on optimization
                if context_optimization.priority_tools:
                    processing_context["available_tools"] = context_optimization.priority_tools

                # Track performance
                context_execution_time = time.time() - context_start_time
                performance_tracker = self.enhancement_components.get('performance_tracker')
                if performance_tracker:
                    performance_tracker.track_context_optimization(context_optimization, context_execution_time)

                self.logger.debug("Context optimized for query", extra={
                    "compression_ratio": context_optimization.utilization_stats.get("compression_ratio", 0),
                    "optimized_message_count": len(context_optimization.optimized_messages),
                    "execution_time": context_execution_time
                })

            except Exception as e:
                context_execution_time = time.time() - context_start_time
                performance_tracker = self.enhancement_components.get('performance_tracker')
                if performance_tracker:
                    performance_tracker.track_operation("context_manager", "optimize_context", context_execution_time, success=False)
                self.logger.error(f"Context optimization failed: {e}")

        # Step 2: Chain-of-Thought Enhancement
        cot_engine = self.enhancement_components.get('cot_engine')
        if cot_engine:
            reasoning_start_time = time.time()
            try:
                enhanced_query, reasoning_result = cot_engine.enhance_query_with_reasoning(query, processing_context)
                processing_context["reasoning_result"] = reasoning_result
                processing_context["enhancement_used"] = True

                # Track performance
                reasoning_execution_time = time.time() - reasoning_start_time
                performance_tracker = self.enhancement_components.get('performance_tracker')
                if performance_tracker:
                    performance_tracker.track_reasoning_enhancement(reasoning_result, reasoning_execution_time)

                self.logger.debug("Query enhanced with reasoning", extra={
                    "reasoning_type": reasoning_result.reasoning_type.value,
                    "original_length": len(query),
                    "enhanced_length": len(enhanced_query),
                    "execution_time": reasoning_execution_time
                })

            except Exception as e:
                reasoning_execution_time = time.time() - reasoning_start_time
                performance_tracker = self.enhancement_components.get('performance_tracker')
                if performance_tracker:
                    performance_tracker.track_operation("cot_engine", "enhance_reasoning", reasoning_execution_time, success=False)
                self.logger.error(f"Chain-of-thought enhancement failed: {e}")

        return enhanced_query, processing_context

    async def _enhance_response_quality(self, response: str, query: str, context: Dict[str, Any]) -> str:
        """
        Enhance response quality using the quality optimizer

        Args:
            response: Original AI response
            query: Original user query
            context: Processing context from query enhancement

        Returns:
            Enhanced response
        """
        quality_optimizer = self.enhancement_components.get('quality_optimizer')
        if not quality_optimizer:
            return response

        quality_start_time = time.time()
        try:
            # Prepare context for quality optimization
            quality_context = {
                "tool_results": context.get("tool_results", []),
                "reasoning_trace": context.get("reasoning_result"),
                "context_optimization": context.get("context_optimization"),
                "available_tools": context.get("available_tools", [])
            }

            # Optimize response quality
            enhancement_result = quality_optimizer.optimize_response(response, query, quality_context)

            # Extract reasoning from response if we have a reasoning result
            if context.get("reasoning_result") and enhancement_result.enhanced_response:
                cot_engine = self.enhancement_components.get('cot_engine')
                if cot_engine:
                    context["reasoning_result"] = cot_engine.extract_reasoning_from_response(
                        enhancement_result.enhanced_response,
                        context["reasoning_result"]
                    )

            # Track performance
            quality_execution_time = time.time() - quality_start_time
            performance_tracker = self.enhancement_components.get('performance_tracker')
            if performance_tracker:
                performance_tracker.track_quality_optimization(enhancement_result, quality_execution_time)

            self.logger.info("Response quality enhanced", extra={
                "before_score": enhancement_result.before_assessment.overall_score,
                "after_score": enhancement_result.after_assessment.overall_score,
                "improvements_made": len(enhancement_result.quality_improvements),
                "execution_time": quality_execution_time
            })

            return enhancement_result.enhanced_response

        except Exception as e:
            quality_execution_time = time.time() - quality_start_time
            performance_tracker = self.enhancement_components.get('performance_tracker')
            if performance_tracker:
                performance_tracker.track_operation("quality_optimizer", "optimize_response", quality_execution_time, success=False)
            self.logger.error(f"Response quality enhancement failed: {e}")
            return response

    def _update_conversation_context(self, query: str, response: str, tool_calls: List = None, tool_results: List = None) -> None:
        """Update conversation context with enhanced tracking"""
        # Add to basic conversation context (maintain backward compatibility)
        self._conversation_context.append({
            "role": "user",
            "content": query,
            "timestamp": time.time()
        })

        self._conversation_context.append({
            "role": "assistant",
            "content": response,
            "timestamp": time.time(),
            "tool_calls": tool_calls or [],
            "tool_results": tool_results or []
        })

        # Enhanced context tracking
        context_manager = self.enhancement_components.get('context_manager')
        if context_manager:
            try:
                context_manager.add_message("assistant", response, tool_calls, tool_results)
            except Exception as e:
                self.logger.error(f"Enhanced context tracking failed: {e}")

    def get_performance_report(self, time_period: str = "session") -> Optional[Dict[str, Any]]:
        """
        Get comprehensive performance report from the performance tracker

        Args:
            time_period: Time period for the report ("session", "hour", "day")

        Returns:
            Performance report dictionary or None if tracking is disabled
        """
        performance_tracker = self.enhancement_components.get('performance_tracker')
        if not performance_tracker:
            return None

        try:
            report = performance_tracker.generate_performance_report(time_period)
            return {
                "report_timestamp": report.report_timestamp,
                "time_period": report.time_period,
                "overall_performance": report.overall_performance,
                "component_performances": {
                    name: {
                        "component_name": perf.component_name,
                        "total_operations": perf.total_operations,
                        "success_rate": perf.success_rate,
                        "average_execution_time": perf.average_execution_time,
                        "average_quality_score": perf.average_quality_score,
                        "error_rate": perf.error_rate,
                        "efficiency_score": perf.efficiency_score
                    }
                    for name, perf in report.component_performances.items()
                },
                "trends": report.trends,
                "recommendations": report.recommendations,
                "alerts": report.alerts
            }
        except Exception as e:
            self.logger.error(f"Failed to generate performance report: {e}")
            return None

    def get_performance_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get quick performance summary

        Returns:
            Performance summary dictionary or None if tracking is disabled
        """
        performance_tracker = self.enhancement_components.get('performance_tracker')
        if not performance_tracker:
            return None

        try:
            return performance_tracker.get_performance_summary()
        except Exception as e:
            self.logger.error(f"Failed to get performance summary: {e}")
            return None

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
                # Use newer Gemini 2.5 models (Gemini 1.5 was retired Sept 24, 2025)
                model_options = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"]
                model_name = self.config.default_model if hasattr(self.config, 'default_model') else model_options[0]
                self._ai_client = genai.GenerativeModel(model_name)
                self.logger.info("Using Google Gemini AI client")
            else:
                raise AIProviderError("No AI API key configured")

        return self._ai_client

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
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

            # Create session with timeout and retry logic
            try:
                session = await asyncio.wait_for(
                    self.exit_stack.enter_async_context(ClientSession(read, write)),
                    timeout=self.config.global_settings.connection_timeout,
                )

                # Initialize session with longer timeout for initialization
                await asyncio.wait_for(
                    session.initialize(),
                    timeout=max(self.config.global_settings.initialization_timeout, 30),
                )
            except asyncio.TimeoutError as e:
                self.logger.error(f"Timeout during session setup for {name}: {e}")
                raise TimeoutError(f"Session setup timeout for {name}") from e

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

        # Connect to servers concurrently with staggered startup
        connection_tasks = []
        for i, (name, config) in enumerate(enabled_servers.items()):
            # Stagger server connections to reduce race conditions
            async def delayed_connect(server_name, server_config, delay):
                if delay > 0:
                    await asyncio.sleep(delay)
                return await self.connect_to_server(server_name, server_config)

            task = asyncio.create_task(
                delayed_connect(name, config, i * 0.5), name=f"connect_{name}"
            )
            connection_tasks.append((name, task))

        # Wait for all connections with individual error handling and timeout
        for name, task in connection_tasks:
            try:
                connection = await asyncio.wait_for(task, timeout=60.0)
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
        Process a query using Anthropic's Claude with enhanced AI capabilities

        Args:
            query: User query

        Returns:
            AI response
        """
        # Enhanced query processing
        enhanced_query, processing_context = await self._enhance_query_processing(query, "anthropic")

        # Use optimized tools if available
        tools_to_use = processing_context.get("available_tools", self.available_tools)

        messages = [{"role": "user", "content": enhanced_query}]

        try:
            response = self.ai_client.messages.create(
                max_tokens=self.config.max_tokens,
                model=self.config.default_model,
                tools=tools_to_use,
                messages=messages,
                temperature=self.config.temperature,
            )

            process_query = True
            final_response = ""
            tool_calls_made = []
            tool_results_obtained = []

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

                            # Track tool usage for enhancements
                            tool_calls_made.append({
                                "tool_name": tool_name,
                                "tool_args": tool_args,
                                "tool_id": tool_id
                            })
                            tool_results_obtained.append({
                                "tool_name": tool_name,
                                "result": result,
                                "tool_id": tool_id
                            })

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
                                tools=tools_to_use,
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
                                tools=tools_to_use,
                                messages=messages,
                                temperature=self.config.temperature,
                            )

            # Enhanced response quality optimization
            processing_context["tool_results"] = tool_results_obtained
            enhanced_response = await self._enhance_response_quality(final_response, query, processing_context)

            # Update conversation context with enhanced tracking
            self._update_conversation_context(query, enhanced_response, tool_calls_made, tool_results_obtained)

            return enhanced_response

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
        Process a query using Google Gemini with enhanced AI capabilities

        Args:
            query: User query

        Returns:
            AI response
        """
        # Enhanced query processing
        enhanced_query, processing_context = await self._enhance_query_processing(query, "gemini")

        # Use optimized tools if available
        tools_to_use = processing_context.get("available_tools", self.available_tools)
        # Filter tools to avoid overwhelming Gemini (max 50 tools)
        if len(tools_to_use) > 50:
            tools_to_use = tools_to_use[:50]

        try:
            model = self.ai_client

            # Convert MCP tools to Gemini function calling format with filtering
            function_declarations = []
            
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
                    # Send enhanced user query with conversation context
                    response = self._gemini_chat.send_message(
                        enhanced_query,
                        tools=tools,
                        generation_config=genai.types.GenerationConfig(
                            temperature=self.config.temperature,
                            max_output_tokens=self.config.max_tokens,
                        )
                    )
                except Exception as e:
                    error_msg = str(e)
                    self.logger.warning(f"Function calling failed, falling back to text-only: {error_msg}")

                    # Check if it's an unrecognized finish_reason error (e.g., finish_reason: 12)
                    if ("Unrecognized FinishReason enum value" in error_msg or
                        "finish_reason:" in error_msg):
                        self.logger.info("Detected unrecognized finish_reason, falling back to text-only mode")
                        self._gemini_chat = None  # Reset session completely

                        # Immediate fallback to simple generation without function calling
                        try:
                            context_query = f"{system_context}\n\nUser Query: {enhanced_query}"
                            response = model.generate_content(
                                context_query,
                                generation_config=genai.types.GenerationConfig(
                                    temperature=self.config.temperature,
                                    max_output_tokens=self.config.max_tokens,
                                )
                            )
                            return response.text
                        except Exception as fallback_error:
                            self.logger.error(f"Text-only fallback also failed: {fallback_error}")
                            return f"I apologize, but I'm experiencing technical difficulties with the AI service. Error details: {str(fallback_error)}"

                    # Check if it's a malformed function call - if so, reset and retry with simpler tools
                    elif "MALFORMED_FUNCTION_CALL" in error_msg or "INVALID_ARGUMENT" in error_msg:
                        self.logger.info("Detected function call issue, implementing progressive fallback")
                        self._gemini_chat = None

                        # Progressive fallback: try fewer tools, then simpler tools, then no tools
                        fallback_attempts = [
                            function_declarations[:5],   # Try with 5 tools
                            function_declarations[:2],   # Try with 2 tools
                            []                           # Try with no tools
                        ]

                        for attempt, tools_subset in enumerate(fallback_attempts):
                            try:
                                self._gemini_chat = model.start_chat()
                                if tools_subset:
                                    response = self._gemini_chat.send_message(
                                        f"{system_context}\n\nUser Query: {enhanced_query}",
                                        tools=[{"function_declarations": tools_subset}],
                                        generation_config=genai.types.GenerationConfig(
                                            temperature=self.config.temperature,
                                            max_output_tokens=self.config.max_tokens,
                                        )
                                    )
                                else:
                                    # Final fallback: no tools at all
                                    response = self._gemini_chat.send_message(
                                        f"{system_context}\n\nUser Query: {enhanced_query}",
                                        generation_config=genai.types.GenerationConfig(
                                            temperature=self.config.temperature,
                                            max_output_tokens=self.config.max_tokens,
                                        )
                                    )

                                self.logger.info(f"Fallback attempt {attempt + 1} succeeded")
                                break
                            except Exception as retry_error:
                                self.logger.warning(f"Fallback attempt {attempt + 1} failed: {retry_error}")
                                if attempt == len(fallback_attempts) - 1:
                                    # Final fallback failed, use direct generation
                                    self._gemini_chat = None
                                    context_query = f"{system_context}\n\nUser Query: {enhanced_query}"
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
                        context_query = f"{system_context}\n\nUser Query: {enhanced_query}"
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
                    try:
                        response = self._gemini_chat.send_message(
                            function_responses,
                            generation_config=genai.types.GenerationConfig(
                                temperature=self.config.temperature,
                                max_output_tokens=self.config.max_tokens,
                            )
                        )
                    except Exception as e:
                        error_msg = str(e)
                        self.logger.warning(f"Function response failed: {error_msg}")

                        # Check for finish_reason error in function responses
                        if ("Unrecognized FinishReason enum value" in error_msg or
                            "finish_reason:" in error_msg):
                            self.logger.info("finish_reason error in function response, returning current results")
                            break  # Exit the loop and return what we have so far
                        else:
                            # Other errors, also break but log differently
                            self.logger.warning(f"Breaking function call loop due to: {error_msg}")
                            break

                # Track tool usage for enhancements
                tool_calls_made = []
                tool_results_obtained = []
                for part in response.candidates[0].content.parts if response.candidates else []:
                    if hasattr(part, 'function_call') and part.function_call:
                        tool_calls_made.append({
                            "tool_name": part.function_call.name,
                            "tool_args": dict(part.function_call.args) if part.function_call.args else {},
                            "tool_id": getattr(part.function_call, 'id', 'gemini_call')
                        })

                # Enhanced response quality optimization
                processing_context["tool_results"] = tool_results_obtained
                enhanced_response = await self._enhance_response_quality(final_response, query, processing_context)

                # Update conversation context with enhanced tracking
                self._update_conversation_context(query, enhanced_response, tool_calls_made, tool_results_obtained)

                return enhanced_response

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
                        enhanced_query,
                        generation_config=genai.types.GenerationConfig(
                            temperature=self.config.temperature,
                            max_output_tokens=self.config.max_tokens,
                        )
                    )

                    # Enhanced response quality optimization (no tools case)
                    processing_context["tool_results"] = []
                    enhanced_response = await self._enhance_response_quality(response.text, query, processing_context)

                    # Update conversation context with enhanced tracking
                    self._update_conversation_context(query, enhanced_response, [], [])

                    return enhanced_response
                except Exception as e:
                    self.logger.warning(f"Chat session failed, using direct generation: {e}")
                    # Final fallback
                    server_info = self.get_server_info()
                    context_query = f"""
You are an AI assistant integrated with a Model Context Protocol (MCP) client.
Currently connected to {server_info['total_servers']} MCP servers: {', '.join(server_info['connected_servers'])}
No tools are currently available.

User Query: {enhanced_query}
"""
                    response = model.generate_content(
                        context_query,
                        generation_config=genai.types.GenerationConfig(
                            temperature=self.config.temperature,
                            max_output_tokens=self.config.max_tokens,
                        )
                    )

                    # Enhanced response quality optimization (fallback case)
                    processing_context["tool_results"] = []
                    enhanced_response = await self._enhance_response_quality(response.text, query, processing_context)

                    # Update conversation context with enhanced tracking
                    self._update_conversation_context(query, enhanced_response, [], [])

                    return enhanced_response

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
        """Clean up all resources with robust error handling and timeouts"""
        self.logger.info("Cleaning up Global MCP Client...")

        # Set cleanup timeout
        cleanup_timeout = 30.0

        try:
            # Close all connections with proper error handling and timeout
            if hasattr(self, 'exit_stack') and self.exit_stack:
                try:
                    await asyncio.wait_for(self.exit_stack.aclose(), timeout=cleanup_timeout)
                except asyncio.TimeoutError:
                    self.logger.warning(f"Exit stack cleanup timed out after {cleanup_timeout}s, using fallback")
                    await self._fallback_cleanup()
                except RuntimeError as e:
                    if "cancel scope" in str(e) or "different task" in str(e):
                        # This is the asyncio context issue - try alternative cleanup
                        self.logger.debug(f"Asyncio context issue during exit_stack cleanup, using fallback: {e}")
                        await self._fallback_cleanup()
                    else:
                        self.logger.error(f"Unexpected RuntimeError during cleanup: {e}")
                        await self._fallback_cleanup()
        except Exception as e:
            # Log cleanup errors but don't crash
            self.logger.warning(f"Error during connection cleanup: {e}")
            await self._fallback_cleanup()

        # Clear state immediately to prevent further issues
        self.connections.clear()
        self.tool_to_server.clear()
        self.available_tools.clear()
        self.is_initialized = False

        # Reset conversation state
        self._gemini_chat = None
        if hasattr(self, '_conversation_context'):
            self._conversation_context.clear()

        self.logger.info("Cleanup completed successfully")

    async def _fallback_cleanup(self) -> None:
        """Fallback cleanup method for when exit_stack fails"""
        self.logger.info("Using fallback cleanup method")

        # Try individual connection cleanup as fallback
        cleanup_tasks = []
        for name, connection in list(self.connections.items()):
            async def cleanup_connection(conn_name, conn):
                try:
                    if hasattr(conn, 'session'):
                        if hasattr(conn.session, 'close'):
                            await asyncio.wait_for(conn.session.close(), timeout=5.0)
                        elif hasattr(conn.session, '__aexit__'):
                            await asyncio.wait_for(conn.session.__aexit__(None, None, None), timeout=5.0)
                except asyncio.TimeoutError:
                    self.logger.warning(f"Connection cleanup timeout for {conn_name}")
                except Exception as conn_error:
                    self.logger.warning(f"Error closing connection {conn_name}: {conn_error}")

            cleanup_tasks.append(cleanup_connection(name, connection))

        if cleanup_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*cleanup_tasks, return_exceptions=True),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                self.logger.warning("Individual connection cleanup timed out")

        # Force close exit_stack without awaiting if it exists
        if hasattr(self, 'exit_stack') and self.exit_stack:
            try:
                # Try to close synchronously if possible
                if hasattr(self.exit_stack, '_exit_stack'):
                    self.exit_stack._exit_stack.clear()
                # Try alternative cleanup methods
                if hasattr(self.exit_stack, 'close') and not asyncio.iscoroutinefunction(self.exit_stack.close):
                    self.exit_stack.close()
            except Exception as e:
                self.logger.debug(f"Could not clear exit stack: {e}")

        self.logger.info("Fallback cleanup completed")

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect_to_all_servers()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()
