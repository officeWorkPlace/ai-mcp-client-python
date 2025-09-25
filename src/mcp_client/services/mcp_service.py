"""
Shared MCP service for all interfaces
Provides a single point of access to MCP functionality
"""

import asyncio
import time
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

# Import from new structure
from ..core import GlobalMCPClient, Config, setup_logging
from ..core.exceptions import GlobalMCPClientError


class MCPService:
    """
    Shared MCP service that can be used by all interfaces
    This ensures consistent behavior across chatbot, REST API, and WebSocket
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the MCP service

        Args:
            config: MCP client configuration
        """
        self.config = config or Config()
        self._client: Optional[GlobalMCPClient] = None
        self._initialized = False
        self._lock = asyncio.Lock()

        # Setup logging
        self.logger = setup_logging(
            log_level=self.config.log_level,
            log_file=self.config.log_file,
            enable_file_logging=self.config.enable_file_logging,
            enable_console_logging=self.config.enable_console_logging,
            development_mode=self.config.development_mode,
        )

        # Statistics
        self.stats = {
            "queries_processed": 0,
            "tools_called": 0,
            "errors": 0,
            "start_time": time.time(),
        }

    @property
    def client(self) -> GlobalMCPClient:
        """Get the MCP client instance"""
        if not self._client:
            raise RuntimeError("MCP service not initialized. Call initialize() first.")
        return self._client

    @property
    def is_initialized(self) -> bool:
        """Check if the service is initialized"""
        return self._initialized

    async def initialize(self) -> bool:
        """
        Initialize the MCP service

        Returns:
            True if successful, False otherwise
        """
        async with self._lock:
            if self._initialized:
                return True

            try:
                self.logger.info("Initializing MCP service")
                self._client = GlobalMCPClient(self.config)
                await self._client.connect_to_all_servers()
                self._initialized = True
                self.logger.info("MCP service initialized successfully")
                return True

            except Exception as e:
                self.logger.error(f"Failed to initialize MCP service: {e}")
                self._initialized = False
                return False

    async def cleanup(self):
        """Cleanup the MCP service"""
        async with self._lock:
            if self._client and self._initialized:
                try:
                    await self._client.cleanup()
                    self.logger.info("MCP service cleaned up")
                except Exception as e:
                    self.logger.error(f"Error during MCP service cleanup: {e}")
                finally:
                    self._client = None
                    self._initialized = False

    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Process a query using the MCP client

        Args:
            query: The query to process
            context: Optional context information

        Returns:
            The response from the MCP client

        Raises:
            RuntimeError: If service not initialized
            GlobalMCPClientError: If query processing fails
        """
        if not self._initialized:
            raise RuntimeError("MCP service not initialized")

        self.stats["queries_processed"] += 1

        try:
            response = await self.client.process_query(query)
            return response

        except GlobalMCPClientError as e:
            self.stats["errors"] += 1
            self.logger.error(f"Query processing failed: {e}")
            raise
        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error(f"Unexpected error processing query: {e}")
            raise GlobalMCPClientError(f"Query processing failed: {str(e)}")

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a specific tool

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool result

        Raises:
            RuntimeError: If service not initialized
            Exception: If tool call fails
        """
        if not self._initialized:
            raise RuntimeError("MCP service not initialized")

        self.stats["tools_called"] += 1

        try:
            result = await self.client.call_tool(tool_name, arguments)
            return result

        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error(f"Tool call failed: {e}")
            raise

    def get_server_info(self) -> Dict[str, Any]:
        """
        Get server information

        Returns:
            Server information dictionary

        Raises:
            RuntimeError: If service not initialized
        """
        if not self._initialized:
            raise RuntimeError("MCP service not initialized")

        return self.client.get_server_info()

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check

        Returns:
            Health status dictionary

        Raises:
            RuntimeError: If service not initialized
        """
        if not self._initialized:
            raise RuntimeError("MCP service not initialized")

        return await self.client.health_check()

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get list of available tools

        Returns:
            List of available tools

        Raises:
            RuntimeError: If service not initialized
        """
        if not self._initialized:
            raise RuntimeError("MCP service not initialized")

        return self.client.available_tools

    def reset_conversation(self):
        """
        Reset conversation history

        Raises:
            RuntimeError: If service not initialized
        """
        if not self._initialized:
            raise RuntimeError("MCP service not initialized")

        if hasattr(self.client, 'reset_conversation'):
            self.client.reset_conversation()

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        uptime = time.time() - self.stats["start_time"]
        return {
            **self.stats,
            "uptime": uptime,
            "success_rate": (
                (self.stats["queries_processed"] - self.stats["errors"])
                / max(1, self.stats["queries_processed"]) * 100
            ),
            "initialized": self._initialized,
        }

    @asynccontextmanager
    async def managed_lifecycle(self):
        """Context manager for automatic lifecycle management"""
        try:
            await self.initialize()
            yield self
        finally:
            await self.cleanup()