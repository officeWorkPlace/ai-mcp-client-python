"""
Interface coordinator for managing multiple MCP client interfaces
Coordinates between chatbot, REST API, and WebSocket interfaces
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any
from enum import Enum

# Import from new structure
from ..core import Config, setup_logging

from ..interfaces import MCPChatBot, MCPRestAPI, MCPWebSocketServer
from .mcp_service import MCPService


class InterfaceType(Enum):
    """Available interface types"""
    CHATBOT = "chatbot"
    REST_API = "rest_api"
    WEBSOCKET = "websocket"


class InterfaceCoordinator:
    """
    Coordinates multiple MCP client interfaces
    Allows running chatbot, REST API, and WebSocket simultaneously
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the interface coordinator

        Args:
            config: MCP client configuration
        """
        self.config = config or Config()
        self.mcp_service = MCPService(self.config)

        # Interface instances
        self.chatbot: Optional[MCPChatBot] = None
        self.rest_api: Optional[MCPRestAPI] = None
        self.websocket: Optional[MCPWebSocketServer] = None

        # Configuration
        self.api_host = "127.0.0.1"
        self.api_port = 8000
        self.ws_host = "localhost"
        self.ws_port = 8765

        # State
        self.active_interfaces: List[InterfaceType] = []
        self.tasks: List[asyncio.Task] = []

        # Setup logging
        self.logger = setup_logging(
            log_level=self.config.log_level,
            log_file=self.config.log_file,
            enable_file_logging=self.config.enable_file_logging,
            enable_console_logging=self.config.enable_console_logging,
            development_mode=self.config.development_mode,
        )

    def configure_rest_api(self, host: str = "127.0.0.1", port: int = 8000):
        """Configure REST API settings"""
        self.api_host = host
        self.api_port = port

    def configure_websocket(self, host: str = "localhost", port: int = 8765):
        """Configure WebSocket settings"""
        self.ws_host = host
        self.ws_port = port

    async def start_interface(self, interface_type: InterfaceType) -> bool:
        """
        Start a specific interface

        Args:
            interface_type: Type of interface to start

        Returns:
            True if started successfully, False otherwise
        """
        try:
            if interface_type == InterfaceType.CHATBOT:
                return await self._start_chatbot()
            elif interface_type == InterfaceType.REST_API:
                return await self._start_rest_api()
            elif interface_type == InterfaceType.WEBSOCKET:
                return await self._start_websocket()
            else:
                self.logger.error(f"Unknown interface type: {interface_type}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to start {interface_type.value}: {e}")
            return False

    async def _start_chatbot(self) -> bool:
        """Start the chatbot interface"""
        if InterfaceType.CHATBOT in self.active_interfaces:
            self.logger.warning("Chatbot interface already active")
            return True

        # Initialize MCP service if not already done
        if not self.mcp_service.is_initialized:
            if not await self.mcp_service.initialize():
                return False

        # Create chatbot with shared MCP service
        self.chatbot = MCPChatBot(self.config)

        # Replace chatbot's client with our shared service
        self.chatbot.client = self.mcp_service.client

        # Start chatbot in a task
        task = asyncio.create_task(self.chatbot.run())
        self.tasks.append(task)
        self.active_interfaces.append(InterfaceType.CHATBOT)

        self.logger.info("Chatbot interface started")
        return True

    async def _start_rest_api(self) -> bool:
        """Start the REST API interface"""
        if InterfaceType.REST_API in self.active_interfaces:
            self.logger.warning("REST API interface already active")
            return True

        # Initialize MCP service if not already done
        if not self.mcp_service.is_initialized:
            if not await self.mcp_service.initialize():
                return False

        # Create REST API with shared MCP service
        self.rest_api = MCPRestAPI(self.config, self.api_host, self.api_port)

        # Replace REST API's client with our shared service
        self.rest_api.mcp_client = self.mcp_service.client

        # Start REST API in a task
        def run_api():
            self.rest_api.run()

        task = asyncio.create_task(asyncio.to_thread(run_api))
        self.tasks.append(task)
        self.active_interfaces.append(InterfaceType.REST_API)

        self.logger.info(f"REST API interface started on {self.api_host}:{self.api_port}")
        return True

    async def _start_websocket(self) -> bool:
        """Start the WebSocket interface"""
        if InterfaceType.WEBSOCKET in self.active_interfaces:
            self.logger.warning("WebSocket interface already active")
            return True

        # Initialize MCP service if not already done
        if not self.mcp_service.is_initialized:
            if not await self.mcp_service.initialize():
                return False

        # Create WebSocket server with shared MCP service
        self.websocket = MCPWebSocketServer(self.config, self.ws_host, self.ws_port)

        # Replace WebSocket's client with our shared service
        self.websocket.mcp_client = self.mcp_service.client

        # Start WebSocket server in a task
        task = asyncio.create_task(self.websocket.run_forever())
        self.tasks.append(task)
        self.active_interfaces.append(InterfaceType.WEBSOCKET)

        self.logger.info(f"WebSocket interface started on ws://{self.ws_host}:{self.ws_port}")
        return True

    async def start_all_interfaces(self) -> Dict[InterfaceType, bool]:
        """
        Start all interfaces

        Returns:
            Dictionary mapping interface types to success status
        """
        results = {}

        for interface_type in InterfaceType:
            results[interface_type] = await self.start_interface(interface_type)

        return results

    async def start_interfaces(self, interface_types: List[InterfaceType]) -> Dict[InterfaceType, bool]:
        """
        Start specific interfaces

        Args:
            interface_types: List of interface types to start

        Returns:
            Dictionary mapping interface types to success status
        """
        results = {}

        for interface_type in interface_types:
            results[interface_type] = await self.start_interface(interface_type)

        return results

    async def stop_all_interfaces(self):
        """Stop all active interfaces"""
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

        # Cleanup individual interfaces
        if self.websocket:
            await self.websocket.stop()

        # Cleanup shared MCP service
        await self.mcp_service.cleanup()

        # Clear state
        self.tasks.clear()
        self.active_interfaces.clear()

        self.logger.info("All interfaces stopped")

    async def run_interactive_selection(self):
        """
        Run interactive interface selection
        Allows user to choose which interfaces to start
        """
        print("MCP Client Interface Coordinator")
        print("=" * 40)
        print()
        print("Available interfaces:")
        print("1. Chatbot (Interactive terminal interface)")
        print("2. REST API (HTTP endpoints)")
        print("3. WebSocket (Real-time communication)")
        print("4. All interfaces")
        print()

        try:
            choice = input("Select interfaces to start (1-4): ").strip()

            if choice == "1":
                await self.start_interface(InterfaceType.CHATBOT)
            elif choice == "2":
                await self.start_interface(InterfaceType.REST_API)
            elif choice == "3":
                await self.start_interface(InterfaceType.WEBSOCKET)
            elif choice == "4":
                await self.start_all_interfaces()
            else:
                print("Invalid choice. Starting chatbot interface by default.")
                await self.start_interface(InterfaceType.CHATBOT)

            # Wait for all tasks to complete
            if self.tasks:
                await asyncio.gather(*self.tasks, return_exceptions=True)

        except KeyboardInterrupt:
            print("\nShutdown requested. Stopping all interfaces...")
            await self.stop_all_interfaces()

    async def run_with_interfaces(self, interface_types: List[InterfaceType]):
        """
        Run the coordinator with specific interfaces

        Args:
            interface_types: List of interface types to start
        """
        try:
            results = await self.start_interfaces(interface_types)

            # Log results
            for interface_type, success in results.items():
                if success:
                    self.logger.info(f"{interface_type.value} started successfully")
                else:
                    self.logger.error(f"Failed to start {interface_type.value}")

            # Wait for all tasks to complete
            if self.tasks:
                await asyncio.gather(*self.tasks, return_exceptions=True)

        except KeyboardInterrupt:
            self.logger.info("Shutdown requested. Stopping all interfaces...")
            await self.stop_all_interfaces()
        except Exception as e:
            self.logger.error(f"Error running interfaces: {e}")
            await self.stop_all_interfaces()

    def get_status(self) -> Dict[str, Any]:
        """Get coordinator status"""
        return {
            "active_interfaces": [itype.value for itype in self.active_interfaces],
            "mcp_service_initialized": self.mcp_service.is_initialized,
            "mcp_service_stats": self.mcp_service.get_stats(),
            "running_tasks": len(self.tasks),
            "api_config": {"host": self.api_host, "port": self.api_port},
            "websocket_config": {"host": self.ws_host, "port": self.ws_port},
        }


async def main():
    """Main function for running the interface coordinator"""
    import argparse

    parser = argparse.ArgumentParser(description="MCP Client Interface Coordinator")
    parser.add_argument(
        "--interfaces",
        nargs="+",
        choices=["chatbot", "rest_api", "websocket", "all"],
        default=["chatbot"],
        help="Interfaces to start"
    )
    parser.add_argument("--api-host", default="127.0.0.1", help="REST API host")
    parser.add_argument("--api-port", type=int, default=8000, help="REST API port")
    parser.add_argument("--ws-host", default="localhost", help="WebSocket host")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket port")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Configure
    config = Config()
    if args.debug:
        config.log_level = "DEBUG"
        config.development_mode = True

    coordinator = InterfaceCoordinator(config)
    coordinator.configure_rest_api(args.api_host, args.api_port)
    coordinator.configure_websocket(args.ws_host, args.ws_port)

    # Determine which interfaces to start
    if "all" in args.interfaces:
        interface_types = list(InterfaceType)
    else:
        interface_types = [InterfaceType(iface) for iface in args.interfaces]

    # Run coordinator
    await coordinator.run_with_interfaces(interface_types)


if __name__ == "__main__":
    asyncio.run(main())