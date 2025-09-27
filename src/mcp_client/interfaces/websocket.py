"""
WebSocket interface for MCP Client
Provides real-time communication with MCP services
"""

import asyncio
import json
import logging
import time
import traceback
from typing import Dict, Any, Set, Optional

import websockets
import websockets.server
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

# Import from new structure
from ..core import GlobalMCPClient, Config, setup_logging
from ..core.exceptions import GlobalMCPClientError


class MCPWebSocketServer:
    """
    WebSocket server for real-time MCP communication
    Architecture: Client <-WebSocket-> MCPWebSocketServer <--> MCP Servers
    """

    def __init__(self, config: Optional[Config] = None, host: str = "localhost", port: int = 8765):
        """
        Initialize the WebSocket server

        Args:
            config: MCP client configuration
            host: WebSocket server host
            port: WebSocket server port
        """
        self.config = config or Config()
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.mcp_client: Optional[GlobalMCPClient] = None
        self.server = None
        self.is_running = False

        # Setup logging
        self.logger = setup_logging(
            log_level=self.config.log_level,
            log_file=self.config.log_file,
            enable_file_logging=self.config.enable_file_logging,
            enable_console_logging=self.config.enable_console_logging,
            development_mode=self.config.development_mode,
        )

        self.logger.info("MCP WebSocket Server initialized")

    async def start(self):
        """Start the WebSocket server and MCP client"""
        try:
            # Initialize MCP client
            self.mcp_client = GlobalMCPClient(self.config)
            await self.mcp_client.connect_to_all_servers()

            # Start WebSocket server
            self.server = await websockets.serve(
                self.handle_client,
                self.host,
                self.port,
                logger=self.logger
            )

            self.is_running = True
            self.logger.info(f"MCP WebSocket Server started on ws://{self.host}:{self.port}")

            # Broadcast server status to connected clients
            await self.broadcast_server_status()

        except Exception as e:
            self.logger.error(f"Failed to start WebSocket server: {e}")
            raise

    async def stop(self):
        """Stop the WebSocket server and cleanup"""
        self.is_running = False

        if self.server:
            self.server.close()
            await self.server.wait_closed()

        # Disconnect all clients
        if self.clients:
            await asyncio.gather(
                *[client.close() for client in self.clients],
                return_exceptions=True
            )
        self.clients.clear()

        # Cleanup MCP client
        if self.mcp_client:
            await self.mcp_client.cleanup()

        self.logger.info("MCP WebSocket Server stopped")

    async def handle_client(self, websocket):
        """Handle a new WebSocket client connection"""
        client_addr = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        self.logger.info(f"New client connected: {client_addr}")

        self.clients.add(websocket)

        try:
            # Send welcome message with server info
            await self.send_server_info(websocket)

            # Handle incoming messages
            async for message in websocket:
                try:
                    await self.handle_message(websocket, message)
                except Exception as e:
                    self.logger.error(f"Error handling message from {client_addr}: {e}")
                    await self.send_error(websocket, str(e))

        except (ConnectionClosedError, ConnectionClosedOK):
            self.logger.info(f"Client disconnected: {client_addr}")
        except Exception as e:
            self.logger.error(f"Error with client {client_addr}: {e}")
        finally:
            self.clients.discard(websocket)

    async def handle_message(self, websocket, message: str):
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            message_id = data.get("id", "unknown")
            payload = data.get("payload", {})

            self.logger.debug(f"Received message: {message_type} (id: {message_id})")

            # Route message based on type
            if message_type == "query":
                await self.handle_query(websocket, message_id, payload)
            elif message_type == "get_server_info":
                await self.handle_get_server_info(websocket, message_id)
            elif message_type == "health_check":
                await self.handle_health_check(websocket, message_id)
            elif message_type == "list_tools":
                await self.handle_list_tools(websocket, message_id)
            elif message_type == "call_tool":
                await self.handle_call_tool(websocket, message_id, payload)
            elif message_type == "ping":
                await self.send_response(websocket, message_id, "pong", {"timestamp": time.time()})
            elif message_type == "reset_conversation":
                await self.handle_reset_conversation(websocket, message_id)
            else:
                await self.send_error(websocket, f"Unknown message type: {message_type}", message_id)

        except json.JSONDecodeError as e:
            await self.send_error(websocket, f"Invalid JSON: {e}")
        except Exception as e:
            self.logger.error(f"Error handling message: {e}\n{traceback.format_exc()}")
            await self.send_error(websocket, f"Internal error: {e}")

    async def handle_query(self, websocket, message_id: str, payload: Dict[str, Any]):
        """Handle AI query request"""
        query = payload.get("query")
        if not query:
            await self.send_error(websocket, "Query is required", message_id)
            return

        try:
            # Send processing status
            await self.send_response(websocket, message_id, "processing", {
                "status": "Processing query with AI...",
                "query": query
            })

            # Process query with MCP client
            response = await self.mcp_client.process_query(query)

            # Send response
            await self.send_response(websocket, message_id, "query_response", {
                "query": query,
                "response": response,
                "timestamp": time.time()
            })

        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            await self.send_error(websocket, f"Query failed: {e}", message_id)

    async def handle_get_server_info(self, websocket, message_id: str):
        """Handle get server info request"""
        try:
            info = self.mcp_client.get_server_info()
            await self.send_response(websocket, message_id, "server_info", info)
        except Exception as e:
            await self.send_error(websocket, f"Failed to get server info: {e}", message_id)

    async def handle_health_check(self, websocket, message_id: str):
        """Handle health check request"""
        try:
            health = await self.mcp_client.health_check()
            await self.send_response(websocket, message_id, "health_status", health)
        except Exception as e:
            await self.send_error(websocket, f"Health check failed: {e}", message_id)

    async def handle_list_tools(self, websocket, message_id: str):
        """Handle list tools request"""
        try:
            tools = self.mcp_client.available_tools
            await self.send_response(websocket, message_id, "tools_list", {
                "tools": tools,
                "count": len(tools)
            })
        except Exception as e:
            await self.send_error(websocket, f"Failed to list tools: {e}", message_id)

    async def handle_call_tool(self, websocket, message_id: str, payload: Dict[str, Any]):
        """Handle direct tool call request"""
        tool_name = payload.get("tool_name")
        arguments = payload.get("arguments", {})

        if not tool_name:
            await self.send_error(websocket, "tool_name is required", message_id)
            return

        try:
            result = await self.mcp_client.call_tool(tool_name, arguments)
            await self.send_response(websocket, message_id, "tool_result", {
                "tool_name": tool_name,
                "arguments": arguments,
                "result": result,
                "timestamp": time.time()
            })
        except Exception as e:
            await self.send_error(websocket, f"Tool call failed: {e}", message_id)

    async def handle_reset_conversation(self, websocket, message_id: str):
        """Handle conversation reset request"""
        try:
            if hasattr(self.mcp_client, 'reset_conversation'):
                self.mcp_client.reset_conversation()
            await self.send_response(websocket, message_id, "conversation_reset", {
                "status": "Conversation history cleared",
                "timestamp": time.time()
            })
        except Exception as e:
            await self.send_error(websocket, f"Failed to reset conversation: {e}", message_id)

    async def send_response(self, websocket, message_id: str, response_type: str, data: Any):
        """Send response to WebSocket client"""
        response = {
            "id": message_id,
            "type": response_type,
            "payload": data,
            "timestamp": time.time()
        }

        try:
            await websocket.send(json.dumps(response))
        except Exception as e:
            self.logger.error(f"Failed to send response: {e}")

    async def send_error(self, websocket, error_message: str, message_id: str = None):
        """Send error response to WebSocket client"""
        response = {
            "id": message_id or "error",
            "type": "error",
            "payload": {
                "error": error_message,
                "timestamp": time.time()
            }
        }

        try:
            await websocket.send(json.dumps(response))
        except Exception as e:
            self.logger.error(f"Failed to send error: {e}")

    async def send_server_info(self, websocket):
        """Send initial server info to newly connected client"""
        try:
            info = self.mcp_client.get_server_info()
            response = {
                "id": "welcome",
                "type": "server_info",
                "payload": {
                    **info,
                    "websocket_server": {
                        "host": self.host,
                        "port": self.port,
                        "connected_clients": len(self.clients)
                    }
                },
                "timestamp": time.time()
            }
            await websocket.send(json.dumps(response))
        except Exception as e:
            self.logger.error(f"Failed to send server info: {e}")

    async def broadcast_server_status(self):
        """Broadcast server status to all connected clients"""
        if not self.clients:
            return

        try:
            info = self.mcp_client.get_server_info()
            message = {
                "id": "broadcast",
                "type": "server_status_update",
                "payload": info,
                "timestamp": time.time()
            }

            # Send to all connected clients
            await asyncio.gather(
                *[client.send(json.dumps(message)) for client in self.clients],
                return_exceptions=True
            )
        except Exception as e:
            self.logger.error(f"Failed to broadcast server status: {e}")

    async def run_forever(self):
        """Run the WebSocket server indefinitely"""
        await self.start()

        try:
            # Keep the server running
            while self.is_running:
                await asyncio.sleep(1)

                # Periodic health check and broadcast (every 5 minutes)
                if int(time.time()) % 300 == 0:
                    await self.broadcast_server_status()

        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
        finally:
            await self.stop()


async def async_main():
    """Async main function to run the WebSocket server"""
    import argparse

    parser = argparse.ArgumentParser(description="MCP WebSocket Server")
    parser.add_argument("--host", default="localhost", help="WebSocket server host")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket server port")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Create server
    config = Config()
    if args.debug:
        config.log_level = "DEBUG"
        config.development_mode = True

    server = MCPWebSocketServer(config, args.host, args.port)

    try:
        await server.run_forever()
    except Exception as e:
        logging.error(f"Server error: {e}")
        return 1

    return 0


def main():
    """Synchronous entry point for CLI"""
    import sys
    sys.exit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()