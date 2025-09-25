"""
MCP Client - Model Context Protocol Client with Multiple Interfaces

A comprehensive MCP client that provides:
- Interactive chatbot interface
- REST API for HTTP-based integration
- WebSocket server for real-time communication
- Shared MCP service layer for consistency
"""

from .interfaces import MCPChatBot, MCPRestAPI, MCPWebSocketServer
from .services import MCPService, InterfaceCoordinator

# Import core components from new structure
from .core import GlobalMCPClient, Config
from .core.exceptions import GlobalMCPClientError

__version__ = "1.0.0"
__author__ = "MCP Client Team"
__email__ = "support@mcpclient.com"

__all__ = [
    "MCPChatBot",
    "MCPRestAPI",
    "MCPWebSocketServer",
    "MCPService",
    "InterfaceCoordinator",
    "GlobalMCPClient",
    "Config",
    "GlobalMCPClientError",
]