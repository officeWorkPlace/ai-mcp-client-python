"""
Service layer for MCP Client
Provides shared services and coordination between interfaces
"""

from .mcp_service import MCPService
from .interface_coordinator import InterfaceCoordinator

__all__ = ["MCPService", "InterfaceCoordinator"]