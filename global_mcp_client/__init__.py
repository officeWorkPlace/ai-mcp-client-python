"""
Global MCP Client - A production-ready MCP client for connecting to multiple MCP servers
"""

__version__ = "1.0.0"
__author__ = "Global MCP Client Team"
__email__ = "team@globalmcp.com"
__license__ = "MIT"

from .core.client import GlobalMCPClient
from .core.config import Config
from .core.logger import setup_logging

__all__ = [
    "GlobalMCPClient",
    "Config",
    "setup_logging",
    "__version__",
]
