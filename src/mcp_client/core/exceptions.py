"""
Exception classes for Global MCP Client
"""


class GlobalMCPClientError(Exception):
    """Base exception for Global MCP Client"""

    pass


class ConfigurationError(GlobalMCPClientError):
    """Raised when there's a configuration issue"""

    pass


class ServerConnectionError(GlobalMCPClientError):
    """Raised when connection to MCP server fails"""

    pass


class ServerInitializationError(GlobalMCPClientError):
    """Raised when MCP server initialization fails"""

    pass


class ToolExecutionError(GlobalMCPClientError):
    """Raised when tool execution fails"""

    pass


class AIProviderError(GlobalMCPClientError):
    """Raised when AI provider interaction fails"""

    pass


class ValidationError(GlobalMCPClientError):
    """Raised when validation fails"""

    pass


class TimeoutError(GlobalMCPClientError):
    """Raised when operations timeout"""

    pass


class RateLimitError(GlobalMCPClientError):
    """Raised when rate limits are exceeded"""

    pass


class AuthenticationError(GlobalMCPClientError):
    """Raised when authentication fails"""

    pass


class ServerNotFoundError(GlobalMCPClientError):
    """Raised when requested server is not found"""

    pass


class ToolNotFoundError(GlobalMCPClientError):
    """Raised when requested tool is not found"""

    pass
