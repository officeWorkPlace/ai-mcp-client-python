"""
Configuration management for Global MCP Client
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server"""

    command: str
    args: List[str] = Field(default_factory=list)
    description: str = ""
    enabled: bool = True
    timeout: int = 30
    retry_attempts: int = 3
    cwd: Optional[str] = None
    env: Dict[str, str] = Field(default_factory=dict)


class GlobalSettings(BaseModel):
    """Global MCP client settings"""

    max_concurrent_connections: int = 10
    connection_timeout: int = 30
    initialization_timeout: int = 10
    tool_call_timeout: int = 60
    enable_health_checks: bool = True
    health_check_interval: int = 300
    auto_reconnect: bool = True
    max_reconnect_attempts: int = 3
    log_tool_calls: bool = True
    validate_schemas: bool = True


class Config:
    """Central configuration management"""

    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = (
            Path(config_dir) if config_dir else Path(__file__).parent.parent / "configs"
        )
        self.env_file = Path.cwd() / ".env"

        # Load environment variables
        if self.env_file.exists():
            load_dotenv(self.env_file)

        # Initialize configuration
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from files"""
        # Load MCP servers configuration
        servers_config_file = self.config_dir / "mcp_servers.json"
        if servers_config_file.exists():
            with open(servers_config_file, "r") as f:
                config_data = json.load(f)

            # Expand environment variables
            config_data = self._expand_env_vars(config_data)

            # Parse server configurations
            self.mcp_servers = {}
            for name, server_config in config_data.get("mcpServers", {}).items():
                self.mcp_servers[name] = MCPServerConfig(**server_config)

            # Parse global settings
            self.global_settings = GlobalSettings(
                **config_data.get("global_settings", {})
            )
        else:
            self.mcp_servers = {}
            self.global_settings = GlobalSettings()

    def _expand_env_vars(self, data: Any) -> Any:
        """Recursively expand environment variables in configuration"""
        if isinstance(data, dict):
            return {key: self._expand_env_vars(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._expand_env_vars(item) for item in data]
        elif isinstance(data, str) and data.startswith("${") and data.endswith("}"):
            env_var = data[2:-1]
            return os.getenv(env_var, data)
        return data

    @property
    def anthropic_api_key(self) -> Optional[str]:
        """Get Anthropic API key"""
        return os.getenv("ANTHROPIC_API_KEY")

    @property
    def openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key"""
        return os.getenv("OPENAI_API_KEY")

    @property
    def gemini_api_key(self) -> Optional[str]:
        """Get Gemini API key"""
        return os.getenv("GEMINI_API_KEY")

    @property
    def default_model(self) -> str:
        """Get default AI model"""
        return os.getenv("DEFAULT_MODEL", "claude-3-7-sonnet-20250219")

    @property
    def max_tokens(self) -> int:
        """Get maximum tokens for AI responses"""
        return int(os.getenv("MAX_TOKENS", "4096"))

    @property
    def temperature(self) -> float:
        """Get temperature for AI responses"""
        return float(os.getenv("TEMPERATURE", "0.1"))

    @property
    def log_level(self) -> str:
        """Get logging level"""
        return os.getenv("LOG_LEVEL", "INFO")

    @property
    def log_file(self) -> str:
        """Get log file path"""
        return os.getenv("LOG_FILE", "logs/global_mcp_client.log")

    @property
    def enable_file_logging(self) -> bool:
        """Check if file logging is enabled"""
        return os.getenv("ENABLE_FILE_LOGGING", "true").lower() == "true"

    @property
    def enable_console_logging(self) -> bool:
        """Check if console logging is enabled"""
        return os.getenv("ENABLE_CONSOLE_LOGGING", "true").lower() == "true"

    @property
    def debug(self) -> bool:
        """Check if debug mode is enabled"""
        return os.getenv("DEBUG", "false").lower() == "true"

    @property
    def development_mode(self) -> bool:
        """Check if development mode is enabled"""
        return os.getenv("DEVELOPMENT_MODE", "false").lower() == "true"

    def get_enabled_servers(self) -> Dict[str, MCPServerConfig]:
        """Get only enabled MCP servers"""
        return {
            name: config for name, config in self.mcp_servers.items() if config.enabled
        }

    def add_server(self, name: str, config: MCPServerConfig) -> None:
        """Add a new MCP server configuration"""
        self.mcp_servers[name] = config

    def remove_server(self, name: str) -> bool:
        """Remove an MCP server configuration"""
        if name in self.mcp_servers:
            del self.mcp_servers[name]
            return True
        return False

    def save_config(self) -> None:
        """Save current configuration to file"""
        config_data = {
            "mcpServers": {
                name: config.dict() for name, config in self.mcp_servers.items()
            },
            "global_settings": self.global_settings.dict(),
        }

        servers_config_file = self.config_dir / "mcp_servers.json"
        with open(servers_config_file, "w") as f:
            json.dump(config_data, f, indent=2)

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []

        # Check API keys
        if not self.anthropic_api_key and not self.openai_api_key and not self.gemini_api_key:
            issues.append(
                "No API keys configured. Please set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY"
            )

        # Check if any servers are enabled
        enabled_servers = self.get_enabled_servers()
        if not enabled_servers:
            issues.append("No MCP servers are enabled")

        # Validate server configurations
        for name, server_config in enabled_servers.items():
            if not server_config.command:
                issues.append(f"Server '{name}' has no command specified")

            if server_config.cwd and not Path(server_config.cwd).exists():
                issues.append(
                    f"Server '{name}' working directory does not exist: {server_config.cwd}"
                )

        return issues
