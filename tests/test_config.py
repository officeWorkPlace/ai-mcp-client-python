"""
Tests for configuration management
"""

import pytest
import tempfile
import json
from pathlib import Path

from global_mcp_client.core.config import Config, MCPServerConfig, GlobalSettings


class TestConfig:
    """Test configuration management"""
    
    def test_config_initialization(self, mock_env_vars):
        """Test basic configuration initialization"""
        config = Config()
        
        assert config.anthropic_api_key == "test-anthropic-key"
        assert config.log_level == "DEBUG"
        assert config.debug is True
        assert config.development_mode is True
    
    def test_mcp_server_config_validation(self):
        """Test MCP server configuration validation"""
        # Valid configuration
        server_config = MCPServerConfig(
            command="uv",
            args=["run", "server.py"],
            description="Test server",
            enabled=True
        )
        
        assert server_config.command == "uv"
        assert server_config.args == ["run", "server.py"]
        assert server_config.enabled is True
        assert server_config.timeout == 30  # default
    
    def test_global_settings_defaults(self):
        """Test global settings with defaults"""
        settings = GlobalSettings()
        
        assert settings.max_concurrent_connections == 10
        assert settings.connection_timeout == 30
        assert settings.enable_health_checks is True
        assert settings.auto_reconnect is True
    
    def test_config_validation_with_issues(self, monkeypatch, tmp_path):
        """Test configuration validation with issues"""
        # Remove API keys to create validation issues
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        # Use temp directory to avoid loading .env file
        monkeypatch.chdir(tmp_path)

        # Create config after removing environment variables
        config = Config()
        issues = config.validate()

        assert len(issues) > 0
        assert any("API keys" in issue for issue in issues)
    
    def test_config_file_loading(self, tmp_path):
        """Test loading configuration from file"""
        # Create temporary config file
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        
        config_file = config_dir / "mcp_servers.json"
        test_config = {
            "mcpServers": {
                "test-server": {
                    "command": "test",
                    "args": ["--test"],
                    "description": "Test server",
                    "enabled": True,
                    "timeout": 30,
                    "retry_attempts": 3
                }
            },
            "global_settings": {
                "max_concurrent_connections": 5,
                "connection_timeout": 20
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(test_config, f)
        
        # Load configuration
        config = Config(config_dir=str(config_dir))
        
        assert "test-server" in config.mcp_servers
        assert config.mcp_servers["test-server"].command == "test"
        assert config.global_settings.max_concurrent_connections == 5
    
    def test_environment_variable_expansion(self, tmp_path, monkeypatch):
        """Test environment variable expansion in config"""
        monkeypatch.setenv("TEST_COMMAND", "expanded_command")
        
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        
        config_file = config_dir / "mcp_servers.json"
        test_config = {
            "mcpServers": {
                "test-server": {
                    "command": "${TEST_COMMAND}",
                    "args": [],
                    "description": "Test server",
                    "enabled": True,
                    "timeout": 30,
                    "retry_attempts": 3
                }
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(test_config, f)
        
        config = Config(config_dir=str(config_dir))
        
        assert config.mcp_servers["test-server"].command == "expanded_command"
    
    def test_enabled_servers_filtering(self, tmp_path):
        """Test filtering of enabled servers"""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        
        config_file = config_dir / "mcp_servers.json"
        test_config = {
            "mcpServers": {
                "enabled-server": {
                    "command": "test1",
                    "args": [],
                    "description": "Enabled server",
                    "enabled": True,
                    "timeout": 30,
                    "retry_attempts": 3
                },
                "disabled-server": {
                    "command": "test2",
                    "args": [],
                    "description": "Disabled server",
                    "enabled": False,
                    "timeout": 30,
                    "retry_attempts": 3
                }
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(test_config, f)
        
        config = Config(config_dir=str(config_dir))
        enabled_servers = config.get_enabled_servers()
        
        assert len(enabled_servers) == 1
        assert "enabled-server" in enabled_servers
        assert "disabled-server" not in enabled_servers
    
    def test_config_save(self, tmp_path):
        """Test saving configuration"""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        
        config = Config(config_dir=str(config_dir))
        
        # Add a server
        server_config = MCPServerConfig(
            command="test_command",
            args=["--test"],
            description="Test server"
        )
        config.add_server("test-server", server_config)
        
        # Save configuration
        config.save_config()
        
        # Verify file was created and contains correct data
        config_file = config_dir / "mcp_servers.json"
        assert config_file.exists()
        
        with open(config_file, 'r') as f:
            saved_config = json.load(f)
        
        assert "test-server" in saved_config["mcpServers"]
        assert saved_config["mcpServers"]["test-server"]["command"] == "test_command"


@pytest.mark.asyncio
class TestAsyncConfig:
    """Test async configuration operations"""
    
    async def test_async_config_operations(self, mock_env_vars):
        """Test that config can be used in async context"""
        config = Config()
        
        # This should work without issues in async context
        assert config.anthropic_api_key == "test-anthropic-key"
        
        enabled_servers = config.get_enabled_servers()
        assert isinstance(enabled_servers, dict)
