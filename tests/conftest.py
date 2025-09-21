"""
Test configuration for Global MCP Client
"""

import pytest
import asyncio
from pathlib import Path


# Test configuration
TEST_CONFIG_DIR = Path(__file__).parent.parent / "configs"
TEST_LOG_DIR = Path(__file__).parent.parent / "logs" / "tests"

# Ensure test directories exist
TEST_LOG_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config():
    """Provide test configuration"""
    return {
        "test_mode": True,
        "log_level": "DEBUG",
        "config_dir": str(TEST_CONFIG_DIR),
        "log_file": str(TEST_LOG_DIR / "test.log")
    }


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for testing"""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("DEVELOPMENT_MODE", "true")
