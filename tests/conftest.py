"""
Cross-platform test configuration for MCP Client
Compatible with Windows, Mac, and Linux
"""

import pytest
import asyncio
import platform
import tempfile
from pathlib import Path


# Test configuration
TEST_CONFIG_DIR = Path(__file__).parent.parent / "configs"
TEST_LOG_DIR = Path(__file__).parent.parent / "logs" / "tests"

# Ensure test directories exist
TEST_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Platform information for tests
CURRENT_PLATFORM = platform.system()
IS_WINDOWS = CURRENT_PLATFORM == "Windows"
IS_MAC = CURRENT_PLATFORM == "Darwin"
IS_LINUX = CURRENT_PLATFORM == "Linux"


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


@pytest.fixture
def temp_log_file():
    """Create a temporary log file for cross-platform testing"""
    import gc
    import time

    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        temp_file_name = f.name

    yield temp_file_name

    # Cross-platform cleanup with Windows file handle handling
    try:
        # Force garbage collection to close any open handles
        gc.collect()

        if IS_WINDOWS:
            # On Windows, we need to be more careful about file handles
            import logging
            # Clear any lingering file handlers
            for handler in logging.getLogger().handlers[:]:
                if hasattr(handler, 'close'):
                    handler.close()
            # Additional small delay for Windows file system
            time.sleep(0.1)

        # Attempt cleanup
        temp_path = Path(temp_file_name)
        if temp_path.exists():
            temp_path.unlink()

    except (PermissionError, OSError) as e:
        # On Windows, if file is still locked, try a few more times
        if IS_WINDOWS:
            import logging
            # Remove and close all handlers
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                if hasattr(handler, 'close'):
                    handler.close()
                root_logger.removeHandler(handler)

            # Try cleanup again after a brief wait
            for attempt in range(3):
                time.sleep(0.2)
                gc.collect()
                try:
                    temp_path = Path(temp_file_name)
                    if temp_path.exists():
                        temp_path.unlink()
                    break
                except (PermissionError, OSError):
                    if attempt == 2:  # Last attempt
                        # Log warning but don't fail the test
                        print(f"Warning: Could not cleanup temp file {temp_file_name}: {e}")
        else:
            # On Unix systems, log warning but don't fail
            print(f"Warning: Could not cleanup temp file {temp_file_name}: {e}")


@pytest.fixture
def platform_paths():
    """Provide platform-specific test paths"""
    if IS_WINDOWS:
        return {
            "safe_paths": [
                "data\\file.txt",
                "logs\\app.log",
                ".\\config\\settings.json"
            ],
            "dangerous_paths": [
                "C:\\Windows\\System32\\cmd.exe",
                "C:\\Program Files\\test.exe",
                "..\\..\\..\\Windows\\System32"
            ]
        }
    else:  # Mac/Linux
        return {
            "safe_paths": [
                "data/file.txt",
                "logs/app.log",
                "./config/settings.json"
            ],
            "dangerous_paths": [
                "/etc/passwd",
                "/bin/sh",
                "../../../etc/shadow"
            ]
        }


@pytest.fixture
def cross_platform_config(tmp_path):
    """Create a cross-platform test configuration"""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    return {
        "config_dir": str(config_dir),
        "log_dir": str(tmp_path / "logs"),
        "platform": CURRENT_PLATFORM,
        "is_windows": IS_WINDOWS,
        "is_mac": IS_MAC,
        "is_linux": IS_LINUX
    }
