"""
Utility helper functions
"""

import os
import sys
import json
import yaml
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone


def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).parent.parent.parent


def ensure_directory_exists(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def load_json_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON data from a file

    Args:
        file_path: Path to JSON file

    Returns:
        Parsed JSON data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with open(path_obj, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(
    data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2
) -> None:
    """
    Save data to a JSON file

    Args:
        data: Data to save
        file_path: Path to save the file
        indent: JSON indentation
    """
    path_obj = Path(file_path)
    ensure_directory_exists(path_obj.parent)

    with open(path_obj, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_yaml_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML data from a file

    Args:
        file_path: Path to YAML file

    Returns:
        Parsed YAML data
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")

    with open(path_obj, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml_file(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Save data to a YAML file

    Args:
        data: Data to save
        file_path: Path to save the file
    """
    path_obj = Path(file_path)
    ensure_directory_exists(path_obj.parent)

    with open(path_obj, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)


def generate_hash(text: str, algorithm: str = "sha256") -> str:
    """
    Generate hash for a text string

    Args:
        text: Text to hash
        algorithm: Hash algorithm to use

    Returns:
        Hex digest of the hash
    """
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(text.encode("utf-8"))
    return hash_obj.hexdigest()


def get_timestamp(include_microseconds: bool = False) -> str:
    """
    Get current timestamp in ISO format

    Args:
        include_microseconds: Whether to include microseconds

    Returns:
        ISO formatted timestamp
    """
    now = datetime.now(timezone.utc)
    if include_microseconds:
        return now.isoformat()
    else:
        return now.replace(microsecond=0).isoformat()


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable format

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_bytes(bytes_count: int) -> str:
    """
    Format byte count to human-readable format

    Args:
        bytes_count: Number of bytes

    Returns:
        Formatted byte string
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_count < 1024.0:
            return f"{bytes_count:.1f}{unit}"
        bytes_count /= 1024.0
    return f"{bytes_count:.1f}PB"


def safe_get_nested(data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    """
    Safely get nested dictionary value

    Args:
        data: Dictionary to search
        keys: List of keys to traverse
        default: Default value if key path doesn't exist

    Returns:
        Value at the key path or default
    """
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries

    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence)

    Returns:
        Merged dictionary
    """
    result = dict1.copy()

    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


def flatten_dict(data: Dict[str, Any], separator: str = ".") -> Dict[str, Any]:
    """
    Flatten a nested dictionary

    Args:
        data: Dictionary to flatten
        separator: Separator for nested keys

    Returns:
        Flattened dictionary
    """

    def _flatten(obj: Any, parent_key: str = "") -> Dict[str, Any]:
        items = []

        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{parent_key}{separator}{key}" if parent_key else key
                items.extend(_flatten(value, new_key).items())
        else:
            return {parent_key: obj}

        return dict(items)

    return _flatten(data)


def retry_on_exception(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    Decorator for retrying function calls on exceptions

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Factor to multiply delay by after each attempt
        exceptions: Tuple of exceptions to catch and retry on
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        raise e

                    time.sleep(current_delay)
                    current_delay *= backoff_factor

        return wrapper

    return decorator


def async_retry_on_exception(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    Decorator for retrying async function calls on exceptions

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Factor to multiply delay by after each attempt
        exceptions: Tuple of exceptions to catch and retry on
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            import asyncio

            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        raise e

                    await asyncio.sleep(current_delay)
                    current_delay *= backoff_factor

        return wrapper

    return decorator


def get_system_info() -> Dict[str, Any]:
    """
    Get system information

    Returns:
        Dictionary containing system information
    """
    import platform
    import psutil

    return {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_total": psutil.virtual_memory().total,
        "memory_available": psutil.virtual_memory().available,
        "disk_usage": {
            "total": (
                psutil.disk_usage("/").total
                if os.name != "nt"
                else psutil.disk_usage("C:").total
            ),
            "used": (
                psutil.disk_usage("/").used
                if os.name != "nt"
                else psutil.disk_usage("C:").used
            ),
            "free": (
                psutil.disk_usage("/").free
                if os.name != "nt"
                else psutil.disk_usage("C:").free
            ),
        },
    }


def validate_environment() -> List[str]:
    """
    Validate the environment for required dependencies

    Returns:
        List of missing dependencies or issues
    """
    issues = []

    # Check Python version
    if sys.version_info < (3, 8):
        issues.append("Python 3.8 or higher is required")

    # Check required packages
    required_packages = [
        "mcp",
        "anthropic",
        "rich",
        "pydantic",
        "structlog",
        "tenacity",
    ]

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            issues.append(f"Required package '{package}' not found")

    return issues


def setup_signal_handlers():
    """Set up graceful shutdown signal handlers"""
    import signal
    import asyncio

    def signal_handler(signum, frame):
        """Handle shutdown signals"""
        print(f"\nReceived signal {signum}. Initiating graceful shutdown...")
        # Set the event loop to stop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.stop()

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal_handler)


def check_port_availability(host: str = "localhost", port: int = 8000) -> bool:
    """
    Check if a port is available

    Args:
        host: Host to check
        port: Port number to check

    Returns:
        True if port is available, False otherwise
    """
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, port))
            return True
    except OSError:
        return False


def get_available_port(
    start_port: int = 8000, max_attempts: int = 100
) -> Optional[int]:
    """
    Find an available port starting from a given port

    Args:
        start_port: Starting port number
        max_attempts: Maximum number of ports to try

    Returns:
        Available port number or None if none found
    """
    for port in range(start_port, start_port + max_attempts):
        if check_port_availability(port=port):
            return port
    return None
