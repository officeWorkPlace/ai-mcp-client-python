"""
Cross-platform logging configuration for Global MCP Client
Compatible with Windows, Mac, and Linux
"""

import logging
import logging.handlers
import sys
import platform
from pathlib import Path
from typing import Optional
import structlog
from rich.console import Console
from rich.logging import RichHandler


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_file_logging: bool = True,
    enable_console_logging: bool = True,
    development_mode: bool = False,
) -> logging.Logger:
    """
    Set up comprehensive logging for the application

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        enable_file_logging: Whether to enable file logging
        enable_console_logging: Whether to enable console logging
        development_mode: Whether to use development-friendly formatting

    Returns:
        Configured logger instance
    """

    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Set log level
    log_level_obj = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(log_level_obj)

    handlers = []

    # Console handler with rich formatting
    if enable_console_logging:
        console = Console()
        rich_handler = RichHandler(
            console=console,
            show_path=development_mode,
            show_time=True,
            rich_tracebacks=True,
            tracebacks_show_locals=development_mode,
        )

        if development_mode:
            console_format = "%(message)s"
        else:
            console_format = "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"

        rich_handler.setFormatter(logging.Formatter(console_format))
        handlers.append(rich_handler)

    # Cross-platform file handler with smart rotation
    if enable_file_logging and log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_format = (
            "[%(asctime)s] %(levelname)s - %(name)s:%(lineno)d - "
            "%(funcName)s() - %(message)s"
        )

        try:
            # Use different strategies based on platform
            current_platform = platform.system()

            if current_platform == "Windows":
                # Windows: Use regular FileHandler to avoid permission issues with rotation
                file_handler = logging.FileHandler(
                    log_path, mode='a', encoding="utf-8"
                )
            else:
                # Mac/Linux: Use RotatingFileHandler for better log management
                file_handler = logging.handlers.RotatingFileHandler(
                    log_path,
                    maxBytes=10 * 1024 * 1024,  # 10MB
                    backupCount=5,
                    encoding="utf-8"
                )

            file_handler.setFormatter(logging.Formatter(file_format))
            handlers.append(file_handler)

        except (PermissionError, OSError) as e:
            # Graceful fallback for any file system issues on any platform
            fallback_msg = f"Warning: Cannot write to log file {log_path} ({e}). Continuing with console logging only."
            if enable_console_logging:
                print(fallback_msg)
            else:
                # If console logging is also disabled, we need to enable it as fallback
                console = Console()
                fallback_handler = RichHandler(console=console, show_time=True)
                fallback_handler.setFormatter(logging.Formatter("%(message)s"))
                handlers.append(fallback_handler)
                print(fallback_msg)
                print("Enabled console logging as fallback.")

    # Add handlers to root logger
    for handler in handlers:
        root_logger.addHandler(handler)

    # Configure structlog for structured logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Create application logger
    logger = logging.getLogger("global_mcp_client")

    # Log initial setup
    logger.info(
        "Logging initialized",
        extra={
            "log_level": log_level,
            "file_logging": enable_file_logging,
            "console_logging": enable_console_logging,
            "log_file": log_file if enable_file_logging else None,
        },
    )

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module"""
    return logging.getLogger(f"global_mcp_client.{name}")


class LoggerMixin:
    """Mixin class to add logging capabilities to any class"""

    @property
    def logger(self) -> logging.Logger:
        """Get logger instance for this class"""
        return get_logger(self.__class__.__module__ + "." + self.__class__.__name__)


def log_function_call(func):
    """Decorator to log function calls with arguments and results"""

    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")

        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}")
            raise

    return wrapper


async def log_async_function_call(func):
    """Decorator to log async function calls with arguments and results"""

    async def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling async {func.__name__} with args={args}, kwargs={kwargs}")

        try:
            result = await func(*args, **kwargs)
            logger.debug(f"Async {func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Async {func.__name__} failed with error: {e}")
            raise

    return wrapper
