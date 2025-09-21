"""
Main entry point for Global MCP Client
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from global_mcp_client.chatbot import main as chatbot_main
from global_mcp_client.core import Config, setup_logging
from global_mcp_client.utils import setup_signal_handlers, validate_environment


def main():
    """Main entry point for the application"""
    try:
        # Setup signal handlers for graceful shutdown
        setup_signal_handlers()

        # Validate environment
        issues = validate_environment()
        if issues:
            print("Environment validation failed:")
            for issue in issues:
                print(f"  - {issue}")
            sys.exit(1)

        # Load configuration
        try:
            config = Config()
        except Exception as e:
            print(f"Failed to load configuration: {e}")
            sys.exit(1)

        # Setup basic logging
        logger = setup_logging(
            log_level=config.log_level,
            log_file=config.log_file,
            enable_file_logging=config.enable_file_logging,
            enable_console_logging=config.enable_console_logging,
            development_mode=config.development_mode,
        )

        logger.info("Starting Global MCP Client")

        # Run the chatbot
        asyncio.run(chatbot_main())

    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        if "--debug" in sys.argv:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
