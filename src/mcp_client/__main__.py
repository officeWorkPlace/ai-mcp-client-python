"""
Main entry point for MCP Client
Provides CLI interface to run chatbot, REST API, and WebSocket interfaces
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path for transition period
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from .services.interface_coordinator import InterfaceCoordinator, InterfaceType
from .core import Config


def main():
    """Main CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="MCP Client - Multi-Interface MCP Client",
        epilog="""
Examples:
  python -m mcp_client --interfaces chatbot
  python -m mcp_client --interfaces rest_api --api-port 8080
  python -m mcp_client --interfaces websocket --ws-port 9000
  python -m mcp_client --interfaces all
  python -m mcp_client --interfaces chatbot rest_api websocket
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Interface selection
    parser.add_argument(
        "--interfaces",
        nargs="+",
        choices=["chatbot", "rest_api", "websocket", "all"],
        default=["chatbot"],
        help="Interfaces to start (default: chatbot)"
    )

    # REST API configuration
    parser.add_argument(
        "--api-host",
        default="127.0.0.1",
        help="REST API host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="REST API port (default: 8000)"
    )

    # WebSocket configuration
    parser.add_argument(
        "--ws-host",
        default="localhost",
        help="WebSocket host (default: localhost)"
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=8765,
        help="WebSocket port (default: 8765)"
    )

    # General configuration
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--config-dir",
        default="configs",
        help="Configuration directory (default: configs)"
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = Config(args.config_dir)
        if args.debug:
            config.log_level = "DEBUG"
            config.development_mode = True
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return 1

    # Create coordinator
    coordinator = InterfaceCoordinator(config)
    coordinator.configure_rest_api(args.api_host, args.api_port)
    coordinator.configure_websocket(args.ws_host, args.ws_port)

    # Determine interfaces to start
    if "all" in args.interfaces:
        interface_types = list(InterfaceType)
        print("Starting all interfaces:")
        print(f"  • Chatbot: Interactive terminal interface")
        print(f"  • REST API: http://{args.api_host}:{args.api_port}")
        print(f"  • WebSocket: ws://{args.ws_host}:{args.ws_port}")
    else:
        interface_types = [InterfaceType(iface) for iface in args.interfaces]
        print("Starting interfaces:")
        for iface_type in interface_types:
            if iface_type == InterfaceType.CHATBOT:
                print(f"  • Chatbot: Interactive terminal interface")
            elif iface_type == InterfaceType.REST_API:
                print(f"  • REST API: http://{args.api_host}:{args.api_port}")
            elif iface_type == InterfaceType.WEBSOCKET:
                print(f"  • WebSocket: ws://{args.ws_host}:{args.ws_port}")

    print()

    # Run coordinator
    try:
        asyncio.run(coordinator.run_with_interfaces(interface_types))
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        return 0
    except Exception as e:
        print(f"Fatal error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())