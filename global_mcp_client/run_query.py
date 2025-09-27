#!/usr/bin/env python3
"""
Simple script to run a single query with enhanced MCP client
This avoids the asyncio cleanup issues when piping input
"""

import asyncio
import sys
import os
from typing import Optional

from .core import GlobalMCPClient, Config, setup_logging
from .core.exceptions import GlobalMCPClientError


async def run_single_query(query: str, config: Optional[Config] = None) -> str:
    """
    Run a single query and return the result

    Args:
        query: Query to process
        config: Optional configuration

    Returns:
        Response string
    """
    config = config or Config()

    # Setup logging with minimal output for piped usage
    logger = setup_logging(
        log_level="ERROR",  # Only show errors
        enable_console_logging=False,
        enable_file_logging=False,
    )

    client = None
    try:
        print("[INFO] Initializing enhanced MCP client...", file=sys.stderr)

        # Initialize client
        client = GlobalMCPClient(config)
        await asyncio.wait_for(client.connect_to_all_servers(), timeout=30.0)

        print(f"[INFO] Connected with {len(client.enhancement_components)} AI enhancements active", file=sys.stderr)
        print(f"[INFO] Processing query with enhanced reasoning...", file=sys.stderr)

        # Process query with enhancements - add timeout for piped usage
        response = await asyncio.wait_for(client.process_query(query), timeout=120.0)

        return response

    except asyncio.TimeoutError:
        return "Error: Query processing timed out (2 minutes). Try a simpler query or check MCP server status."
    except GlobalMCPClientError as e:
        return f"MCP Client Error: {str(e)}"
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        return f"Error: {str(e)}"
    finally:
        if client:
            print("[INFO] Cleaning up...", file=sys.stderr)
            try:
                # Quick cleanup without waiting for complex async cleanup
                await asyncio.wait_for(client.cleanup(), timeout=3.0)
            except:
                # Suppress all cleanup errors for piped mode
                pass


async def main():
    """Main function for single query execution"""
    if len(sys.argv) > 1:
        # Query passed as command line argument
        query = " ".join(sys.argv[1:])
    else:
        # Read from stdin
        try:
            query = sys.stdin.read().strip()
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)

    if not query:
        print("No query provided", file=sys.stderr)
        sys.exit(1)

    try:
        config = Config("configs")
        response = await run_single_query(query, config)
        print(response)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # Set event loop policy for Windows compatibility
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    asyncio.run(main())