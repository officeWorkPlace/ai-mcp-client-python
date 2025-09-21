#!/usr/bin/env python3
"""
Startup script for MCP WebSocket Server
Run this to start the WebSocket server that bridges the desktop UI with MCP servers
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from global_mcp_client.websocket_server import MCPWebSocketServer, main

if __name__ == "__main__":
    print("🚀 Starting MCP WebSocket Server...")
    print("📋 Architecture: UI <-WebSocket-> MCP-Client <-stdio-> MCP-Servers")
    print("🌐 Default WebSocket URL: ws://localhost:8765")
    print("🔄 Press Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)
