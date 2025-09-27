#!/usr/bin/env python3
"""
Quick test script to verify MCP Client chatbot is working
"""

import asyncio
import sys
from pathlib import Path

# Add src to path so we can import mcp_client
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mcp_client.core import Config, setup_logging
from mcp_client.services import MCPService

async def test_mcp_service():
    """Test that the MCP service initializes correctly"""
    print("🧪 Testing MCP Client Service...")

    # Setup minimal config
    config = Config()

    # Setup logging without file logging to avoid permission issues
    setup_logging(
        log_level="INFO",
        enable_file_logging=False,  # Disable file logging for test
        enable_console_logging=True
    )

    # Test MCP service initialization
    mcp_service = MCPService(config)

    try:
        success = await mcp_service.initialize()

        if success:
            print("✅ MCP Service initialized successfully!")

            # Test basic functionality
            if hasattr(mcp_service, '_client') and mcp_service._client:
                servers = mcp_service._client.servers
                tools_count = sum(len(tools) for tools in mcp_service._client.tools.values())

                print(f"✅ Connected servers: {len(servers)}")
                print(f"✅ Available tools: {tools_count}")

                for server_name in servers:
                    print(f"   📡 {server_name}")

            print("\n🎉 MCP Client is working correctly!")
            print("\n🚀 You can now start the chatbot with:")
            print("   mcp-client --interfaces chatbot")

        else:
            print("❌ Failed to initialize MCP service")

    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        if hasattr(mcp_service, '_client') and mcp_service._client:
            await mcp_service._client.cleanup()

if __name__ == "__main__":
    print("🧪 MCP Client Test Script")
    print("=" * 40)
    asyncio.run(test_mcp_service())