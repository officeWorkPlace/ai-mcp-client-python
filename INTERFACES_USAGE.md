# MCP Client - Multi-Interface Usage Guide

This document explains how to run the MCP Client with its three interfaces: Chatbot, REST API, and WebSocket.

## Quick Start

### Run Single Interface

```bash
# Chatbot only (default)
python -m mcp_client --interfaces chatbot

# REST API only
python -m mcp_client --interfaces rest_api --api-port 8000

# WebSocket only
python -m mcp_client --interfaces websocket --ws-port 8765
```

### Run Multiple Interfaces

```bash
# Run all interfaces
python -m mcp_client --interfaces all

# Run specific combination
python -m mcp_client --interfaces chatbot rest_api
python -m mcp_client --interfaces rest_api websocket
```

### Custom Configuration

```bash
# Custom ports and hosts
python -m mcp_client \
  --interfaces all \
  --api-host 0.0.0.0 --api-port 8080 \
  --ws-host 0.0.0.0 --ws-port 9000 \
  --debug

# Different config directory
python -m mcp_client --config-dir /path/to/configs --interfaces all
```

## Interface Details

### 1. Chatbot Interface
- **Type**: Interactive terminal interface
- **Usage**: Direct conversation with MCP services
- **Features**: Rich UI, command support, session stats
- **Access**: Terminal/console

### 2. REST API Interface
- **Type**: HTTP API server
- **Default**: http://127.0.0.1:8000
- **Features**: JSON endpoints, query processing, tool calls
- **Access**: HTTP clients, web apps, curl

#### API Endpoints:
- `GET /` - Server status
- `POST /query` - Process queries
- `POST /tools/call` - Direct tool calls
- `GET /server/info` - Server information
- `GET /server/health` - Health check
- `GET /tools/list` - List available tools
- `POST /conversation/reset` - Reset conversation

### 3. WebSocket Interface
- **Type**: Real-time WebSocket server
- **Default**: ws://localhost:8765
- **Features**: Real-time messaging, JSON protocol
- **Access**: WebSocket clients, web apps

#### WebSocket Messages:
- `{"type": "query", "payload": {"query": "..."}}` - Send query
- `{"type": "get_server_info"}` - Get server info
- `{"type": "health_check"}` - Health check
- `{"type": "list_tools"}` - List tools
- `{"type": "call_tool", "payload": {...}}` - Call tool
- `{"type": "reset_conversation"}` - Reset conversation

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    Chatbot      │    │    REST API     │    │   WebSocket     │
│   Interface     │    │   Interface     │    │   Interface     │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴───────────┐
                    │     MCP Service         │
                    │   (Shared Instance)     │
                    └─────────────┬───────────┘
                                  │
                    ┌─────────────┴───────────┐
                    │   Global MCP Client     │
                    │  (Connected Servers)    │
                    └─────────────────────────┘
```

## Communication Between Interfaces

All three interfaces share the same MCP service instance, which means:

- **Consistent State**: All interfaces see the same server connections and tools
- **Shared Conversations**: Conversation history is shared (when using REST API or WebSocket)
- **Efficient Resources**: Single connection pool to MCP servers
- **Coordinated Operations**: Tool calls and queries are coordinated

## Examples

### Using REST API

```bash
# Start REST API
python -m mcp_client --interfaces rest_api --api-port 8000

# Test with curl
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What tools are available?"}'

curl http://127.0.0.1:8000/server/info
```

### Using WebSocket

```python
import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        # Send query
        message = {
            "type": "query",
            "id": "test-1",
            "payload": {"query": "List available tools"}
        }
        await websocket.send(json.dumps(message))

        # Receive response
        response = await websocket.recv()
        print(json.loads(response))

# Run WebSocket server first:
# python -m mcp_client --interfaces websocket

# Then run this test
asyncio.run(test_websocket())
```

### Running All Interfaces

```bash
# Terminal 1: Start all interfaces
python -m mcp_client --interfaces all

# Terminal 2: Test REST API
curl http://127.0.0.1:8000/server/info

# Terminal 3: Test WebSocket (using wscat if installed)
wscat -c ws://localhost:8765

# Terminal 1: Use chatbot interactively
# The chatbot interface will be available in the same terminal
```

## Environment Variables

```bash
# Configuration
export MCP_CONFIG_DIR="/path/to/configs"
export MCP_LOG_LEVEL="DEBUG"

# API Configuration
export MCP_API_HOST="0.0.0.0"
export MCP_API_PORT="8080"

# WebSocket Configuration
export MCP_WS_HOST="0.0.0.0"
export MCP_WS_PORT="9000"
```

## Troubleshooting

### Port Conflicts
```bash
# Use different ports
python -m mcp_client --interfaces all --api-port 8001 --ws-port 8766
```

### Debug Mode
```bash
# Enable debug logging
python -m mcp_client --interfaces all --debug
```

### Check Server Status
```bash
curl http://127.0.0.1:8000/server/health
```

This setup allows you to use the MCP client through whichever interface is most suitable for your use case, while maintaining consistency across all interfaces.