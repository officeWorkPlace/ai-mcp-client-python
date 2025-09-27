# ⚡ WebSocket Reference Guide

Complete reference for the MCP Client WebSocket interface with detailed examples and implementations.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Connection](#connection)
- [Message Protocol](#message-protocol)
- [Message Types](#message-types)
- [Implementation Examples](#implementation-examples)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)

---

## 🎯 Overview

The MCP Client WebSocket interface provides **real-time bidirectional communication** for AI-enhanced query processing. Built for modern web applications that need instant responses and streaming capabilities.

### Key Features
- **Real-time Communication** - Instant bidirectional messaging
- **Streaming Responses** - Get partial results as they're generated
- **Connection Management** - Automatic reconnection and health monitoring
- **AI-Enhanced Processing** - Full access to chain-of-thought reasoning
- **Multi-Client Support** - Handle multiple concurrent connections
- **Event-Driven Architecture** - Subscribe to specific events and updates

---

## 🚀 Getting Started

### Start the WebSocket Server
```bash
# Start WebSocket server on default port 8765
mcp-client --interfaces websocket

# Start on custom port
mcp-client --interfaces websocket --ws-port 9000

# Start with debug mode
mcp-client --interfaces websocket --ws-port 8765 --debug
```

### Verify Server is Running
```bash
# Test with websocat (install: cargo install websocat)
websocat ws://localhost:8765

# Or use wscat (install: npm install -g wscat)
wscat -c ws://localhost:8765

# Send a test message
{"type": "ping"}
```

---

## 🔌 Connection

### WebSocket URL
```
ws://localhost:8765
```

### Connection Example (JavaScript)
```javascript
const ws = new WebSocket('ws://localhost:8765');

ws.onopen = function(event) {
    console.log('Connected to MCP WebSocket server');

    // Send ping to verify connection
    ws.send(JSON.stringify({
        type: 'ping',
        id: 'connection-test'
    }));
};

ws.onmessage = function(event) {
    const message = JSON.parse(event.data);
    console.log('Received:', message);
};

ws.onerror = function(error) {
    console.error('WebSocket error:', error);
};

ws.onclose = function(event) {
    console.log('Disconnected from server:', event.code, event.reason);
};
```

### Connection with Node.js
```javascript
const WebSocket = require('ws');

const ws = new WebSocket('ws://localhost:8765');

ws.on('open', function open() {
    console.log('Connected to MCP WebSocket server');

    // Send authentication or initialization message
    ws.send(JSON.stringify({
        type: 'init',
        id: 'client-001',
        payload: {
            client_name: 'Node.js Client',
            version: '1.0.0'
        }
    }));
});

ws.on('message', function message(data) {
    const msg = JSON.parse(data.toString());
    console.log('Received:', msg);
});

ws.on('error', function error(err) {
    console.error('WebSocket error:', err);
});

ws.on('close', function close() {
    console.log('Disconnected from server');
});
```

### Connection with Python
```python
import asyncio
import websockets
import json

async def connect_to_mcp():
    uri = "ws://localhost:8765"

    async with websockets.connect(uri) as websocket:
        print("Connected to MCP WebSocket server")

        # Send ping message
        ping_msg = {
            "type": "ping",
            "id": "python-client-001"
        }
        await websocket.send(json.dumps(ping_msg))

        # Listen for messages
        async for message in websocket:
            data = json.loads(message)
            print(f"Received: {data}")

# Run the client
asyncio.run(connect_to_mcp())
```

---

## 📨 Message Protocol

### Message Structure
All WebSocket messages use JSON format with a consistent structure:

```json
{
  "type": "message_type",
  "id": "unique_message_id",
  "payload": {
    "key": "value"
  },
  "timestamp": 1758768459.837
}
```

### Required Fields
- **`type`** (string) - Message type identifier
- **`id`** (string) - Unique message identifier for correlation

### Optional Fields
- **`payload`** (object) - Message-specific data
- **`timestamp`** (number) - Unix timestamp (added by server)

---

## 🎭 Message Types

## 1. Ping/Pong Messages

### Ping Request
```json
{
  "type": "ping",
  "id": "ping-001"
}
```

### Pong Response
```json
{
  "type": "pong",
  "id": "ping-001",
  "timestamp": 1758768459.837
}
```

### Example with websocat
```bash
# Terminal 1: Start server
mcp-client --interfaces websocket --ws-port 8765

# Terminal 2: Test ping/pong
echo '{"type": "ping", "id": "test-001"}' | websocat ws://localhost:8765
```

---

## 2. Query Processing

### Query Request
```json
{
  "type": "query",
  "id": "query-001",
  "payload": {
    "query": "Show me the top 10 customers by loan amount",
    "context": {
      "user_id": "analyst_001",
      "session_id": "session_123"
    }
  }
}
```

### Query Response
```json
{
  "type": "query_response",
  "id": "query-001",
  "payload": {
    "query": "Show me the top 10 customers by loan amount",
    "result": "Here are the top 10 customers by loan amount:\n\n1. John Smith - $450,000\n2. Maria Garcia - $425,000\n...",
    "processing_time": 2.34,
    "tools_used": ["execute_sql_query", "format_results"],
    "ai_enhancements": {
      "context_optimization": true,
      "chain_of_thought": true,
      "quality_score": 9.2
    }
  },
  "timestamp": 1758768459.837
}
```

### Example with websocat
```bash
echo '{
  "type": "query",
  "id": "test-query-001",
  "payload": {
    "query": "What is the current database schema?"
  }
}' | websocat ws://localhost:8765
```

---

## 3. Streaming Responses

### Stream Start
```json
{
  "type": "stream_start",
  "id": "query-002",
  "payload": {
    "query": "Generate comprehensive business intelligence report",
    "estimated_duration": 15.5
  },
  "timestamp": 1758768459.837
}
```

### Stream Chunk
```json
{
  "type": "stream_chunk",
  "id": "query-002",
  "payload": {
    "chunk": "## Customer Demographics Analysis\n\nTotal Customers: 1,247\n",
    "chunk_index": 1,
    "is_final": false
  },
  "timestamp": 1758768460.123
}
```

### Stream End
```json
{
  "type": "stream_end",
  "id": "query-002",
  "payload": {
    "final_result": "Complete business intelligence report...",
    "total_chunks": 15,
    "processing_time": 14.7,
    "tools_used": ["get_customer_data", "analyze_trends", "generate_report"]
  },
  "timestamp": 1758768474.456
}
```

---

## 4. Tool Execution

### Tool Call Request
```json
{
  "type": "call_tool",
  "id": "tool-001",
  "payload": {
    "tool_name": "execute_sql_query",
    "arguments": {
      "query": "SELECT COUNT(*) FROM customers WHERE loan_amount > 100000"
    }
  }
}
```

### Tool Call Response
```json
{
  "type": "tool_result",
  "id": "tool-001",
  "payload": {
    "tool_name": "execute_sql_query",
    "result": {
      "columns": ["COUNT(*)"],
      "rows": [["342"]],
      "row_count": 1
    },
    "processing_time": 0.45,
    "success": true
  },
  "timestamp": 1758768459.837
}
```

---

## 5. Server Information

### Server Info Request
```json
{
  "type": "get_server_info",
  "id": "info-001"
}
```

### Server Info Response
```json
{
  "type": "server_info",
  "id": "info-001",
  "payload": {
    "connected_servers": [
      {
        "name": "oracle-db",
        "status": "connected",
        "tools_count": 93
      }
    ],
    "total_tools": 93,
    "server_status": "healthy",
    "uptime": "2h 34m 12s"
  },
  "timestamp": 1758768459.837
}
```

---

## 6. Error Messages

### Error Response
```json
{
  "type": "error",
  "id": "query-003",
  "payload": {
    "error_code": "TOOL_EXECUTION_ERROR",
    "error_message": "Failed to execute SQL query: table 'invalid_table' does not exist",
    "details": {
      "tool_name": "execute_sql_query",
      "query": "SELECT * FROM invalid_table"
    }
  },
  "timestamp": 1758768459.837
}
```

---

## 🛠️ Implementation Examples

## Complete JavaScript Client

```html
<!DOCTYPE html>
<html>
<head>
    <title>MCP WebSocket Client</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        #messages { height: 400px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; }
        #queryInput { width: 80%; padding: 10px; }
        #sendBtn { padding: 10px 20px; }
        .message { margin: 5px 0; padding: 5px; border-radius: 3px; }
        .sent { background-color: #e3f2fd; }
        .received { background-color: #f3e5f5; }
        .error { background-color: #ffebee; color: #c62828; }
    </style>
</head>
<body>
    <h1>MCP WebSocket Client</h1>

    <div id="connectionStatus">Disconnected</div>

    <div id="messages"></div>

    <div>
        <input type="text" id="queryInput" placeholder="Enter your query..." />
        <button id="sendBtn" onclick="sendQuery()">Send Query</button>
        <button onclick="pingServer()">Ping</button>
        <button onclick="getServerInfo()">Server Info</button>
    </div>

    <script>
        let ws = null;
        let messageId = 1;

        function connect() {
            ws = new WebSocket('ws://localhost:8765');

            ws.onopen = function(event) {
                updateConnectionStatus('Connected', 'green');
                addMessage('Connected to MCP WebSocket server', 'system');
            };

            ws.onmessage = function(event) {
                const message = JSON.parse(event.data);
                handleMessage(message);
            };

            ws.onerror = function(error) {
                addMessage('WebSocket error: ' + error, 'error');
            };

            ws.onclose = function(event) {
                updateConnectionStatus('Disconnected', 'red');
                addMessage('Disconnected from server', 'system');
            };
        }

        function handleMessage(message) {
            switch(message.type) {
                case 'pong':
                    addMessage('Pong received: ' + message.id, 'received');
                    break;

                case 'query_response':
                    addMessage('Query Result: ' + message.payload.result, 'received');
                    addMessage('Processing time: ' + message.payload.processing_time + 's', 'received');
                    break;

                case 'stream_start':
                    addMessage('Stream started for: ' + message.payload.query, 'received');
                    break;

                case 'stream_chunk':
                    addMessage('Stream chunk: ' + message.payload.chunk, 'received');
                    break;

                case 'stream_end':
                    addMessage('Stream completed in ' + message.payload.processing_time + 's', 'received');
                    break;

                case 'server_info':
                    const info = message.payload;
                    addMessage('Server Info: ' + info.connected_servers.length + ' servers, ' + info.total_tools + ' tools', 'received');
                    break;

                case 'error':
                    addMessage('Error: ' + message.payload.error_message, 'error');
                    break;

                default:
                    addMessage('Unknown message type: ' + message.type, 'received');
            }
        }

        function sendQuery() {
            const query = document.getElementById('queryInput').value;
            if (!query || !ws) return;

            const message = {
                type: 'query',
                id: 'query-' + messageId++,
                payload: {
                    query: query,
                    context: {
                        user_id: 'web_client_001'
                    }
                }
            };

            ws.send(JSON.stringify(message));
            addMessage('Sent query: ' + query, 'sent');
            document.getElementById('queryInput').value = '';
        }

        function pingServer() {
            if (!ws) return;

            const message = {
                type: 'ping',
                id: 'ping-' + messageId++
            };

            ws.send(JSON.stringify(message));
            addMessage('Ping sent', 'sent');
        }

        function getServerInfo() {
            if (!ws) return;

            const message = {
                type: 'get_server_info',
                id: 'info-' + messageId++
            };

            ws.send(JSON.stringify(message));
            addMessage('Server info requested', 'sent');
        }

        function addMessage(text, type) {
            const messages = document.getElementById('messages');
            const div = document.createElement('div');
            div.className = 'message ' + type;
            div.textContent = new Date().toLocaleTimeString() + ': ' + text;
            messages.appendChild(div);
            messages.scrollTop = messages.scrollHeight;
        }

        function updateConnectionStatus(status, color) {
            const statusEl = document.getElementById('connectionStatus');
            statusEl.textContent = 'Status: ' + status;
            statusEl.style.color = color;
        }

        // Handle Enter key in input
        document.getElementById('queryInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendQuery();
            }
        });

        // Auto-connect on page load
        connect();
    </script>
</body>
</html>
```

---

## Node.js Client with Reconnection

```javascript
const WebSocket = require('ws');
const readline = require('readline');

class MCPWebSocketClient {
    constructor(url = 'ws://localhost:8765') {
        this.url = url;
        this.ws = null;
        this.messageId = 1;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;

        this.setupReadline();
    }

    connect() {
        console.log(`Connecting to ${this.url}...`);

        this.ws = new WebSocket(this.url);

        this.ws.on('open', () => {
            console.log('✅ Connected to MCP WebSocket server');
            this.reconnectAttempts = 0;
            this.showPrompt();
        });

        this.ws.on('message', (data) => {
            const message = JSON.parse(data.toString());
            this.handleMessage(message);
        });

        this.ws.on('error', (error) => {
            console.error('❌ WebSocket error:', error.message);
        });

        this.ws.on('close', (code, reason) => {
            console.log(`🔌 Connection closed: ${code} ${reason}`);
            this.attemptReconnect();
        });
    }

    handleMessage(message) {
        switch(message.type) {
            case 'pong':
                console.log(`🏓 Pong received: ${message.id}`);
                break;

            case 'query_response':
                console.log(`\n📥 Query Response:`);
                console.log(`Result: ${message.payload.result}`);
                console.log(`Processing time: ${message.payload.processing_time}s`);
                console.log(`Tools used: ${message.payload.tools_used.join(', ')}`);
                break;

            case 'stream_start':
                console.log(`\n🌊 Stream started: ${message.payload.query}`);
                break;

            case 'stream_chunk':
                process.stdout.write(message.payload.chunk);
                break;

            case 'stream_end':
                console.log(`\n✅ Stream completed in ${message.payload.processing_time}s`);
                break;

            case 'server_info':
                const info = message.payload;
                console.log(`\n📊 Server Info:`);
                console.log(`- Connected servers: ${info.connected_servers.length}`);
                console.log(`- Total tools: ${info.total_tools}`);
                console.log(`- Status: ${info.server_status}`);
                console.log(`- Uptime: ${info.uptime}`);
                break;

            case 'error':
                console.error(`\n❌ Error: ${message.payload.error_message}`);
                if (message.payload.details) {
                    console.error(`Details:`, message.payload.details);
                }
                break;

            default:
                console.log(`\n❓ Unknown message:`, message);
        }

        this.showPrompt();
    }

    sendMessage(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
            return true;
        } else {
            console.log('❌ Not connected to server');
            return false;
        }
    }

    sendQuery(query) {
        const message = {
            type: 'query',
            id: `query-${this.messageId++}`,
            payload: {
                query: query,
                context: {
                    client: 'node-js-client',
                    session_id: Date.now().toString()
                }
            }
        };

        this.sendMessage(message);
    }

    ping() {
        const message = {
            type: 'ping',
            id: `ping-${this.messageId++}`
        };

        this.sendMessage(message);
    }

    getServerInfo() {
        const message = {
            type: 'get_server_info',
            id: `info-${this.messageId++}`
        };

        this.sendMessage(message);
    }

    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`🔄 Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);

            setTimeout(() => {
                this.connect();
            }, this.reconnectDelay * this.reconnectAttempts);
        } else {
            console.log('❌ Max reconnection attempts reached');
        }
    }

    setupReadline() {
        this.rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout
        });

        this.rl.on('line', (input) => {
            const trimmed = input.trim();

            if (trimmed === '') {
                this.showPrompt();
                return;
            }

            if (trimmed.startsWith('/')) {
                this.handleCommand(trimmed);
            } else {
                this.sendQuery(trimmed);
            }
        });

        this.rl.on('SIGINT', () => {
            console.log('\n👋 Goodbye!');
            if (this.ws) {
                this.ws.close();
            }
            process.exit();
        });
    }

    handleCommand(command) {
        switch(command) {
            case '/ping':
                this.ping();
                break;
            case '/info':
                this.getServerInfo();
                break;
            case '/help':
                this.showHelp();
                break;
            case '/quit':
            case '/exit':
                console.log('👋 Goodbye!');
                process.exit();
                break;
            default:
                console.log(`❓ Unknown command: ${command}`);
                this.showHelp();
        }

        this.showPrompt();
    }

    showHelp() {
        console.log(`\n📚 Available commands:`);
        console.log(`/ping     - Send ping to server`);
        console.log(`/info     - Get server information`);
        console.log(`/help     - Show this help`);
        console.log(`/quit     - Exit the client`);
        console.log(`\nOr type any query to send to the MCP server\n`);
    }

    showPrompt() {
        this.rl.setPrompt('MCP> ');
        this.rl.prompt();
    }

    start() {
        console.log('🚀 MCP WebSocket Client Starting...');
        this.connect();
    }
}

// Usage
const client = new MCPWebSocketClient();
client.start();
```

---

## Python Async Client

```python
import asyncio
import websockets
import json
import sys
from datetime import datetime

class MCPWebSocketClient:
    def __init__(self, url="ws://localhost:8765"):
        self.url = url
        self.websocket = None
        self.message_id = 1
        self.running = False

    async def connect(self):
        """Connect to the WebSocket server"""
        try:
            print(f"Connecting to {self.url}...")
            self.websocket = await websockets.connect(self.url)
            print("✅ Connected to MCP WebSocket server")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False

    async def send_message(self, message):
        """Send a message to the server"""
        if self.websocket:
            await self.websocket.send(json.dumps(message))
            return True
        return False

    async def listen_for_messages(self):
        """Listen for incoming messages"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                await self.handle_message(data)
        except websockets.exceptions.ConnectionClosed:
            print("🔌 Connection closed by server")
        except Exception as e:
            print(f"❌ Error listening for messages: {e}")

    async def handle_message(self, message):
        """Handle incoming messages"""
        msg_type = message.get('type')
        timestamp = datetime.now().strftime("%H:%M:%S")

        if msg_type == 'pong':
            print(f"[{timestamp}] 🏓 Pong received: {message['id']}")

        elif msg_type == 'query_response':
            payload = message['payload']
            print(f"\n[{timestamp}] 📥 Query Response:")
            print(f"Result: {payload['result']}")
            print(f"Processing time: {payload['processing_time']}s")
            print(f"Tools used: {', '.join(payload['tools_used'])}")

        elif msg_type == 'stream_start':
            print(f"\n[{timestamp}] 🌊 Stream started: {message['payload']['query']}")

        elif msg_type == 'stream_chunk':
            print(message['payload']['chunk'], end='')

        elif msg_type == 'stream_end':
            payload = message['payload']
            print(f"\n[{timestamp}] ✅ Stream completed in {payload['processing_time']}s")

        elif msg_type == 'server_info':
            info = message['payload']
            print(f"\n[{timestamp}] 📊 Server Info:")
            print(f"- Connected servers: {len(info['connected_servers'])}")
            print(f"- Total tools: {info['total_tools']}")
            print(f"- Status: {info['server_status']}")
            print(f"- Uptime: {info['uptime']}")

        elif msg_type == 'error':
            print(f"\n[{timestamp}] ❌ Error: {message['payload']['error_message']}")
            if 'details' in message['payload']:
                print(f"Details: {message['payload']['details']}")

        else:
            print(f"\n[{timestamp}] ❓ Unknown message: {message}")

    async def send_query(self, query):
        """Send a query to the server"""
        message = {
            "type": "query",
            "id": f"query-{self.message_id}",
            "payload": {
                "query": query,
                "context": {
                    "client": "python-client",
                    "session_id": str(self.message_id)
                }
            }
        }
        self.message_id += 1
        await self.send_message(message)

    async def ping(self):
        """Send ping to server"""
        message = {
            "type": "ping",
            "id": f"ping-{self.message_id}"
        }
        self.message_id += 1
        await self.send_message(message)

    async def get_server_info(self):
        """Get server information"""
        message = {
            "type": "get_server_info",
            "id": f"info-{self.message_id}"
        }
        self.message_id += 1
        await self.send_message(message)

    async def interactive_session(self):
        """Run interactive session"""
        print("\n📚 Available commands:")
        print("/ping     - Send ping to server")
        print("/info     - Get server information")
        print("/help     - Show this help")
        print("/quit     - Exit the client")
        print("\nOr type any query to send to the MCP server\n")

        while self.running:
            try:
                user_input = await asyncio.to_thread(input, "MCP> ")
                user_input = user_input.strip()

                if user_input == "":
                    continue

                if user_input.startswith('/'):
                    await self.handle_command(user_input)
                else:
                    await self.send_query(user_input)

            except (EOFError, KeyboardInterrupt):
                break

    async def handle_command(self, command):
        """Handle user commands"""
        if command == '/ping':
            await self.ping()
        elif command == '/info':
            await self.get_server_info()
        elif command == '/help':
            print("\n📚 Available commands:")
            print("/ping     - Send ping to server")
            print("/info     - Get server information")
            print("/help     - Show this help")
            print("/quit     - Exit the client")
            print("\nOr type any query to send to the MCP server\n")
        elif command in ['/quit', '/exit']:
            self.running = False
        else:
            print(f"❓ Unknown command: {command}")

    async def run(self):
        """Main run method"""
        if not await self.connect():
            return

        self.running = True

        # Start listening for messages and interactive session
        listen_task = asyncio.create_task(self.listen_for_messages())
        interact_task = asyncio.create_task(self.interactive_session())

        try:
            # Wait for either task to complete
            done, pending = await asyncio.wait(
                [listen_task, interact_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel remaining tasks
            for task in pending:
                task.cancel()

        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            if self.websocket:
                await self.websocket.close()
            print("👋 Goodbye!")

# Usage
async def main():
    client = MCPWebSocketClient()
    await client.run()

if __name__ == "__main__":
    print("🚀 MCP WebSocket Client Starting...")
    asyncio.run(main())
```

---

## 📊 Testing with Command Line Tools

### Using websocat
```bash
# Install websocat
cargo install websocat

# Connect and send messages interactively
websocat ws://localhost:8765

# Send specific messages
echo '{"type": "ping", "id": "test-001"}' | websocat ws://localhost:8765

echo '{
  "type": "query",
  "id": "test-query",
  "payload": {
    "query": "Show database statistics"
  }
}' | websocat ws://localhost:8765

# Test streaming
echo '{
  "type": "query",
  "id": "stream-test",
  "payload": {
    "query": "Generate a comprehensive business report",
    "streaming": true
  }
}' | websocat ws://localhost:8765
```

### Using wscat (Node.js)
```bash
# Install wscat
npm install -g wscat

# Connect interactively
wscat -c ws://localhost:8765

# Send ping
{"type": "ping", "id": "test-001"}

# Send query
{
  "type": "query",
  "id": "test-query",
  "payload": {
    "query": "List all database tables"
  }
}

# Get server info
{"type": "server_info", "id": "info-001"}
```

### Batch Testing Script
```bash
#!/bin/bash

# Batch WebSocket testing script
WS_URL="ws://localhost:8765"

echo "Testing MCP WebSocket Server..."

# Test 1: Ping
echo "1. Testing ping/pong..."
echo '{"type": "ping", "id": "batch-ping-001"}' | websocat $WS_URL

# Test 2: Server info
echo "2. Getting server info..."
echo '{"type": "server_info", "id": "batch-info-001"}' | websocat $WS_URL

# Test 3: Simple query
echo "3. Testing simple query..."
echo '{
  "type": "query",
  "id": "batch-query-001",
  "payload": {
    "query": "What is the current time?"
  }
}' | websocat $WS_URL

# Test 4: Database query
echo "4. Testing database query..."
echo '{
  "type": "query",
  "id": "batch-query-002",
  "payload": {
    "query": "Show total number of customers"
  }
}' | websocat $WS_URL

# Test 5: Tool call
echo "5. Testing direct tool call..."
echo '{
  "type": "tool_call",
  "id": "batch-tool-001",
  "payload": {
    "tool_name": "get_database_statistics",
    "arguments": {}
  }
}' | websocat $WS_URL

echo "Batch testing complete!"
```

---

## ❌ Error Handling

### Connection Errors
```javascript
ws.onerror = function(error) {
    console.error('WebSocket error:', error);
    // Implement reconnection logic
    setTimeout(connect, 5000);
};

ws.onclose = function(event) {
    if (event.code !== 1000) { // Not normal closure
        console.log('Unexpected close, attempting reconnection...');
        setTimeout(connect, 2000);
    }
};
```

### Message Validation
```javascript
function validateMessage(message) {
    if (!message.type) {
        throw new Error('Message must have a type field');
    }

    if (!message.id) {
        throw new Error('Message must have an id field');
    }

    // Additional validation based on message type
    switch(message.type) {
        case 'query':
            if (!message.payload || !message.payload.query) {
                throw new Error('Query message must have payload.query');
            }
            break;
        // Add more validation as needed
    }
}
```

### Timeout Handling
```javascript
class WebSocketClient {
    constructor() {
        this.pendingRequests = new Map();
        this.requestTimeout = 30000; // 30 seconds
    }

    async sendRequestWithTimeout(message) {
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                this.pendingRequests.delete(message.id);
                reject(new Error(`Request ${message.id} timed out`));
            }, this.requestTimeout);

            this.pendingRequests.set(message.id, {
                resolve,
                reject,
                timeout
            });

            this.ws.send(JSON.stringify(message));
        });
    }

    handleResponse(response) {
        const pending = this.pendingRequests.get(response.id);
        if (pending) {
            clearTimeout(pending.timeout);
            this.pendingRequests.delete(response.id);
            pending.resolve(response);
        }
    }
}
```

---

## 🎯 Best Practices

### 1. Connection Management
```javascript
class RobustWebSocketClient {
    constructor(url) {
        this.url = url;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.heartbeatInterval = null;
    }

    connect() {
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
            this.reconnectAttempts = 0;
            this.startHeartbeat();
        };

        this.ws.onclose = (event) => {
            this.stopHeartbeat();
            if (event.code !== 1000) {
                this.attemptReconnect();
            }
        };
    }

    startHeartbeat() {
        this.heartbeatInterval = setInterval(() => {
            if (this.ws.readyState === WebSocket.OPEN) {
                this.ping();
            }
        }, 30000); // Ping every 30 seconds
    }

    stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
    }

    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
            setTimeout(() => this.connect(), delay);
        }
    }
}
```

### 2. Message Queuing
```javascript
class QueuedWebSocketClient {
    constructor(url) {
        this.url = url;
        this.messageQueue = [];
        this.connected = false;
    }

    send(message) {
        if (this.connected && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
        } else {
            this.messageQueue.push(message);
        }
    }

    onConnected() {
        this.connected = true;
        // Send queued messages
        while (this.messageQueue.length > 0) {
            const message = this.messageQueue.shift();
            this.ws.send(JSON.stringify(message));
        }
    }
}
```

### 3. Response Correlation
```javascript
class CorrelatedWebSocketClient {
    constructor(url) {
        this.url = url;
        this.pendingRequests = new Map();
        this.messageId = 1;
    }

    async sendQuery(query) {
        const id = `query-${this.messageId++}`;
        const message = {
            type: 'query',
            id: id,
            payload: { query }
        };

        return new Promise((resolve, reject) => {
            this.pendingRequests.set(id, { resolve, reject });
            this.ws.send(JSON.stringify(message));

            // Set timeout
            setTimeout(() => {
                if (this.pendingRequests.has(id)) {
                    this.pendingRequests.delete(id);
                    reject(new Error('Request timeout'));
                }
            }, 30000);
        });
    }

    handleMessage(message) {
        if (message.type === 'response' || message.type === 'error') {
            const pending = this.pendingRequests.get(message.id);
            if (pending) {
                this.pendingRequests.delete(message.id);
                if (message.type === 'response') {
                    pending.resolve(message.payload);
                } else {
                    pending.reject(new Error(message.payload.error_message));
                }
            }
        }
    }
}
```

---

**⚡ Your WebSocket interface is ready for real-time AI interactions!**