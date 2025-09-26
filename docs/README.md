# 📚 MCP Client Documentation

Professional documentation for the **MCP Client** - A production-ready Python application with three interfaces: Interactive Chatbot, REST API, and WebSocket server.

---

## 📖 Documentation Structure

### 🚀 Getting Started
- **[Complete Setup Guide](user-guide/GETTING_STARTED.md)** - Comprehensive installation and setup for Windows, macOS, and Linux
  - Prerequisites and system requirements
  - Platform-specific installation steps
  - Environment configuration
  - Service startup commands
  - Verification and testing procedures
  - Troubleshooting guide

### 🌐 REST API Reference
- **[API Reference Guide](api-reference/API_REFERENCE.md)** - Complete REST API documentation with examples
  - All endpoints with detailed descriptions
  - curl examples for every endpoint
  - Request/response formats
  - Error handling and status codes
  - Rate limiting information
  - Complete implementation examples

### ⚡ WebSocket Reference
- **[WebSocket Reference Guide](api-reference/WEBSOCKET_REFERENCE.md)** - Real-time communication interface
  - Message protocol specification
  - All message types with examples
  - JavaScript, Node.js, and Python client implementations
  - Streaming response handling
  - Connection management and reconnection
  - Error handling and best practices

### 🏗️ Developer Guide
- **[Architecture Guide](developer-guide/ARCHITECTURE.md)** - Technical architecture for developers
  - Project structure and organization
  - Core components and interfaces
  - AI enhancement system design
  - Configuration management
  - Development guidelines and best practices
  - Deployment architecture

---

## 🎯 Quick Navigation

### For Developers
- **New to MCP Client?** → Start with [Getting Started Guide](user-guide/GETTING_STARTED.md)
- **Building web apps?** → Check out [REST API Reference](api-reference/API_REFERENCE.md)
- **Need real-time features?** → See [WebSocket Reference](api-reference/WEBSOCKET_REFERENCE.md)
- **Contributing to the project?** → Read [Architecture Guide](developer-guide/ARCHITECTURE.md)

### By Platform
- **Windows Users** → [Windows Setup Section](user-guide/GETTING_STARTED.md#windows-setup)
- **macOS Users** → [macOS Setup Section](user-guide/GETTING_STARTED.md#macos-setup)
- **Linux Users** → [Linux Setup Section](user-guide/GETTING_STARTED.md#linux-setup)

### By Interface
- **Chatbot Interface** → [Getting Started - Starting Services](user-guide/GETTING_STARTED.md#starting-services)
- **REST API** → [API Reference Guide](api-reference/API_REFERENCE.md)
- **WebSocket** → [WebSocket Reference Guide](api-reference/WEBSOCKET_REFERENCE.md)

---

## 🌟 Key Features Covered

### 🎯 **Triple Interface Architecture**
- **🤖 Interactive Chatbot**: Rich terminal interface with AI enhancements
- **🌐 REST API**: Professional HTTP endpoints for web integration
- **⚡ WebSocket**: Real-time bidirectional communication server

### 🧠 **AI-Powered Intelligence**
- **Chain-of-Thought Reasoning**: Advanced problem-solving capabilities
- **Context Management**: Intelligent conversation context optimization
- **Quality Enhancement**: Response optimization and validation
- **Performance Tracking**: Real-time AI performance monitoring

### 🏗️ **Enterprise Architecture**
- **Multi-Server Support**: Connect to multiple MCP servers simultaneously
- **Oracle Database Integration**: Built-in database tools and operations
- **Production Monitoring**: Health checks, logging, and error handling
- **Cross-Platform Compatibility**: Windows, macOS, and Linux support

---

## 📋 Quick Reference

### Installation Commands
```bash
# Clone repository
git clone <repository-url>
cd MCP-CLIENT

# Install dependencies
pip install -e .

# Set up environment
cp .env.example .env
mkdir -p logs

# Start all interfaces
mcp-client --interfaces all --api-port 8000 --ws-port 8765
```

### Service URLs
- **Chatbot**: Terminal interface (run `mcp-client --interfaces chatbot`)
- **REST API**: http://127.0.0.1:8000
- **WebSocket**: ws://localhost:8765
- **API Docs**: http://127.0.0.1:8000/docs

### Essential Commands
```bash
# Individual interfaces
mcp-client --interfaces chatbot
mcp-client --interfaces rest_api --api-port 8000
mcp-client --interfaces websocket --ws-port 8765

# All interfaces together
mcp-client --interfaces all --api-port 8000 --ws-port 8765

# With debug mode
mcp-client --interfaces all --debug
```

### Test Commands
```bash
# Test REST API
curl http://127.0.0.1:8000/
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello, what can you do?"}'

# Test WebSocket
echo '{"type": "ping", "id": "test-001"}' | websocat ws://localhost:8765
```

---

## 🛠️ Configuration Files

### Environment Variables (`.env`)
```env
# AI Provider - Set at least one
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here
GEMINI_API_KEY=your_gemini_key_here

# Oracle Database (for database features)
ORACLE_HOST=localhost
ORACLE_PORT=1521
ORACLE_SERVICE_NAME=XE
ORACLE_USERNAME=your_username
ORACLE_PASSWORD=your_password

# AI Enhancements
ENABLE_CHAIN_OF_THOUGHT=true
ENABLE_INTELLIGENT_CONTEXT=true
ENABLE_QUALITY_OPTIMIZATION=true
ENABLE_PERFORMANCE_TRACKING=true
```

### MCP Servers (`configs/mcp_servers.json`)
```json
{
  "mcpServers": {
    "oracle-db": {
      "command": "java",
      "args": ["-jar", "path/to/mcp-oracledb-server.jar"],
      "enabled": true,
      "env": {
        "ORACLE_HOST": "${ORACLE_HOST}",
        "ORACLE_USERNAME": "${ORACLE_USERNAME}",
        "ORACLE_PASSWORD": "${ORACLE_PASSWORD}"
      }
    }
  }
}
```

---

## 🔧 Troubleshooting Quick Reference

### Common Issues
| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: mcp_client` | Run `pip install -e .` |
| `Port already in use` | Use different ports: `--api-port 8001 --ws-port 8766` |
| `Permission denied (Windows)` | Run as Administrator or disable file logging |
| `Oracle connection failed` | Check database is running and credentials in `.env` |
| `API key invalid` | Verify API keys in `.env` file |

### Debug Commands
```bash
# Enable debug mode
mcp-client --interfaces all --debug --log-level DEBUG

# Check logs
tail -f logs/mcp_client.log        # Linux/macOS
type logs\mcp_client.log           # Windows

# Test configuration
python -c "from mcp_client.core import Config; print('Config OK')"
```

---

## 📊 API Quick Reference

### REST API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API status and information |
| `POST` | `/query` | Process natural language queries |
| `POST` | `/tool/call` | Direct tool execution |
| `GET` | `/server/info` | Connected servers information |
| `GET` | `/server/health` | Server health status |
| `GET` | `/tools/list` | List all available tools |

### WebSocket Message Types
| Type | Direction | Description |
|------|-----------|-------------|
| `ping` | Client → Server | Connection health check |
| `pong` | Server → Client | Ping response |
| `query` | Client → Server | Natural language query |
| `response` | Server → Client | Query result |
| `stream_*` | Server → Client | Streaming response chunks |
| `tool_call` | Client → Server | Direct tool execution |
| `error` | Server → Client | Error notification |

---

## 🎯 Examples Library

### REST API Examples
```bash
# Basic query
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Show database statistics"}'

# Business intelligence
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Generate loan portfolio analysis"}'

# Direct tool call
curl -X POST http://127.0.0.1:8000/tool/call \
  -H "Content-Type: application/json" \
  -d '{"tool_name": "execute_sql_query", "arguments": {"query": "SELECT COUNT(*) FROM customers"}}'
```

### WebSocket Examples
```javascript
// JavaScript client
const ws = new WebSocket('ws://localhost:8765');
ws.send(JSON.stringify({
  type: 'query',
  id: 'test-001',
  payload: {query: 'Analyze customer data'}
}));

// Python client
import asyncio
import websockets
import json

async def query_mcp():
    async with websockets.connect("ws://localhost:8765") as websocket:
        await websocket.send(json.dumps({
            "type": "query",
            "id": "python-001",
            "payload": {"query": "Show database schema"}
        }))
        response = await websocket.recv()
        print(json.loads(response))

asyncio.run(query_mcp())
```

---

## 🚀 Production Deployment

### Performance Considerations
- **Memory**: ~50MB per interface
- **CPU**: Low usage for typical queries
- **Concurrent Users**:
  - Chatbot: 1 (terminal)
  - REST API: 100+ (configurable)
  - WebSocket: 50+ connections

### Security Checklist
- ✅ Configure environment variables securely
- ✅ Enable rate limiting in production
- ✅ Set up proper logging and monitoring
- ✅ Use HTTPS/WSS in production
- ✅ Validate all input data
- ✅ Monitor for security vulnerabilities

### Scaling Options
- **Horizontal**: Run multiple instances behind a load balancer
- **Vertical**: Increase memory and CPU resources
- **Database**: Use connection pooling for Oracle database
- **Caching**: Implement Redis for response caching

---

## 🎉 Success Metrics

After following this documentation, you should have:

✅ **All three interfaces running** - Chatbot, REST API, and WebSocket
✅ **AI enhancements active** - Chain-of-thought, context optimization
✅ **Database integration working** - Oracle DB with 93+ tools
✅ **Professional monitoring** - Health checks, logging, error handling
✅ **Cross-platform compatibility** - Working on Windows, macOS, Linux
✅ **Production-ready deployment** - Scalable and secure configuration

---

## 📞 Support & Resources

### Documentation
- **Getting Started**: [Complete setup guide](user-guide/GETTING_STARTED.md)
- **API Reference**: [REST API documentation](api-reference/API_REFERENCE.md)
- **WebSocket Guide**: [Real-time interface docs](api-reference/WEBSOCKET_REFERENCE.md)
- **Architecture Guide**: [Technical architecture](developer-guide/ARCHITECTURE.md)

### Interactive Documentation
- **Swagger UI**: http://127.0.0.1:8000/docs (when REST API is running)
- **ReDoc**: http://127.0.0.1:8000/redoc (alternative API docs)

### Community
- **Issues**: Report bugs and request features via GitHub Issues
- **Discussions**: Join community discussions for best practices
- **Contributing**: Follow standard GitHub contribution workflow

---

**🚀 Your MCP Client is ready for professional use with advanced AI capabilities!**

*Built with ❤️ using modern Python practices and AI-enhanced architecture*