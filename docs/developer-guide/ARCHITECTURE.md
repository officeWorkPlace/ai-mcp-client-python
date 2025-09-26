# 🏗️ MCP Client Architecture Guide

Technical architecture documentation for developers working on the MCP Client codebase.

---

## 📋 Table of Contents

- [Project Structure](#project-structure)
- [Architecture Overview](#architecture-overview)
- [Core Components](#core-components)
- [Interface Layer](#interface-layer)
- [Service Layer](#service-layer)
- [AI Enhancement System](#ai-enhancement-system)
- [Configuration Management](#configuration-management)
- [Development Guidelines](#development-guidelines)

---

## 📁 Project Structure

### Modern src/ Layout (Current)
```
src/mcp_client/                    # Main package following modern Python standards
├── __init__.py                   # Package exports
├── __main__.py                   # CLI entry point
├── interfaces/                   # User interfaces
│   ├── __init__.py
│   ├── chatbot.py               # 🤖 Interactive terminal interface
│   ├── rest_api.py              # 🌐 HTTP API server (FastAPI)
│   └── websocket.py             # ⚡ Real-time WebSocket server
├── services/                     # Service layer
│   ├── __init__.py
│   ├── mcp_service.py           # Shared MCP service
│   └── interface_coordinator.py  # Interface management
├── core/                         # Core functionality
│   ├── __init__.py
│   ├── client.py                # Main MCP client
│   ├── config.py                # Configuration management
│   ├── logger.py                # Logging setup
│   └── exceptions.py            # Custom exceptions
├── ai/                          # AI enhancements
│   ├── __init__.py
│   ├── context/
│   │   ├── __init__.py
│   │   └── context_manager.py   # Context optimization
│   ├── reasoning/
│   │   ├── __init__.py
│   │   └── cot_engine.py        # Chain-of-thought reasoning
│   ├── quality/
│   │   ├── __init__.py
│   │   └── response_optimizer.py # Response enhancement
│   ├── monitoring/
│   │   ├── __init__.py
│   │   └── performance_tracker.py # Performance monitoring
│   └── metacognition/
│       ├── __init__.py
│       └── meta_engine.py       # Meta-cognitive processing
├── utils/                       # Utilities
│   ├── __init__.py
│   ├── validators.py            # Input validation
│   ├── rate_limiter.py          # Rate limiting
│   └── helpers.py               # Helper functions
└── servers/                     # Example MCP servers
    ├── __init__.py
    ├── calculator_server.py     # Calculator server example
    └── weather_server.py        # Weather server example
```

### Legacy Structure (Deprecated)
```
global_mcp_client/                # Legacy package (being phased out)
├── main.py                      # Old main entry point
├── chatbot.py                   # Legacy chatbot
├── core/                        # Legacy core components
└── ...                          # Other legacy modules
```

---

## 🏛️ Architecture Overview

### High-Level Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    User Interfaces                         │
├─────────────────┬─────────────────┬─────────────────────────┤
│   🤖 Chatbot    │   🌐 REST API   │   ⚡ WebSocket          │
│   (Terminal)    │   (HTTP/JSON)   │   (Real-time)           │
└─────────────────┴─────────────────┴─────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────┐
│               Interface Coordinator                         │
│            (Manages multiple interfaces)                    │
└─────────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────┐
│                   MCP Service Layer                         │
│         (Shared business logic and MCP client)              │
└─────────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────┐
│                 AI Enhancement System                       │
├─────────────────┬─────────────────┬─────────────────────────┤
│ Context Manager │ CoT Engine      │ Quality Optimizer       │
│ Performance     │ Meta Engine     │ Response Enhancement    │
└─────────────────┴─────────────────┴─────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────┐
│                    MCP Servers                              │
├─────────────────┬─────────────────┬─────────────────────────┤
│   Oracle DB     │   Filesystem    │   Memory/Notes          │
│   (93+ tools)   │   Operations    │   Management            │
└─────────────────┴─────────────────┴─────────────────────────┘
```

### Data Flow
1. **User Input** → Interface (Chatbot/API/WebSocket)
2. **Interface** → Interface Coordinator
3. **Coordinator** → MCP Service
4. **MCP Service** → AI Enhancement System
5. **AI System** → MCP Servers (Oracle DB, etc.)
6. **MCP Servers** → Business Logic Processing
7. **Response** → AI Enhancement → Interface → User

---

## 🔧 Core Components

### 1. Interface Coordinator (`interface_coordinator.py`)
**Purpose**: Manages multiple interfaces simultaneously

```python
class InterfaceCoordinator:
    """Coordinates multiple MCP client interfaces"""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.mcp_service = MCPService(self.config)

        # Interface instances
        self.chatbot: Optional[MCPChatBot] = None
        self.rest_api: Optional[MCPRestAPI] = None
        self.websocket: Optional[MCPWebSocketServer] = None
```

**Key Features**:
- Unified configuration management
- Shared MCP service instance
- Graceful shutdown handling
- Interface lifecycle management

### 2. MCP Service (`mcp_service.py`)
**Purpose**: Core business logic and MCP server communication

**Responsibilities**:
- MCP server connection management
- Tool execution coordination
- Response processing
- Error handling and recovery

### 3. Global MCP Client (`core/client.py`)
**Purpose**: Low-level MCP protocol handling

**Features**:
- Multi-server connection support
- Tool discovery and execution
- Health monitoring
- Automatic reconnection

---

## 🎭 Interface Layer

### 1. Chatbot Interface (`interfaces/chatbot.py`)
**Technology**: Rich Terminal UI
**Entry Point**: `mcp-chatbot` or `mcp-client --interfaces chatbot`

```python
class MCPChatBot:
    """Modern chatbot interface for MCP Client"""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.console = Console()  # Rich terminal
        self.validator = InputValidator()
        self.rate_limiter = RateLimiter()
```

**Features**:
- Rich terminal interface with colors and formatting
- Session statistics tracking
- Input validation and rate limiting
- Professional banner and help system

### 2. REST API Interface (`interfaces/rest_api.py`)
**Technology**: FastAPI + Uvicorn
**Entry Point**: `mcp-api` or `mcp-client --interfaces rest_api`

```python
@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest) -> QueryResponse:
    """Process natural language queries"""
    start_time = time.time()

    try:
        # Process query through MCP service
        result = await mcp_service.process_query(request.query, request.context)

        return QueryResponse(
            query=request.query,
            response=result.response,
            processing_time=time.time() - start_time,
            tools_used=result.tools_used,
            timestamp=time.time()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Endpoints**:
- `GET /` - API status
- `POST /query` - Process queries
- `POST /tool/call` - Direct tool execution
- `GET /server/info` - Server information
- `GET /server/health` - Health checks
- `GET /tools/list` - Available tools

### 3. WebSocket Interface (`interfaces/websocket.py`)
**Technology**: Python websockets
**Entry Point**: `mcp-websocket` or `mcp-client --interfaces websocket`

```python
class MCPWebSocketServer:
    """WebSocket server for real-time MCP communication"""

    async def handle_client(self, websocket, path):
        """Handle individual WebSocket connections"""
        self.clients.add(websocket)
        try:
            async for message in websocket:
                await self.process_message(websocket, json.loads(message))
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
```

**Message Types**:
- `ping/pong` - Connection health
- `query` - Natural language queries
- `tool_call` - Direct tool execution
- `stream_*` - Streaming responses
- `server_info` - Server information

---

## 🤖 AI Enhancement System

### 1. Context Manager (`ai/context/context_manager.py`)
**Purpose**: Intelligent conversation context optimization

```python
class ContextManager:
    """Manages conversation context and optimization"""

    async def optimize_context(self, query: str, history: List[str]) -> str:
        """Optimize query context for better AI responses"""
        # Context analysis and optimization logic
        pass
```

### 2. Chain-of-Thought Engine (`ai/reasoning/cot_engine.py`)
**Purpose**: Advanced reasoning and problem-solving

```python
class CoTEngine:
    """Chain-of-thought reasoning engine"""

    async def enhance_reasoning(self, query: str) -> Dict[str, Any]:
        """Apply chain-of-thought reasoning to complex queries"""
        # Step-by-step reasoning logic
        pass
```

### 3. Quality Optimizer (`ai/quality/response_optimizer.py`)
**Purpose**: Response enhancement and validation

```python
class ResponseOptimizer:
    """Optimizes and validates AI responses"""

    async def optimize_response(self, response: str, query: str) -> Dict[str, Any]:
        """Optimize response quality and format"""
        # Response optimization logic
        pass
```

### 4. Performance Tracker (`ai/monitoring/performance_tracker.py`)
**Purpose**: Real-time performance monitoring

```python
class PerformanceTracker:
    """Tracks AI performance metrics"""

    async def track_query(self, query: str, response_time: float, tools_used: List[str]):
        """Track query performance metrics"""
        # Performance tracking logic
        pass
```

---

## ⚙️ Configuration Management

### Configuration Sources (Priority Order)
1. **Command-line arguments** (highest priority)
2. **Environment variables** (`.env` file)
3. **Configuration files** (`configs/mcp_servers.json`)
4. **Default values** (lowest priority)

### Configuration Classes
```python
class Config:
    """Main configuration class"""

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = config_dir
        self.load_environment()
        self.load_mcp_servers()
        self.validate()

    def load_environment(self):
        """Load environment variables from .env file"""
        pass

    def load_mcp_servers(self):
        """Load MCP server configurations"""
        pass
```

### Environment Variables
```env
# AI Provider Configuration
ANTHROPIC_API_KEY=sk-ant-api03-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...

# Database Configuration
ORACLE_HOST=localhost
ORACLE_PORT=1521
ORACLE_USERNAME=...
ORACLE_PASSWORD=...

# AI Enhancements
ENABLE_CHAIN_OF_THOUGHT=true
ENABLE_INTELLIGENT_CONTEXT=true
ENABLE_QUALITY_OPTIMIZATION=true
ENABLE_PERFORMANCE_TRACKING=true
```

---

## 🛠️ Development Guidelines

### 1. Code Organization
- **Separation of Concerns**: Each module has a single responsibility
- **Dependency Injection**: Use configuration objects for dependencies
- **Async/Await**: All I/O operations are asynchronous
- **Type Hints**: Full type annotations for better IDE support

### 2. Error Handling
```python
from ..core.exceptions import GlobalMCPClientError

class CustomError(GlobalMCPClientError):
    """Custom error for specific scenarios"""
    pass

try:
    result = await some_operation()
except GlobalMCPClientError as e:
    logger.error(f"MCP operation failed: {e}")
    # Handle gracefully
except Exception as e:
    logger.exception(f"Unexpected error: {e}")
    # Log and re-raise or handle
```

### 3. Logging Standards
```python
from ..core.logger import setup_logging

logger = setup_logging(
    log_level="INFO",
    log_file="logs/component.log",
    enable_file_logging=True
)

# Use structured logging
logger.info("Operation completed", extra={
    "operation": "query_processing",
    "duration": 1.23,
    "tools_used": ["sql_query", "format_result"]
})
```

### 4. Testing Guidelines
- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test interface interactions
- **API Tests**: Test REST API endpoints
- **WebSocket Tests**: Test real-time communication

### 5. Performance Considerations
- **Connection Pooling**: Reuse database connections
- **Caching**: Cache frequent queries and tool results
- **Rate Limiting**: Prevent API abuse
- **Memory Management**: Clean up resources properly

---

## 🔄 Entry Points and CLI

### Package Entry Points (pyproject.toml)
```toml
[project.scripts]
# Main entry points
mcp-client = "mcp_client.__main__:main"
mcp = "mcp_client.__main__:main"

# Interface-specific entry points
mcp-chatbot = "mcp_client.interfaces.chatbot:cli_main"
mcp-api = "mcp_client.interfaces.rest_api:main"
mcp-websocket = "mcp_client.interfaces.websocket:main"

# Service entry points
mcp-coordinator = "mcp_client.services.interface_coordinator:main"
```

### CLI Command Structure
```bash
# Main coordinator command
mcp-client --interfaces chatbot rest_api websocket --api-port 8000 --ws-port 8765

# Individual interface commands
mcp-chatbot                    # Direct chatbot access
mcp-api --port 8000           # Direct API access
mcp-websocket --port 8765     # Direct WebSocket access
```

---

## 🚀 Deployment Architecture

### Development Environment
```bash
# Install in development mode
pip install -e .

# Run with debug
mcp-client --interfaces all --debug
```

### Production Environment
```bash
# Install from package
pip install mcp-client

# Run with production settings
mcp-client --interfaces all --api-port 8000 --ws-port 8765
```

### Container Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install .

EXPOSE 8000 8765

CMD ["mcp-client", "--interfaces", "all", "--api-port", "8000", "--ws-port", "8765"]
```

---

## 📊 Monitoring and Observability

### Health Checks
- **API Health**: `GET /server/health`
- **WebSocket Health**: Ping/Pong messages
- **MCP Server Health**: Connection status monitoring

### Metrics Collection
- **Request Counts**: Total, successful, failed requests
- **Response Times**: Average, P95, P99 response times
- **Tool Usage**: Most used tools, success rates
- **Connection Stats**: Active connections, reconnections

### Logging Strategy
- **Structured Logging**: JSON format for easy parsing
- **Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Log Rotation**: Automatic log file rotation
- **Centralized Logging**: Support for external log aggregators

---

**🏗️ Your MCP Client architecture is designed for scalability, maintainability, and professional deployment!**