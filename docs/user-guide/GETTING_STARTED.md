# 🚀 MCP Client - Complete Setup Guide

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Cross-Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)](README.md)

A **production-ready Python application** for connecting to multiple Model Context Protocol (MCP) servers with **three professional interfaces**: Interactive Chatbot, REST API, and Real-time WebSocket server.

---

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Installation](#quick-installation)
- [Platform-Specific Setup](#platform-specific-setup)
  - [Windows Setup](#windows-setup)
  - [macOS Setup](#macos-setup)
  - [Linux Setup](#linux-setup)
- [Configuration](#configuration)
- [Starting Services](#starting-services)
- [Verification & Testing](#verification--testing)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Prerequisites

### System Requirements
- **Python 3.8 or higher** (Python 3.9+ recommended)
- **pip** package manager (included with Python)
- **Git** (for cloning the repository)
- **4GB RAM minimum** (8GB recommended for production)
- **1GB free disk space**

### Optional but Recommended
- **uv** - Fast Python package manager: [Installation Guide](https://docs.astral.sh/uv/getting-started/installation/)
- **Oracle Database** - For database integration features
- **Node.js 18+** - For certain MCP servers (filesystem, memory, etc.)

---

## ⚡ Quick Installation

### 1. Clone the Repository
```bash
# Clone the repository
git clone <repository-url>
cd MCP-CLIENT

# Verify you're in the correct directory
ls -la  # Should show pyproject.toml, src/, configs/, etc.
```

### 2. Install Dependencies
```bash
# Option A: Using uv (recommended - faster)
uv sync

# Option B: Using pip
pip install -e .

# Option C: Install from requirements
pip install -r requirements.txt
```

### 3. Set Up Environment
```bash
# Copy environment template
cp .env.example .env

# Create logs directory
mkdir -p logs
```

### 4. Configure Environment
Edit `.env` file with your API keys:
```env
# At minimum, set one AI provider
ANTHROPIC_API_KEY=your_anthropic_key_here
# OR
OPENAI_API_KEY=your_openai_key_here
# OR
GEMINI_API_KEY=your_gemini_key_here

# Oracle Database (if using database features)
ORACLE_HOST=localhost
ORACLE_PORT=1521
ORACLE_SERVICE_NAME=XE
ORACLE_USERNAME=your_username
ORACLE_PASSWORD=your_password
```

### 5. Start the Application
```bash
# Start interactive chatbot
mcp-client --interfaces chatbot

# Or start all interfaces
mcp-client --interfaces all --api-port 8000 --ws-port 8765
```

---

## 🖥️ Platform-Specific Setup

## Windows Setup

### Prerequisites Installation
```cmd
# 1. Install Python (if not already installed)
# Download from: https://www.python.org/downloads/windows/
# ✅ IMPORTANT: Check "Add Python to PATH" during installation

# 2. Verify Python installation
python --version
pip --version

# 3. Install Git (if not already installed)
# Download from: https://git-scm.com/download/win

# 4. Optional: Install uv for faster package management
# Download from: https://docs.astral.sh/uv/getting-started/installation/
```

### Setup Steps
```cmd
# 1. Clone the repository
git clone <repository-url>
cd MCP-CLIENT

# 2. Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -e .

# 4. Set up environment
copy .env.example .env
mkdir logs

# 5. Edit .env file
notepad .env
REM Add your API keys and database configuration

# 6. Verify installation
mcp-client --help
```

### Start Services on Windows
```cmd
# Start chatbot interface
mcp-client --interfaces chatbot

# Start REST API (available at http://127.0.0.1:8000)
mcp-client --interfaces rest_api --api-port 8000

# Start WebSocket server (available at ws://localhost:8765)
mcp-client --interfaces websocket --ws-port 8765

# Start all interfaces together
mcp-client --interfaces all --api-port 8000 --ws-port 8765
```

---

## macOS Setup

### Prerequisites Installation
```bash
# 1. Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install Python
brew install python3

# 3. Install Git
brew install git

# 4. Optional: Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

### Setup Steps
```bash
# 1. Clone the repository
git clone <repository-url>
cd MCP-CLIENT

# 2. Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -e .

# 4. Set up environment
cp .env.example .env
mkdir -p logs

# 5. Edit .env file
vim .env  # or nano .env, or open -e .env
# Add your API keys and database configuration

# 6. Verify installation
mcp-client --help
```

### Start Services on macOS
```bash
# Start chatbot interface
mcp-client --interfaces chatbot

# Start REST API (available at http://127.0.0.1:8000)
mcp-client --interfaces rest_api --api-port 8000

# Start WebSocket server (available at ws://localhost:8765)
mcp-client --interfaces websocket --ws-port 8765

# Start all interfaces together
mcp-client --interfaces all --api-port 8000 --ws-port 8765
```

---

## Linux Setup

### Prerequisites Installation

#### Ubuntu/Debian
```bash
# Update package list
sudo apt update

# Install Python and pip
sudo apt install python3 python3-pip python3-venv git

# Optional: Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

#### CentOS/RHEL/Fedora
```bash
# Install Python and pip
sudo dnf install python3 python3-pip git  # Fedora
# OR
sudo yum install python3 python3-pip git  # CentOS/RHEL

# Optional: Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

#### Arch Linux
```bash
# Install Python and pip
sudo pacman -S python python-pip git

# Optional: Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

### Setup Steps
```bash
# 1. Clone the repository
git clone <repository-url>
cd MCP-CLIENT

# 2. Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -e .

# 4. Set up environment
cp .env.example .env
mkdir -p logs

# 5. Edit .env file
vim .env  # or nano .env
# Add your API keys and database configuration

# 6. Verify installation
mcp-client --help
```

### Start Services on Linux
```bash
# Start chatbot interface
mcp-client --interfaces chatbot

# Start REST API (available at http://127.0.0.1:8000)
mcp-client --interfaces rest_api --api-port 8000

# Start WebSocket server (available at ws://localhost:8765)
mcp-client --interfaces websocket --ws-port 8765

# Start all interfaces together
mcp-client --interfaces all --api-port 8000 --ws-port 8765
```

---

## ⚙️ Configuration

### Environment Variables (.env)

Create and edit your `.env` file:

```env
# =================================
# AI Provider Configuration
# =================================
# Set at least one AI provider API key
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
OPENAI_API_KEY=sk-your-openai-key-here
GEMINI_API_KEY=your-gemini-key-here

# Default AI Model Settings
DEFAULT_MODEL=claude-3-7-sonnet-20250219
MAX_TOKENS=4096
TEMPERATURE=0.1

# =================================
# Oracle Database Configuration
# =================================
# Required for database features
ORACLE_HOST=localhost
ORACLE_PORT=1521
ORACLE_SERVICE_NAME=XE
ORACLE_USERNAME=your_oracle_username
ORACLE_PASSWORD=your_oracle_password

# =================================
# Logging Configuration
# =================================
LOG_LEVEL=INFO
LOG_FILE=logs/global_mcp_client.log
ENABLE_FILE_LOGGING=true
ENABLE_CONSOLE_LOGGING=true

# =================================
# MCP Client Configuration
# =================================
MCP_TIMEOUT=30
MCP_RETRY_ATTEMPTS=3
MCP_RETRY_DELAY=1

# =================================
# WebSocket Configuration (Optional)
# =================================
WEBSOCKET_HOST=localhost
WEBSOCKET_PORT=8765
ENABLE_WEBSOCKET=false

# =================================
# AI Enhancement Features
# =================================
ENABLE_CHAIN_OF_THOUGHT=true
ENABLE_INTELLIGENT_CONTEXT=true
ENABLE_QUALITY_OPTIMIZATION=true
ENABLE_PERFORMANCE_TRACKING=true

# =================================
# Security Configuration
# =================================
ENABLE_RATE_LIMITING=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600

# =================================
# Development Settings
# =================================
DEBUG=false
DEVELOPMENT_MODE=false
```

### MCP Server Configuration (configs/mcp_servers.json)

The application comes with several pre-configured MCP servers. Edit `configs/mcp_servers.json` to enable/disable servers:

```json
{
  "mcpServers": {
    "oracle-db": {
      "command": "java",
      "args": ["-jar", "path/to/your/mcp-oracledb-server.jar", "--spring.profiles.active=mcp-run"],
      "description": "Oracle Database MCP Server with 93+ tools",
      "enabled": true,
      "timeout": 45,
      "retry_attempts": 3,
      "cwd": "path/to/your/oracle/server/directory",
      "env": {
        "ORACLE_HOST": "${ORACLE_HOST}",
        "ORACLE_PORT": "${ORACLE_PORT}",
        "ORACLE_SID": "XE",
        "ORACLE_USERNAME": "${ORACLE_USERNAME}",
        "ORACLE_PASSWORD": "${ORACLE_PASSWORD}",
        "MCP_TOOLS_EXPOSURE": "public"
      }
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
      "description": "File system operations server",
      "enabled": false,
      "timeout": 30,
      "retry_attempts": 3
    }
  },
  "global_settings": {
    "max_concurrent_connections": 10,
    "connection_timeout": 30,
    "tool_call_timeout": 60,
    "enable_health_checks": true,
    "auto_reconnect": true
  }
}
```

---

## 🚀 Starting Services

### Available Interfaces

The MCP Client provides three professional interfaces:

1. **🤖 Interactive Chatbot** - Rich terminal interface
2. **🌐 REST API** - HTTP endpoints for web integration
3. **⚡ WebSocket Server** - Real-time bidirectional communication

### Command Options

#### Individual Interface Commands
```bash
# Start only the chatbot interface
mcp-client --interfaces chatbot

# Start only the REST API (FastAPI)
mcp-client --interfaces rest_api --api-port 8000

# Start only the WebSocket server
mcp-client --interfaces websocket --ws-port 8765
```

#### Alternative Entry Points
```bash
# Direct interface access
mcp-chatbot                    # Interactive chatbot
mcp-api --port 8000           # REST API server
mcp-websocket --port 8765     # WebSocket server
```

#### Multiple Interfaces
```bash
# Start specific interfaces
mcp-client --interfaces chatbot rest_api --api-port 8000

# Start all interfaces simultaneously
mcp-client --interfaces all --api-port 8000 --ws-port 8765

# With debug mode enabled
mcp-client --interfaces all --debug --api-port 8000 --ws-port 8765
```

### Service Details

#### 🤖 Chatbot Interface
```bash
mcp-client --interfaces chatbot --debug
```
**Features:**
- Rich terminal UI with professional banner
- 4 AI enhancement components active
- Natural language query processing
- Connected to all configured MCP servers
- Real-time response streaming

#### 🌐 REST API Interface
```bash
mcp-client --interfaces rest_api --api-port 8000
```
**Available at:** `http://127.0.0.1:8000`

**Endpoints:**
- `GET /` - API status and information
- `POST /query` - Process natural language queries
- `GET /server/info` - Connected server information
- `GET /server/health` - Server health status
- `GET /tools/list` - Available tools listing

**Example Usage:**
```bash
# Test API status
curl http://127.0.0.1:8000/

# Send a query
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Show database statistics"}'
```

#### ⚡ WebSocket Interface
```bash
mcp-client --interfaces websocket --ws-port 8765
```
**Available at:** `ws://localhost:8765`

**Protocol Example:**
```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8765');

// Send query
ws.send(JSON.stringify({
  "type": "query",
  "id": "unique-id",
  "payload": {"query": "Analyze database performance"}
}));

// Receive response
ws.onmessage = (event) => {
  const response = JSON.parse(event.data);
  console.log(response.payload.result);
};
```

---

## ✅ Verification & Testing

### Quick Health Check
```bash
# Test basic functionality
mcp-client --help

# Check configuration
python -c "from mcp_client.core import Config; print('Config loaded successfully')"

# Test import
python -c "import mcp_client; print('Package imported successfully')"
```

### Test Individual Interfaces

#### Test Chatbot
```bash
# Start chatbot in debug mode
mcp-client --interfaces chatbot --debug

# In the chatbot, try:
> help
> list tools
> show connected servers
```

#### Test REST API
```bash
# Terminal 1: Start API
mcp-client --interfaces rest_api --api-port 8000

# Terminal 2: Test endpoints
curl http://127.0.0.1:8000/
curl http://127.0.0.1:8000/server/health
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello, what can you do?"}'
```

#### Test WebSocket
```bash
# Terminal 1: Start WebSocket server
mcp-client --interfaces websocket --ws-port 8765

# Terminal 2: Test with websocat (install: cargo install websocat)
websocat ws://localhost:8765
# Then send: {"type": "query", "id": "test", "payload": {"query": "Hello"}}
```

### Integration Testing
```bash
# Start all interfaces
mcp-client --interfaces all --api-port 8000 --ws-port 8765

# Verify all are running:
curl http://127.0.0.1:8000/server/health
websocat ws://localhost:8765 <<< '{"type": "ping"}'
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Error: ModuleNotFoundError: No module named 'mcp_client'

# Solution: Install in editable mode
pip install -e .

# Or add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

#### 2. Permission Errors (Windows)
```bash
# Error: PermissionError: [WinError 32] The process cannot access the file...

# This is a known logging issue but doesn't affect functionality
# Look for these SUCCESS messages:
# ✅ "Enhancement components initialized"
# ✅ "Successfully connected to oracle-db"
# ✅ "MCP service initialized successfully"

# Solution: Run as Administrator or disable file logging:
# In .env: ENABLE_FILE_LOGGING=false
```

#### 3. Port Already in Use
```bash
# Error: OSError: [Errno 48] Address already in use

# Solution: Use different ports
mcp-client --interfaces all --api-port 8001 --ws-port 8766

# Or find what's using the port:
# Windows: netstat -ano | findstr :8000
# Linux/macOS: lsof -i :8000
```

#### 4. Oracle Database Connection Issues
```bash
# Error: Could not connect to Oracle database

# Check:
1. Oracle database is running
2. Connection details in .env are correct
3. Oracle client libraries are installed
4. Network connectivity to database

# Test connection:
python -c "import cx_Oracle; print('Oracle client available')"
```

#### 5. API Key Issues
```bash
# Error: Invalid API key

# Solution:
1. Verify API keys in .env file
2. Ensure no extra spaces or quotes
3. Check API key permissions
4. Test with a simple API call
```

### Debug Mode
```bash
# Enable comprehensive debugging
mcp-client --interfaces all --debug --log-level DEBUG

# Check logs
tail -f logs/mcp_client.log

# On Windows:
type logs\mcp_client.log
```

### Getting Help
1. **Check logs:** `logs/mcp_client.log`
2. **Enable debug mode:** Add `--debug` flag
3. **Verify configuration:** Ensure `.env` and `configs/mcp_servers.json` are correct
4. **Test connectivity:** Try individual interfaces first
5. **Check dependencies:** Ensure all required packages are installed

---

## 🎉 Success!

Once setup is complete, you'll have:

✅ **Interactive Chatbot** - `mcp-client --interfaces chatbot`
✅ **REST API Server** - `http://127.0.0.1:8000`
✅ **WebSocket Server** - `ws://localhost:8765`
✅ **AI-Enhanced Processing** - Chain-of-thought, context optimization
✅ **Multi-Server Support** - Oracle DB, filesystem, memory, and more
✅ **Production Ready** - Logging, error handling, health checks

### Next Steps
1. **Explore the interfaces** - Try different queries and commands
2. **Configure additional MCP servers** - Enable filesystem, memory, etc.
3. **Integrate with your applications** - Use REST API or WebSocket interfaces
4. **Customize AI behavior** - Adjust models and enhancement settings
5. **Scale for production** - Configure monitoring and load balancing

---

**🚀 Your MCP Client is ready for professional use with advanced AI capabilities!**