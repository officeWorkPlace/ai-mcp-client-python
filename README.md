
# Global MCP Client

A production-ready Python application for connecting to multiple Model Context Protocol (MCP) servers with an intelligent chatbot interface powered by AI.

## 🌟 Features

- **Multi-Server Support**: Connect to multiple MCP servers simultaneously
- **AI-Powered Chat**: Interactive chatbot interface using Anthropic Claude or OpenAI GPT
- **Production Ready**: Comprehensive logging, error handling, rate limiting, and health monitoring
- **Extensible Architecture**: Easy to add new MCP servers and tools
- **Rich CLI Interface**: Beautiful command-line interface with helpful commands
- **Configuration Management**: Flexible JSON-based configuration with environment variable support
- **Oracle Database Integration**: Built-in support for your Oracle DB MCP server
- **Health Monitoring**: Real-time server health checks and monitoring
- **Security**: Input validation, rate limiting, and safe execution

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- uv package manager (recommended) or pip
- Node.js (for some MCP servers)

### Installation

1. **Clone or download the project to your directory:**
   ```bash
   cd "G:\Software G\MCP\python\MCP-CLIENT"
   ```

2. **Install dependencies:**
   ```bash
   # Using uv (recommended)
   uv sync

   # Or using pip
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

4. **Configure your MCP servers:**
   Edit `configs/mcp_servers.json` to add your Oracle DB server and other servers.

### Running the Application

```bash
# Start the interactive chatbot
python -m global_mcp_client.main

# Or use the CLI
python -m global_mcp_client.cli chat

# Run with debug mode
python -m global_mcp_client.cli --debug chat
```

## 📋 Configuration

### Environment Variables

Create a `.env` file with your configuration:

```env
# AI Provider Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Default AI Model Configuration
DEFAULT_MODEL=claude-3-7-sonnet-20250219
MAX_TOKENS=4096
TEMPERATURE=0.1

# Oracle Database Configuration
ORACLE_HOST=localhost
ORACLE_PORT=1521
ORACLE_SERVICE_NAME=XE
ORACLE_USERNAME=your_oracle_username
ORACLE_PASSWORD=your_oracle_password

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/global_mcp_client.log
```

### MCP Server Configuration

Edit `configs/mcp_servers.json` to configure your MCP servers:

```json
{
  "mcpServers": {
    "oracle-db": {
      "command": "uv",
      "args": ["run", "mcp-oracledb-server"],
      "description": "Oracle Database MCP Server",
      "enabled": true,
      "timeout": 45,
      "retry_attempts": 3,
      "cwd": "G:\\Software G\\MCP\\python\\mcp-oracledb-server",
      "env": {
        "ORACLE_HOST": "${ORACLE_HOST}",
        "ORACLE_PORT": "${ORACLE_PORT}",
        "ORACLE_SERVICE_NAME": "${ORACLE_SERVICE_NAME}",
        "ORACLE_USERNAME": "${ORACLE_USERNAME}",
        "ORACLE_PASSWORD": "${ORACLE_PASSWORD}"
      }
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
      "description": "File system operations server",
      "enabled": true
    },
    "fetch": {
      "command": "uvx",
      "args": ["mcp-server-fetch"],
      "description": "Web content fetching server",
      "enabled": true
    }
  }
}
```

## 🎯 Usage

### Interactive Chat

Start the chatbot and use natural language commands:

```bash
python -m global_mcp_client.main
```

Example queries:
- "List files in the current directory"
- "Execute a SQL query to show all users from the database"
- "Fetch the content of https://example.com"
- "Search for Python tutorials"

### CLI Commands

```bash
# Show help
python -m global_mcp_client.cli --help

# Validate configuration
python -m global_mcp_client.cli validate

# Test server connections
python -m global_mcp_client.cli test

# Test specific server
python -m global_mcp_client.cli test --server oracle-db

# Show system information
python -m global_mcp_client.cli info

# Execute single query
python -m global_mcp_client.cli query "What files are in the current directory?"
```

### Chat Commands

While in the interactive chat, you can use these commands:

- `/help` or `/h` - Show help message
- `/info` or `/i` - Show connected servers and available tools
- `/health` - Check server health status
- `/stats` - Show session statistics
- `/config` - Show current configuration
- `/clear` - Clear the screen
- `/quit` or `/q` - Exit the chatbot

## 🔧 Advanced Configuration

### Adding New MCP Servers

1. Add server configuration to `configs/mcp_servers.json`:

```json
{
  "new-server": {
    "command": "your-command",
    "args": ["arg1", "arg2"],
    "description": "Your server description",
    "enabled": true,
    "timeout": 30,
    "retry_attempts": 3,
    "env": {
      "ENV_VAR": "${ENV_VAR_VALUE}"
    }
  }
}
```

2. Restart the application to load the new server.

### Custom Logging

Configure logging in your `.env` file:

```env
LOG_LEVEL=DEBUG
LOG_FILE=logs/custom.log
ENABLE_FILE_LOGGING=true
ENABLE_CONSOLE_LOGGING=true
```

### Health Monitoring

The application includes built-in health monitoring:

- Automatic health checks every 5 minutes
- Real-time server status monitoring
- Error tracking and reporting
- Connection retry with exponential backoff

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=global_mcp_client

# Run specific test file
pytest tests/test_config.py

# Run with verbose output
pytest -v
```

## 🔒 Security Features

- **Input Validation**: All user inputs are validated for safety
- **Rate Limiting**: Configurable rate limiting to prevent abuse
- **Safe Execution**: Sandboxed tool execution environment
- **Error Handling**: Comprehensive error handling and logging
- **Environment Isolation**: Secure environment variable handling

## 📊 Monitoring and Logging

### Log Files

Logs are stored in the `logs/` directory:
- `global_mcp_client.log` - Main application log
- `tests/` - Test logs

### Health Checks

Monitor server health:
```bash
python -m global_mcp_client.cli test
```

View real-time health in chat:
```
/health
```

## 🛠️ Development

### Project Structure

```
global_mcp_client/
├── core/                 # Core application logic
│   ├── client.py        # Main MCP client
│   ├── config.py        # Configuration management
│   ├── logger.py        # Logging setup
│   └── exceptions.py    # Custom exceptions
├── utils/               # Utility functions
│   ├── validators.py    # Input validation
│   ├── rate_limiter.py  # Rate limiting
│   └── helpers.py       # Helper functions
├── servers/             # Custom MCP servers
├── chatbot.py          # Chatbot implementation
├── main.py             # Main entry point
└── cli.py              # CLI interface
```

### Adding Features

1. **New Tools**: Add tools to existing servers or create new server implementations
2. **AI Providers**: Extend the client to support additional AI providers
3. **Integrations**: Add new MCP server integrations
4. **UI Enhancements**: Improve the chat interface and CLI

### Code Quality

The project includes:
- Type hints throughout
- Comprehensive error handling
- Unit tests with pytest
- Code formatting with black
- Import sorting with isort
- Linting with flake8

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **Connection Failures**
   ```bash
   # Test individual servers
   python -m global_mcp_client.cli test --server oracle-db
   
   # Check configuration
   python -m global_mcp_client.cli validate
   ```

2. **Missing Dependencies**
   ```bash
   # Reinstall dependencies
   uv sync --force
   
   # Or with pip
   pip install -r requirements.txt --force-reinstall
   ```

3. **Environment Variables**
   ```bash
   # Verify .env file exists and has correct values
   cat .env
   
   # Check environment validation
   python -m global_mcp_client.cli validate
   ```

4. **Oracle Database Connection**
   - Ensure Oracle database is running
   - Verify connection parameters in `.env`
   - Check that Oracle MCP server is properly installed
   - Test connection manually

### Debug Mode

Run with debug mode for detailed logging:
```bash
python -m global_mcp_client.cli --debug --log-level DEBUG chat
```

### Log Analysis

Check logs for detailed error information:
```bash
tail -f logs/global_mcp_client.log
```

## 📞 Support

For support and questions:

1. Check the troubleshooting section above
2. Review the logs for error details
3. Ensure all prerequisites are installed
4. Verify configuration files are correct

## 🎉 Success! 

You now have a fully functional Global MCP Client that can:

✅ Connect to multiple MCP servers simultaneously  
✅ Integrate with your Oracle Database MCP server  
✅ Provide an intelligent AI-powered chat interface  
✅ Handle production workloads with proper monitoring  
✅ Scale to support additional servers and tools  

Start chatting with your MCP servers and enjoy the power of multi-server AI interactions!

