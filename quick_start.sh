#!/bin/bash

echo ""
echo "===================================="
echo "   Global MCP Client - Quick Start"
echo "===================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "[1/5] Checking Python version..."
PYTHON_VERSION=$(python3 --version)
echo "Python version: $PYTHON_VERSION"

echo ""
echo "[2/5] Installing dependencies..."
echo ""

# Try uv first, fallback to pip
if command -v uv &> /dev/null; then
    echo "Using uv package manager..."
    uv sync
else
    echo "Using pip package manager..."
    python3 -m pip install -r requirements.txt
fi

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

echo ""
echo "[3/5] Setting up environment configuration..."

# Copy .env.example to .env if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file from template"
    echo ""
    echo "IMPORTANT: Please edit .env file with your API keys:"
    echo "- ANTHROPIC_API_KEY or OPENAI_API_KEY"
    echo "- Oracle database connection details (if using Oracle MCP server)"
    echo ""
else
    echo ".env file already exists"
fi

echo ""
echo "[4/5] Creating log directory..."
mkdir -p logs
echo "Log directory created"

echo ""
echo "[5/5] Validating configuration..."
python3 -m global_mcp_client.cli validate

if [ $? -ne 0 ]; then
    echo ""
    echo "WARNING: Configuration validation failed"
    echo "Please check your .env file and server configurations"
    echo ""
fi

echo ""
echo "===================================="
echo "          Setup Complete!"
echo "===================================="
echo ""
echo "Quick commands:"
echo "  1. Start chatbot:           python3 -m global_mcp_client.main"
echo "  2. Test connections:        python3 -m global_mcp_client.cli test"
echo "  3. Show information:        python3 -m global_mcp_client.cli info"
echo "  4. Show help:              python3 -m global_mcp_client.cli --help"
echo ""
echo "Before starting, make sure to:"
echo "  1. Edit .env with your API keys"
echo "  2. Configure MCP servers in configs/mcp_servers.json"
echo "  3. Ensure your Oracle MCP server is accessible (if using)"
echo ""

read -p "Would you like to start the chatbot now? (y/n): " -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Starting Global MCP Client..."
    python3 -m global_mcp_client.main
else
    echo ""
    echo "You can start the chatbot later with: python3 -m global_mcp_client.main"
fi

echo ""
