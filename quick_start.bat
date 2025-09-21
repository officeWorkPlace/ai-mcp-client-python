@echo off
echo.
echo ====================================
echo   Global MCP Client - Quick Start
echo ====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo [1/5] Checking Python version...
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python version: %PYTHON_VERSION%

echo.
echo [2/5] Installing dependencies...
echo.

REM Try uv first, fallback to pip
uv --version >nul 2>&1
if not errorlevel 1 (
    echo Using uv package manager...
    uv sync
) else (
    echo Using pip package manager...
    pip install -r requirements.txt
)

if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo [3/5] Setting up environment configuration...

REM Copy .env.example to .env if it doesn't exist
if not exist .env (
    copy .env.example .env
    echo Created .env file from template
    echo.
    echo IMPORTANT: Please edit .env file with your API keys:
    echo - ANTHROPIC_API_KEY or OPENAI_API_KEY
    echo - Oracle database connection details (if using Oracle MCP server)
    echo.
) else (
    echo .env file already exists
)

echo.
echo [4/5] Creating log directory...
if not exist logs mkdir logs
echo Log directory created

echo.
echo [5/5] Validating configuration...
python -m global_mcp_client.cli validate

if errorlevel 1 (
    echo.
    echo WARNING: Configuration validation failed
    echo Please check your .env file and server configurations
    echo.
)

echo.
echo ====================================
echo          Setup Complete!
echo ====================================
echo.
echo Quick commands:
echo   1. Start chatbot:           python -m global_mcp_client.main
echo   2. Test connections:        python -m global_mcp_client.cli test
echo   3. Show information:        python -m global_mcp_client.cli info
echo   4. Show help:              python -m global_mcp_client.cli --help
echo.
echo Before starting, make sure to:
echo   1. Edit .env with your API keys
echo   2. Configure MCP servers in configs/mcp_servers.json
echo   3. Ensure your Oracle MCP server is accessible (if using)
echo.

set /p CHOICE="Would you like to start the chatbot now? (y/n): "
if /i "%CHOICE%"=="y" (
    echo.
    echo Starting Global MCP Client...
    python -m global_mcp_client.main
) else (
    echo.
    echo You can start the chatbot later with: python -m global_mcp_client.main
)

echo.
pause
