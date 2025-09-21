@echo off
REM ============================================================================
REM Global MCP Client - Complete Installation and Validation Script (Windows)
REM ============================================================================

echo.
echo ============================================================================
echo                    GLOBAL MCP CLIENT INSTALLER
echo                        Complete Setup Script
echo ============================================================================
echo.

REM Set error handling
setlocal enabledelayedexpansion

REM Check if running as administrator (optional but recommended)
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Not running as administrator - some features may be limited
) else (
    echo [INFO] Running with administrator privileges
)

echo.
echo [STEP 1/10] Checking system requirements...
echo ============================================

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo.
    echo Please install Python 3.8 or higher from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to:
    echo - Check "Add Python to PATH" during installation
    echo - Install pip (included by default)
    echo.
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [SUCCESS] Python version: %PYTHON_VERSION%

REM Check pip
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip is not available
    echo Please ensure pip is installed with Python
    pause
    exit /b 1
)
echo [SUCCESS] pip is available

echo.
echo [STEP 2/10] Checking package managers...
echo ==========================================

REM Check for uv (recommended)
uv --version >nul 2>&1
if not errorlevel 1 (
    echo [SUCCESS] uv package manager detected (recommended)
    set PACKAGE_MANAGER=uv
) else (
    echo [INFO] uv not found - using pip as package manager
    echo [INFO] For faster dependency management, consider installing uv:
    echo        pip install uv
    set PACKAGE_MANAGER=pip
)

echo.
echo [STEP 3/10] Checking Node.js (for MCP servers)...
echo ===============================================

node --version >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=1" %%i in ('node --version 2^>^&1') do set NODE_VERSION=%%i
    echo [SUCCESS] Node.js version: !NODE_VERSION!
) else (
    echo [WARNING] Node.js not found
    echo Some MCP servers require Node.js. Install from:
    echo https://nodejs.org/
)

npx --version >nul 2>&1
if not errorlevel 1 (
    echo [SUCCESS] npx is available
) else (
    echo [WARNING] npx not available - some MCP servers may not work
)

echo.
echo [STEP 4/10] Installing Python dependencies...
echo ============================================

if "%PACKAGE_MANAGER%"=="uv" (
    echo [INFO] Installing with uv...
    uv sync
    if errorlevel 1 (
        echo [WARNING] uv sync failed, trying pip install...
        uv pip install -r requirements.txt
        if errorlevel 1 (
            echo [ERROR] Failed to install dependencies with uv
            set PACKAGE_MANAGER=pip
        ) else (
            echo [SUCCESS] Dependencies installed with uv pip
        )
    ) else (
        echo [SUCCESS] Dependencies installed with uv sync
    )
)

if "%PACKAGE_MANAGER%"=="pip" (
    echo [INFO] Installing with pip...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies with pip
        echo.
        echo Try installing dependencies individually:
        echo python -m pip install mcp anthropic rich click pydantic
        pause
        exit /b 1
    ) else (
        echo [SUCCESS] Dependencies installed with pip
    )
)

echo.
echo [STEP 5/10] Installing development dependencies...
echo ================================================

if exist requirements-dev.txt (
    echo [INFO] Installing development dependencies...
    if "%PACKAGE_MANAGER%"=="uv" (
        uv pip install -r requirements-dev.txt
    ) else (
        python -m pip install -r requirements-dev.txt
    )
    
    if not errorlevel 1 (
        echo [SUCCESS] Development dependencies installed
    ) else (
        echo [WARNING] Failed to install development dependencies
        echo This won't affect basic functionality
    )
) else (
    echo [INFO] No development requirements file found
)

echo.
echo [STEP 6/10] Setting up environment configuration...
echo =================================================

REM Create .env file if it doesn't exist
if not exist .env (
    if exist .env.example (
        copy .env.example .env >nul
        echo [SUCCESS] Created .env file from template
        echo [IMPORTANT] Please edit .env file with your API keys:
        echo            - ANTHROPIC_API_KEY or OPENAI_API_KEY
        echo            - Oracle database connection details
    ) else (
        echo [ERROR] .env.example file not found
        goto :create_basic_env
    )
) else (
    echo [INFO] .env file already exists
)
goto :continue_setup

:create_basic_env
echo [INFO] Creating basic .env file...
(
echo # Global MCP Client Environment Configuration
echo # Please fill in your values
echo.
echo # AI Provider Configuration
echo ANTHROPIC_API_KEY=your_anthropic_api_key_here
echo OPENAI_API_KEY=your_openai_api_key_here
echo.
echo # Logging Configuration
echo LOG_LEVEL=INFO
echo LOG_FILE=logs/global_mcp_client.log
echo.
echo # Oracle Database Configuration
echo ORACLE_HOST=localhost
echo ORACLE_PORT=1521
echo ORACLE_SERVICE_NAME=XE
echo ORACLE_USERNAME=your_oracle_username
echo ORACLE_PASSWORD=your_oracle_password
) > .env
echo [SUCCESS] Created basic .env file

:continue_setup

echo.
echo [STEP 7/10] Creating required directories...
echo ==========================================

if not exist logs mkdir logs
echo [SUCCESS] Created logs directory

if not exist configs mkdir configs
echo [SUCCESS] Ensured configs directory exists

echo.
echo [STEP 8/10] Running comprehensive validation...
echo ============================================

python install_and_validate.py
set VALIDATION_RESULT=%errorlevel%

echo.
echo [STEP 9/10] Testing basic functionality...
echo ========================================

echo [INFO] Testing CLI help command...
python -m global_mcp_client.cli --help >nul 2>&1
if not errorlevel 1 (
    echo [SUCCESS] CLI help command works
) else (
    echo [ERROR] CLI help command failed
)

echo [INFO] Testing configuration validation...
python -m global_mcp_client.cli validate
if not errorlevel 1 (
    echo [SUCCESS] Configuration validation passed
) else (
    echo [WARNING] Configuration validation had issues (likely missing API keys)
)

echo.
echo [STEP 10/10] Final setup recommendations...
echo =========================================

echo [INFO] Installation complete! Here's what to do next:
echo.
echo 1. CONFIGURE API KEYS:
echo    Edit .env file and add your API keys:
echo    - ANTHROPIC_API_KEY (for Claude AI)
echo    - OPENAI_API_KEY (for GPT AI)
echo.
echo 2. CONFIGURE MCP SERVERS:
echo    Edit configs/mcp_servers.json to:
echo    - Update Oracle MCP server path
echo    - Enable/disable servers as needed
echo.
echo 3. TEST CONNECTIONS:
echo    python -m global_mcp_client.cli test
echo.
echo 4. START THE APPLICATION:
echo    python -m global_mcp_client.main
echo.

REM Check validation results
if %VALIDATION_RESULT% equ 0 (
    echo [SUCCESS] ✓ All validations passed!
    echo [SUCCESS] ✓ Your Global MCP Client is ready to use!
) else (
    echo [WARNING] ⚠ Some validations failed
    echo [WARNING] ⚠ Please review the validation output above
)

echo.
echo ============================================================================
echo                              QUICK COMMANDS
echo ============================================================================
echo.
echo Start chatbot:           python -m global_mcp_client.main
echo Test connections:        python -m global_mcp_client.cli test
echo Show information:        python -m global_mcp_client.cli info
echo Validate setup:          python -m global_mcp_client.cli validate
echo Show help:              python -m global_mcp_client.cli --help
echo.
echo Configuration files:
echo - Environment:           .env
echo - MCP Servers:          configs/mcp_servers.json
echo - Logs:                 logs/global_mcp_client.log
echo.

set /p START_NOW="Would you like to start the chatbot now? (y/n): "
if /i "%START_NOW%"=="y" (
    echo.
    echo Starting Global MCP Client...
    echo ============================================================================
    python -m global_mcp_client.main
) else (
    echo.
    echo You can start the chatbot later with: python -m global_mcp_client.main
    echo.
    echo Thank you for installing Global MCP Client!
)

pause
