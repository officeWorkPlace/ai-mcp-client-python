"""
Complete Installation and Validation Script for Global MCP Client
================================================================

This script will:
1. Check system requirements
2. Install all dependencies
3. Validate the project structure
4. Test imports and functionality
5. Run comprehensive tests
6. Provide setup recommendations
"""

import sys
import os
import subprocess
import importlib
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

# ANSI color codes for output formatting
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE} {title.center(58)} {Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

def print_success(message: str):
    """Print a success message"""
    print(f"{Colors.GREEN}[+] {message}{Colors.END}")

def print_error(message: str):
    """Print an error message"""
    print(f"{Colors.RED}[!] {message}{Colors.END}")

def print_warning(message: str):
    """Print a warning message"""
    print(f"{Colors.YELLOW}[*] {message}{Colors.END}")

def print_info(message: str):
    """Print an info message"""
    print(f"{Colors.CYAN}[i] {message}{Colors.END}")

def run_command(command: List[str], capture_output: bool = True) -> Tuple[bool, str]:
    """Run a command and return success status and output"""
    try:
        result = subprocess.run(
            command,
            capture_output=capture_output,
            text=True,
            check=False
        )
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)

def check_python_version() -> bool:
    """Check if Python version is compatible"""
    print_header("1. CHECKING PYTHON VERSION")
    
    version_info = sys.version_info
    version_str = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    
    print_info(f"Python version: {version_str}")
    print_info(f"Python executable: {sys.executable}")
    
    if version_info >= (3, 8):
        print_success("Python version is compatible (>=3.8)")
        return True
    else:
        print_error(f"Python version {version_str} is not supported. Please install Python 3.8 or higher.")
        return False

def check_package_managers() -> str:
    """Check available package managers"""
    print_header("2. CHECKING PACKAGE MANAGERS")
    
    # Check for uv
    success, _ = run_command(["uv", "--version"])
    if success:
        print_success("uv package manager is available (recommended)")
        return "uv"
    else:
        print_warning("uv package manager not found")
    
    # Check for pip
    success, _ = run_command([sys.executable, "-m", "pip", "--version"])
    if success:
        print_success("pip package manager is available")
        return "pip"
    else:
        print_error("No package manager found. Please install pip.")
        return None

def install_dependencies(package_manager: str) -> bool:
    """Install project dependencies"""
    print_header("3. INSTALLING DEPENDENCIES")
    
    if package_manager == "uv":
        print_info("Installing dependencies with uv...")
        success, output = run_command(["uv", "sync"], capture_output=False)
        if not success:
            print_warning("uv sync failed, trying uv pip install...")
            success, output = run_command(["uv", "pip", "install", "-r", "requirements.txt"], capture_output=False)
    else:
        print_info("Installing dependencies with pip...")
        success, output = run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], capture_output=False)
    
    if success:
        print_success("Dependencies installed successfully")
        return True
    else:
        print_error("Failed to install dependencies")
        print(output)
        return False

def validate_project_structure() -> bool:
    """Validate the project directory structure"""
    print_header("4. VALIDATING PROJECT STRUCTURE")
    
    required_files = [
        "global_mcp_client/__init__.py",
        "global_mcp_client/main.py",
        "global_mcp_client/cli.py",
        "global_mcp_client/chatbot.py",
        "global_mcp_client/core/__init__.py",
        "global_mcp_client/core/client.py",
        "global_mcp_client/core/config.py",
        "global_mcp_client/core/logger.py",
        "global_mcp_client/core/exceptions.py",
        "global_mcp_client/utils/__init__.py",
        "global_mcp_client/utils/validators.py",
        "global_mcp_client/utils/rate_limiter.py",
        "global_mcp_client/utils/helpers.py",
        "global_mcp_client/servers/__init__.py",
        "configs/mcp_servers.json",
        "requirements.txt",
        "pyproject.toml",
        ".env.example",
        "README.md"
    ]
    
    required_dirs = [
        "global_mcp_client",
        "global_mcp_client/core",
        "global_mcp_client/utils", 
        "global_mcp_client/servers",
        "configs",
        "logs",
        "tests"
    ]
    
    all_valid = True
    
    # Check directories
    print_info("Checking required directories...")
    for dir_path in required_dirs:
        if Path(dir_path).is_dir():
            print_success(f"Directory exists: {dir_path}")
        else:
            print_error(f"Missing directory: {dir_path}")
            all_valid = False
    
    # Check files
    print_info("Checking required files...")
    for file_path in required_files:
        if Path(file_path).is_file():
            print_success(f"File exists: {file_path}")
        else:
            print_error(f"Missing file: {file_path}")
            all_valid = False
    
    return all_valid

def test_imports() -> bool:
    """Test if all modules can be imported"""
    print_header("5. TESTING MODULE IMPORTS")
    
    # Add current directory to Python path
    current_dir = Path.cwd()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    modules_to_test = [
        "global_mcp_client",
        "global_mcp_client.core",
        "global_mcp_client.core.config",
        "global_mcp_client.core.client",
        "global_mcp_client.core.logger",
        "global_mcp_client.core.exceptions",
        "global_mcp_client.utils",
        "global_mcp_client.utils.validators",
        "global_mcp_client.utils.rate_limiter",
        "global_mcp_client.utils.helpers",
        "global_mcp_client.chatbot",
        "global_mcp_client.cli",
        "global_mcp_client.main"
    ]
    
    all_imports_successful = True
    
    for module_name in modules_to_test:
        try:
            importlib.import_module(module_name)
            print_success(f"Successfully imported: {module_name}")
        except ImportError as e:
            print_error(f"Failed to import {module_name}: {e}")
            all_imports_successful = False
        except Exception as e:
            print_error(f"Error importing {module_name}: {e}")
            all_imports_successful = False
    
    return all_imports_successful

def test_dependencies() -> bool:
    """Test if all required dependencies are available"""
    print_header("6. TESTING DEPENDENCIES")
    
    required_packages = [
        "mcp",
        "anthropic", 
        "openai",
        "fastapi",
        "uvicorn",
        "websockets",
        "pydantic",
        "aiofiles",
        "dotenv",
        "rich",
        "click",
        "tenacity",
        "structlog",
        "nest_asyncio",
        "httpx",
        "jinja2",
        "yaml",
        "psutil"
    ]
    
    all_deps_available = True
    
    for package in required_packages:
        try:
            # Handle special import names
            import_name = package
            if package == "dotenv":
                import_name = "python_dotenv"
            elif package == "yaml":
                import_name = "yaml"
            
            importlib.import_module(import_name)
            print_success(f"Dependency available: {package}")
        except ImportError:
            print_error(f"Missing dependency: {package}")
            all_deps_available = False
        except Exception as e:
            print_warning(f"Issue with dependency {package}: {e}")
    
    return all_deps_available

def validate_configuration() -> bool:
    """Validate configuration files"""
    print_header("7. VALIDATING CONFIGURATION")
    
    # Check .env.example
    env_example_path = Path(".env.example")
    if env_example_path.exists():
        print_success("Found .env.example file")
        with open(env_example_path) as f:
            env_content = f.read()
            required_vars = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "LOG_LEVEL", "ORACLE_HOST"]
            for var in required_vars:
                if var in env_content:
                    print_success(f"Environment variable template found: {var}")
                else:
                    print_warning(f"Missing environment variable template: {var}")
    else:
        print_error("Missing .env.example file")
        return False
    
    # Check mcp_servers.json
    config_path = Path("configs/mcp_servers.json")
    if config_path.exists():
        print_success("Found MCP servers configuration")
        try:
            with open(config_path) as f:
                config_data = json.load(f)
                if "mcpServers" in config_data:
                    servers = config_data["mcpServers"]
                    print_info(f"Found {len(servers)} MCP server configurations:")
                    for server_name in servers.keys():
                        print_success(f"  - {server_name}")
                else:
                    print_warning("No mcpServers section in configuration")
        except json.JSONDecodeError as e:
            print_error(f"Invalid JSON in mcp_servers.json: {e}")
            return False
    else:
        print_error("Missing configs/mcp_servers.json file")
        return False
    
    return True

def run_basic_functionality_test() -> bool:
    """Run basic functionality tests"""
    print_header("8. TESTING BASIC FUNCTIONALITY")
    
    try:
        # Test configuration loading
        from global_mcp_client.core.config import Config
        config = Config()
        print_success("Configuration class instantiated successfully")
        
        # Test validators
        from global_mcp_client.utils.validators import InputValidator
        validator = InputValidator()
        if validator.validate_query("test query"):
            print_success("Input validator working correctly")
        
        # Test rate limiter
        from global_mcp_client.utils.rate_limiter import RateLimiter
        limiter = RateLimiter()
        if limiter.allow_request():
            print_success("Rate limiter working correctly")
        
        # Test logger setup
        from global_mcp_client.core.logger import setup_logging
        logger = setup_logging(log_level="INFO", enable_console_logging=False, enable_file_logging=False)
        print_success("Logger setup working correctly")
        
        return True
        
    except Exception as e:
        print_error(f"Basic functionality test failed: {e}")
        return False

def run_pytest_tests() -> bool:
    """Run the pytest test suite if available"""
    print_header("9. RUNNING TEST SUITE")
    
    # Check if pytest is available
    try:
        import pytest
        print_success("pytest is available")
    except ImportError:
        print_warning("pytest not available, installing...")
        success, _ = run_command([sys.executable, "-m", "pip", "install", "pytest", "pytest-asyncio"])
        if not success:
            print_error("Failed to install pytest")
            return False
    
    # Run tests
    if Path("tests").exists():
        print_info("Running test suite...")
        success, output = run_command([sys.executable, "-m", "pytest", "tests/", "-v"])
        if success:
            print_success("All tests passed")
            return True
        else:
            print_warning("Some tests failed or had issues")
            print(output)
            return False
    else:
        print_warning("No tests directory found")
        return True

def test_cli_functionality() -> bool:
    """Test CLI functionality"""
    print_header("10. TESTING CLI FUNCTIONALITY")
    
    try:
        # Test CLI help
        success, output = run_command([sys.executable, "-m", "global_mcp_client.cli", "--help"])
        if success and "Global MCP Client" in output:
            print_success("CLI help command working")
        else:
            print_error("CLI help command failed")
            return False
        
        # Test validation command
        success, output = run_command([sys.executable, "-m", "global_mcp_client.cli", "validate"])
        if success:
            print_success("CLI validate command working")
        else:
            print_warning("CLI validate command had issues (may be due to missing API keys)")
        
        # Test info command
        success, output = run_command([sys.executable, "-m", "global_mcp_client.cli", "info"])
        if success:
            print_success("CLI info command working")
        else:
            print_warning("CLI info command had issues")
        
        return True
        
    except Exception as e:
        print_error(f"CLI testing failed: {e}")
        return False

def provide_setup_recommendations():
    """Provide setup recommendations based on validation results"""
    print_header("SETUP RECOMMENDATIONS")
    
    print_info("To complete the setup:")
    print("1. Copy .env.example to .env and configure your API keys:")
    print("   cp .env.example .env")
    print("   # Edit .env with your ANTHROPIC_API_KEY or OPENAI_API_KEY")
    
    print("\n2. Configure your Oracle MCP server path in configs/mcp_servers.json")
    print("   # Update the 'cwd' path for oracle-db server")
    
    print("\n3. Test the complete application:")
    print("   python -m global_mcp_client.cli validate")
    print("   python -m global_mcp_client.cli test")
    
    print("\n4. Start the chatbot:")
    print("   python -m global_mcp_client.main")
    
    print(f"\n{Colors.BOLD}5. Quick start scripts are available:{Colors.END}")
    print("   Windows: quick_start.bat")
    print("   Linux/Mac: ./quick_start.sh")

def main():
    """Main validation function"""
    print_header("GLOBAL MCP CLIENT - INSTALLATION & VALIDATION")
    print_info("This script will validate your Global MCP Client installation")
    
    results = {}
    
    # Run all validation steps
    results['python_version'] = check_python_version()
    
    package_manager = check_package_managers()
    if not package_manager:
        print_error("Cannot proceed without a package manager")
        return False
    
    results['dependencies'] = install_dependencies(package_manager)
    results['project_structure'] = validate_project_structure()
    results['imports'] = test_imports()
    results['dependency_check'] = test_dependencies()
    results['configuration'] = validate_configuration()
    results['basic_functionality'] = run_basic_functionality_test()
    results['tests'] = run_pytest_tests()
    results['cli'] = test_cli_functionality()
    
    # Summary
    print_header("VALIDATION SUMMARY")
    
    total_checks = len(results)
    passed_checks = sum(results.values())
    
    for check_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        color = Colors.GREEN if passed else Colors.RED
        print(f"{color}{status}{Colors.END} - {check_name.replace('_', ' ').title()}")
    
    print(f"\n{Colors.BOLD}Overall Result: {passed_checks}/{total_checks} checks passed{Colors.END}")
    
    if passed_checks == total_checks:
        print_success("[+] All validations passed! Your Global MCP Client is ready to use.")
    elif passed_checks >= total_checks * 0.8:
        print_warning("[*] Most validations passed. Review failed checks and setup recommendations.")
    else:
        print_error("[!] Multiple validations failed. Please review errors and fix issues before proceeding.")
    
    provide_setup_recommendations()
    
    return passed_checks >= total_checks * 0.8

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Validation interrupted by user{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error during validation: {e}{Colors.END}")
        sys.exit(1)
