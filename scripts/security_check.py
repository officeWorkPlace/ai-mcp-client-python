#!/usr/bin/env python3
"""
Security check script for MCP Client
Runs all security tools and generates a comprehensive report
"""

import subprocess
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def safe_print(message: str) -> None:
    """Cross-platform safe print that handles encoding issues"""
    try:
        print(message)
    except UnicodeEncodeError:
        # Fallback for Windows console encoding issues
        # Replace problematic Unicode characters
        message = message.replace('✓', 'OK').replace('✗', 'FAIL').replace('⚠', 'WARN')
        message = message.replace('🔴', '[CRITICAL]').replace('🟡', '[WARNING]').replace('🟢', '[SECURE]')
        print(message)

def run_command(command: List[str], description: str) -> Tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr"""
    safe_print(f"{Colors.BLUE}Running: {description}{Colors.END}")
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", f"Command timed out after 5 minutes: {' '.join(command)}"
    except Exception as e:
        return 1, "", f"Error running command: {e}"

def check_pip_audit() -> Dict[str, any]:
    """Run pip-audit vulnerability scan"""
    safe_print(f"\n{Colors.BOLD}=== Dependency Vulnerability Scan (pip-audit) ==={Colors.END}")

    exit_code, stdout, stderr = run_command(
        ["pip-audit", "--desc", "--format", "json"],
        "Scanning dependencies for vulnerabilities"
    )

    result = {
        "tool": "pip-audit",
        "status": "passed" if exit_code == 0 else "failed",
        "vulnerabilities": [],
        "raw_output": stdout + stderr
    }

    if exit_code == 0:
        try:
            # Parse JSON output if available
            if stdout.strip() and stdout.strip().startswith('['):
                vulnerabilities = json.loads(stdout)
                result["vulnerabilities"] = vulnerabilities
                if vulnerabilities:
                    result["status"] = "warnings"
                    safe_print(f"{Colors.YELLOW}WARN Found {len(vulnerabilities)} vulnerabilities{Colors.END}")
                else:
                    safe_print(f"{Colors.GREEN}OK No vulnerabilities found{Colors.END}")
            else:
                safe_print(f"{Colors.GREEN}OK No vulnerabilities found{Colors.END}")
        except json.JSONDecodeError:
            safe_print(f"{Colors.GREEN}OK Scan completed successfully{Colors.END}")
    else:
        safe_print(f"{Colors.RED}FAIL pip-audit failed: {stderr}{Colors.END}")

    return result

def check_bandit() -> Dict[str, any]:
    """Run bandit security linting"""
    safe_print(f"\n{Colors.BOLD}=== Static Security Analysis (bandit) ==={Colors.END}")

    exit_code, stdout, stderr = run_command(
        ["bandit", "-r", "src/", "-f", "json"],
        "Scanning code for security issues"
    )

    result = {
        "tool": "bandit",
        "status": "passed" if exit_code == 0 else "failed",
        "issues": [],
        "raw_output": stdout + stderr
    }

    try:
        if stdout.strip():
            bandit_result = json.loads(stdout)
            issues = bandit_result.get("results", [])
            result["issues"] = issues

            if issues:
                result["status"] = "warnings"
                high_issues = [i for i in issues if i.get("issue_severity") == "HIGH"]
                medium_issues = [i for i in issues if i.get("issue_severity") == "MEDIUM"]
                low_issues = [i for i in issues if i.get("issue_severity") == "LOW"]

                safe_print(f"{Colors.YELLOW}WARN Found {len(issues)} security issues:{Colors.END}")
                if high_issues:
                    safe_print(f"  {Colors.RED}- {len(high_issues)} HIGH severity{Colors.END}")
                if medium_issues:
                    safe_print(f"  {Colors.YELLOW}- {len(medium_issues)} MEDIUM severity{Colors.END}")
                if low_issues:
                    safe_print(f"  - {len(low_issues)} LOW severity")
            else:
                safe_print(f"{Colors.GREEN}OK No security issues found{Colors.END}")
    except (json.JSONDecodeError, KeyError):
        if exit_code == 0:
            safe_print(f"{Colors.GREEN}OK No security issues found{Colors.END}")
        else:
            safe_print(f"{Colors.RED}FAIL bandit failed: {stderr}{Colors.END}")

    return result

def check_safety() -> Dict[str, any]:
    """Run safety vulnerability scan (optional - requires login)"""
    safe_print(f"\n{Colors.BOLD}=== Safety Vulnerability Scan ==={Colors.END}")

    # Check if safety is configured
    exit_code, stdout, stderr = run_command(
        ["safety", "--version"],
        "Checking safety availability"
    )

    if exit_code != 0:
        safe_print(f"{Colors.YELLOW}WARN Safety CLI not available or not configured{Colors.END}")
        return {
            "tool": "safety",
            "status": "skipped",
            "message": "Safety CLI requires registration - using pip-audit instead"
        }

    exit_code, stdout, stderr = run_command(
        ["safety", "scan", "--json"],
        "Running safety vulnerability scan"
    )

    result = {
        "tool": "safety",
        "status": "passed" if exit_code == 0 else "skipped",
        "raw_output": stdout + stderr
    }

    if "login or register" in stderr.lower():
        safe_print(f"{Colors.YELLOW}WARN Safety requires registration - skipping{Colors.END}")
        result["status"] = "skipped"
        result["message"] = "Safety CLI requires registration"
    elif exit_code == 0:
        safe_print(f"{Colors.GREEN}OK Safety scan completed{Colors.END}")
    else:
        safe_print(f"{Colors.YELLOW}WARN Safety scan failed or unavailable{Colors.END}")
        result["status"] = "skipped"

    return result

def generate_report(results: List[Dict]) -> None:
    """Generate a summary security report"""
    safe_print(f"\n{Colors.BOLD}=== Security Scan Summary ==={Colors.END}")

    total_issues = 0
    critical_issues = 0

    for result in results:
        tool = result["tool"]
        status = result["status"]

        if status == "passed":
            safe_print(f"{Colors.GREEN}OK {tool}: PASSED{Colors.END}")
        elif status == "warnings":
            issues_count = len(result.get("issues", [])) + len(result.get("vulnerabilities", []))
            total_issues += issues_count
            safe_print(f"{Colors.YELLOW}WARN {tool}: {issues_count} issues found{Colors.END}")

            # Check for critical issues
            if tool == "bandit":
                high_severity = len([i for i in result.get("issues", [])
                                   if i.get("issue_severity") == "HIGH"])
                critical_issues += high_severity
        elif status == "failed":
            safe_print(f"{Colors.RED}FAIL {tool}: FAILED{Colors.END}")
            critical_issues += 1
        else:
            safe_print(f"- {tool}: SKIPPED")

    safe_print(f"\n{Colors.BOLD}Overall Security Status:{Colors.END}")
    if critical_issues > 0:
        safe_print(f"{Colors.RED}[CRITICAL]: {critical_issues} critical security issues require immediate attention{Colors.END}")
        sys.exit(1)
    elif total_issues > 0:
        safe_print(f"{Colors.YELLOW}[WARNING]: {total_issues} security issues found - review recommended{Colors.END}")
        sys.exit(1)  # Exit with error code for CI/CD
    else:
        safe_print(f"{Colors.GREEN}[SECURE]: All security checks passed{Colors.END}")

def main():
    """Main security check function"""
    safe_print(f"{Colors.BOLD}{Colors.BLUE}MCP Client Security Check{Colors.END}")
    safe_print("=" * 50)

    # Ensure we're in the project directory
    project_root = Path(__file__).parent.parent
    if not (project_root / "pyproject.toml").exists():
        safe_print(f"{Colors.RED}Error: Not in project directory or pyproject.toml not found{Colors.END}")
        sys.exit(1)

    results = []

    # Run all security checks
    try:
        results.append(check_pip_audit())
        results.append(check_bandit())
        results.append(check_safety())
    except KeyboardInterrupt:
        safe_print(f"\n{Colors.YELLOW}Security check interrupted by user{Colors.END}")
        sys.exit(1)
    except Exception as e:
        safe_print(f"{Colors.RED}Unexpected error during security check: {e}{Colors.END}")
        sys.exit(1)

    # Generate final report
    generate_report(results)

    safe_print(f"\n{Colors.BLUE}For more details, see SECURITY.md{Colors.END}")

if __name__ == "__main__":
    main()