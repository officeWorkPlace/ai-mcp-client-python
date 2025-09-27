# Security Policy

## Overview

The MCP Client project prioritizes security in all aspects of development and deployment. This document outlines our security practices, tooling, and procedures for reporting vulnerabilities.

## Security Tools & Scanning

### Automated Security Scanning

We use multiple layers of security scanning:

1. **Dependency Vulnerability Scanning**
   - `pip-audit`: Scans for known vulnerabilities in Python packages
   - `safety`: Additional vulnerability database scanning (optional)
   - Run manually: `pip-audit --desc`

2. **Static Code Security Analysis**
   - `bandit`: Security linter for Python code
   - Configured in `pyproject.toml` with comprehensive rule set
   - Run manually: `bandit -r src/`

3. **Pre-commit Security Hooks**
   - Automated security checks on every commit
   - Prevents insecure code from entering the repository
   - Install: `pre-commit install`

### Security Configuration

Security tools are configured in `pyproject.toml`:

```toml
[tool.bandit]
exclude_dirs = ["tests", "htmlcov", ".venv", "venv"]
# Comprehensive security rule set enabled

[project.optional-dependencies]
security = [
    "safety>=3.0.0",
    "bandit>=1.7.0",
    "pip-audit>=2.6.0",
    "semgrep>=1.45.0",
    "cryptography>=41.0.0",
]
```

## Security Best Practices

### Code Security

1. **Input Validation**
   - All user inputs are validated using `InputValidator` class
   - Dangerous patterns and keywords are blocked
   - Path traversal attempts are prevented

2. **Safe Expression Evaluation**
   - Calculator server uses AST-based safe evaluation instead of `eval()`
   - Only whitelisted operations and functions are allowed
   - No arbitrary code execution possible

3. **Secure Configuration Management**
   - API keys and secrets stored in environment variables
   - Configuration files support `${ENV_VAR}` expansion
   - No secrets committed to repository

4. **Error Handling**
   - Structured exception hierarchy prevents information leakage
   - Error messages sanitized for production use
   - Logging configured to avoid exposing sensitive data

### Network Security

1. **API Security**
   - FastAPI endpoints with input validation
   - Rate limiting on all interfaces
   - CORS properly configured
   - No debug mode in production

2. **WebSocket Security**
   - Client connection management and validation
   - Message size limits to prevent DoS
   - Proper connection cleanup and resource management

3. **TLS/HTTPS**
   - All external API calls use HTTPS
   - Certificate validation enabled
   - No insecure HTTP connections

### Authentication & Authorization

1. **API Key Management**
   - Support for multiple AI provider APIs (Anthropic, OpenAI, Gemini)
   - Keys stored securely in environment variables
   - Key rotation supported through configuration reload

2. **Access Controls**
   - Input validation prevents unauthorized operations
   - File path validation prevents directory traversal
   - Tool execution sandboxing through MCP protocol

## Security Architecture

### Threat Model

1. **Input Injection**
   - **Risk**: Malicious input in queries, calculations, or file paths
   - **Mitigation**: Comprehensive input validation, safe evaluation

2. **Dependency Vulnerabilities**
   - **Risk**: Vulnerable third-party packages
   - **Mitigation**: Automated scanning, regular updates

3. **Information Disclosure**
   - **Risk**: Sensitive data exposure in logs or responses
   - **Mitigation**: Structured logging, error sanitization

4. **Code Injection**
   - **Risk**: Arbitrary code execution through eval() or similar
   - **Mitigation**: AST-based safe evaluation, input sanitization

### Security Layers

```
┌─────────────────────────────────────────┐
│ Input Validation & Sanitization Layer   │
├─────────────────────────────────────────┤
│ Rate Limiting & Access Control Layer    │
├─────────────────────────────────────────┤
│ Application Logic & Business Rules      │
├─────────────────────────────────────────┤
│ Secure Communication Layer (TLS/HTTPS)  │
├─────────────────────────────────────────┤
│ Infrastructure & Environment Security   │
└─────────────────────────────────────────┘
```

## Vulnerability Disclosure

### Reporting Security Issues

If you discover a security vulnerability in MCP Client, please follow responsible disclosure:

1. **DO NOT** open a public GitHub issue
2. Email security concerns to: security@mcpclient.dev
3. Include detailed information about the vulnerability
4. Allow reasonable time for response and remediation

### Response Process

1. **Acknowledgment**: Within 24-48 hours
2. **Initial Assessment**: Within 1 week
3. **Remediation**: Based on severity (High: 7 days, Medium: 30 days)
4. **Disclosure**: Coordinated disclosure after fix deployment

### Severity Levels

- **Critical**: Remote code execution, data breach
- **High**: Privilege escalation, significant data exposure
- **Medium**: Limited information disclosure, DoS vulnerabilities
- **Low**: Minor security improvements

## Security Maintenance

### Regular Security Tasks

1. **Weekly**
   - Automated dependency vulnerability scans
   - Pre-commit hooks validation

2. **Monthly**
   - Manual security code review
   - Dependency updates and testing
   - Security configuration review

3. **Quarterly**
   - Comprehensive security audit
   - Threat model review and updates
   - Security training and documentation updates

### Security Monitoring

- Automated vulnerability scanning in CI/CD
- Dependency update notifications
- Security advisories monitoring
- Community security disclosure tracking

## Security Compliance

### Standards Alignment

- **OWASP Top 10**: Protection against common web vulnerabilities
- **NIST Cybersecurity Framework**: Risk management practices
- **Secure Development Lifecycle**: Security-first development approach

### Privacy & Data Protection

1. **Data Minimization**
   - Only collect necessary data for functionality
   - No persistent storage of user queries or responses

2. **Data Processing**
   - All AI provider communications encrypted in transit
   - Local processing when possible
   - Configurable data retention policies

3. **Logging & Monitoring**
   - Structured logging without sensitive data
   - Configurable log levels and retention
   - No user data in application logs

## Security Resources

### Documentation
- [OWASP Python Security Guidelines](https://owasp.org/www-project-code-review-guide/)
- [Python Security Best Practices](https://python.org/dev/security/)
- [Bandit Security Linting](https://bandit.readthedocs.io/)

### Tools & Resources
- [pip-audit Documentation](https://pypi.org/project/pip-audit/)
- [Safety CLI Documentation](https://safetycli.com/)
- [Pre-commit Security Hooks](https://pre-commit.com/hooks.html)

---

**Last Updated**: 2025-09-25
**Next Review**: 2025-12-25

For questions about this security policy, contact: security@mcpclient.dev