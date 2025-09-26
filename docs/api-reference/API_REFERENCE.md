# 🌐 REST API Reference Guide

Complete reference for the MCP Client REST API interface with detailed examples.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Authentication](#authentication)
- [Base URL & Endpoints](#base-url--endpoints)
- [Request/Response Formats](#requestresponse-formats)
- [API Endpoints](#api-endpoints)
- [Complete Examples](#complete-examples)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)

---

## 🎯 Overview

The MCP Client REST API provides HTTP endpoints to interact with Model Context Protocol servers. It's built with **FastAPI** and provides automatic API documentation.

### Key Features
- **RESTful Design** - Standard HTTP methods and status codes
- **JSON Responses** - Structured, predictable response format
- **AI-Enhanced Processing** - Chain-of-thought reasoning and context optimization
- **Multi-Server Support** - Connect to multiple MCP servers simultaneously
- **Tool Execution** - Direct tool calling capabilities
- **Health Monitoring** - Built-in health checks and status endpoints

---

## 🚀 Getting Started

### Start the REST API Server
```bash
# Start API server on default port 8000
mcp-client --interfaces rest_api

# Start on custom port
mcp-client --interfaces rest_api --api-port 8080

# Start with debug mode
mcp-client --interfaces rest_api --api-port 8000 --debug
```

### Verify Server is Running
```bash
# Check if server is running
curl -I http://127.0.0.1:8000

# Get basic API information
curl http://127.0.0.1:8000/
```

### Access Interactive Documentation
- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

---

## 🔐 Authentication

Currently, the API doesn't require authentication for simplicity. For production deployments, consider adding:
- API key authentication
- JWT tokens
- OAuth 2.0

---

## 🌍 Base URL & Endpoints

### Base URL
```
http://127.0.0.1:8000
```

### Available Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API status and information |
| `POST` | `/query` | Process natural language queries |
| `POST` | `/tools/call` | Direct tool execution |
| `GET` | `/server/info` | Connected servers information |
| `GET` | `/server/health` | Server health status |
| `GET` | `/tools/list` | List all available tools |
| `POST` | `/conversation/reset` | Reset conversation history |
| `GET` | `/stats` | API usage statistics |

---

## 📝 Request/Response Formats

### Standard Response Format
```json
{
  "query": "string",
  "response": "string",
  "processing_time": 1.234,
  "tools_used": ["tool1", "tool2"],
  "timestamp": 1234567890.123,
  "ai_enhancements": {
    "context_optimization": true,
    "chain_of_thought": true,
    "quality_score": 8.7
  }
}
```

### Error Response Format
```json
{
  "detail": "Error description",
  "error_code": "ERROR_TYPE",
  "timestamp": 1234567890.123,
  "request_id": "unique-request-id"
}
```

---

## 🛠️ API Endpoints

## 1. Root Endpoint - API Status

### `GET /`

Get basic API information and status.

#### curl Example
```bash
curl -X GET http://127.0.0.1:8000/ \
  -H "Accept: application/json"
```

#### Response
```json
{
  "service": "MCP Client REST API",
  "version": "1.0.0",
  "status": "operational",
  "interfaces": ["chatbot", "rest_api", "websocket"],
  "connected_servers": 1,
  "available_tools": 93,
  "uptime": "2h 34m 12s",
  "timestamp": 1758768459.837
}
```

---

## 2. Query Processing Endpoint

### `POST /query`

Process natural language queries using AI-enhanced reasoning.

#### Request Body
```json
{
  "query": "string (required)",
  "context": {
    "user_id": "optional",
    "session_id": "optional",
    "additional_context": "optional"
  }
}
```

#### curl Examples

##### Basic Query
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "query": "What is the current time?"
  }'
```

##### Database Query
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "query": "Show me the top 10 customers by loan amount"
  }'
```

##### Complex Analysis Query
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "query": "Analyze database performance and provide optimization recommendations",
    "context": {
      "user_id": "analyst_001",
      "session_id": "session_123"
    }
  }'
```

#### Response
```json
{
  "query": "Show me the top 10 customers by loan amount",
  "response": "Here are the top 10 customers by loan amount:\n\n1. John Smith - $450,000\n2. Maria Garcia - $425,000\n...",
  "processing_time": 2.34,
  "tools_used": [
    "get_database_statistics",
    "execute_sql_query",
    "format_results"
  ],
  "timestamp": 1758768459.837,
  "ai_enhancements": {
    "context_optimization": true,
    "chain_of_thought": true,
    "quality_score": 9.2
  }
}
```

---

## 3. Direct Tool Execution

### `POST /tools/call`

Execute a specific tool directly with provided arguments.

#### Request Body
```json
{
  "tool_name": "string (required)",
  "arguments": {
    "arg1": "value1",
    "arg2": "value2"
  }
}
```

#### curl Examples

##### Execute SQL Query
```bash
curl -X POST http://127.0.0.1:8000/tools/call \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "tool_name": "execute_sql_query",
    "arguments": {
      "query": "SELECT COUNT(*) FROM customers"
    }
  }'
```

##### Get Database Schema
```bash
curl -X POST http://127.0.0.1:8000/tools/call \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "tool_name": "get_database_schema",
    "arguments": {
      "schema_name": "C##LOAN_SCHEMA"
    }
  }'
```

#### Response
```json
{
  "tool_name": "execute_sql_query",
  "arguments": {
    "query": "SELECT COUNT(*) FROM customers"
  },
  "result": {
    "columns": ["COUNT(*)"],
    "rows": [["1,247"]],
    "row_count": 1
  },
  "processing_time": 0.45,
  "timestamp": 1758768459.837
}
```

---

## 4. Server Information

### `GET /server/info`

Get information about connected MCP servers.

#### curl Example
```bash
curl -X GET http://127.0.0.1:8000/server/info \
  -H "Accept: application/json"
```

#### Response
```json
{
  "connected_servers": [
    {
      "name": "oracle-db",
      "description": "Oracle Database MCP Server with 93 tools",
      "status": "connected",
      "tools_count": 93,
      "connection_time": "2024-01-15T10:30:00Z"
    }
  ],
  "total_servers": 1,
  "total_tools": 93,
  "timestamp": 1758768459.837
}
```

---

## 5. Health Check

### `GET /server/health`

Check the health status of the API and connected servers.

#### curl Example
```bash
curl -X GET http://127.0.0.1:8000/server/health \
  -H "Accept: application/json"
```

#### Response
```json
{
  "status": "healthy",
  "api_server": {
    "status": "operational",
    "uptime": "2h 34m 12s",
    "memory_usage": "156 MB",
    "cpu_usage": "5.2%"
  },
  "mcp_servers": [
    {
      "name": "oracle-db",
      "status": "connected",
      "response_time": "0.023s",
      "last_check": "2024-01-15T12:45:30Z"
    }
  ],
  "ai_enhancements": {
    "context_manager": "active",
    "cot_engine": "active",
    "quality_optimizer": "active",
    "performance_tracker": "active"
  },
  "timestamp": 1758768459.837
}
```

---

## 6. Tools List

### `GET /tools/list`

Get a list of all available tools from connected servers.

#### curl Examples

##### List All Tools
```bash
curl -X GET http://127.0.0.1:8000/tools/list \
  -H "Accept: application/json"
```

##### List Tools with Filtering
```bash
curl -X GET "http://127.0.0.1:8000/tools/list?category=database&limit=10" \
  -H "Accept: application/json"
```

#### Response
```json
{
  "tools": [
    {
      "name": "execute_sql_query",
      "description": "Execute SQL queries on the database",
      "server": "oracle-db",
      "category": "database",
      "parameters": [
        {
          "name": "query",
          "type": "string",
          "required": true,
          "description": "SQL query to execute"
        }
      ]
    },
    {
      "name": "get_database_schema",
      "description": "Get database schema information",
      "server": "oracle-db",
      "category": "database",
      "parameters": [
        {
          "name": "schema_name",
          "type": "string",
          "required": false,
          "description": "Name of the schema"
        }
      ]
    }
  ],
  "total_tools": 93,
  "categories": ["database", "analysis", "optimization"],
  "servers": ["oracle-db"],
  "timestamp": 1758768459.837
}
```

---

## 7. Conversation Reset

### `POST /conversation/reset`

Reset the conversation history and start fresh.

#### curl Example
```bash
curl -X POST http://127.0.0.1:8000/conversation/reset \
  -H "Accept: application/json"
```

#### Response
```json
{
  "status": "success",
  "message": "Conversation history has been reset",
  "timestamp": 1758768459.837
}
```

---

## 8. API Statistics

### `GET /stats`

Get API usage statistics and performance metrics.

#### curl Example
```bash
curl -X GET http://127.0.0.1:8000/stats \
  -H "Accept: application/json"
```

#### Response
```json
{
  "requests": {
    "total": 1247,
    "successful": 1198,
    "failed": 49,
    "success_rate": "96.1%"
  },
  "performance": {
    "average_response_time": "1.23s",
    "fastest_response": "0.12s",
    "slowest_response": "15.67s"
  },
  "popular_tools": [
    {
      "tool": "execute_sql_query",
      "usage_count": 456
    },
    {
      "tool": "get_database_statistics",
      "usage_count": 234
    }
  ],
  "uptime": "2h 34m 12s",
  "timestamp": 1758768459.837
}
```

---

## 🎯 Complete Examples

### Business Intelligence Query
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "query": "Generate a comprehensive business intelligence report for our loan portfolio, including customer segmentation, risk analysis, and performance metrics for the last quarter"
  }'
```

### Database Performance Analysis
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "query": "Analyze database performance and identify optimization opportunities. Include table sizes, index usage, and query performance statistics."
  }'
```

### Customer Risk Assessment
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "query": "Perform a risk assessment for all customers with outstanding loans above $100,000. Include payment history, credit scores, and risk recommendations."
  }'
```

### Batch Processing Script
```bash
#!/bin/bash

# Batch API queries script
API_BASE="http://127.0.0.1:8000"

echo "Running batch queries..."

# 1. Check API health
echo "1. Checking API health..."
curl -s "${API_BASE}/server/health" | jq '.status'

# 2. Get server info
echo "2. Getting server info..."
curl -s "${API_BASE}/server/info" | jq '.total_tools'

# 3. List available tools
echo "3. Listing tools..."
curl -s "${API_BASE}/tools/list?limit=5" | jq '.tools[].name'

# 4. Execute queries
echo "4. Executing business queries..."

QUERIES=(
  "Show customer demographics"
  "Analyze loan performance"
  "Generate monthly report"
)

for query in "${QUERIES[@]}"; do
  echo "Processing: $query"
  curl -s -X POST "${API_BASE}/query" \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"$query\"}" \
    | jq '.processing_time'
done

echo "Batch processing complete!"
```

---

## ❌ Error Handling

### HTTP Status Codes

| Code | Status | Description |
|------|--------|-------------|
| `200` | OK | Request successful |
| `400` | Bad Request | Invalid request format |
| `422` | Unprocessable Entity | Validation error |
| `429` | Too Many Requests | Rate limit exceeded |
| `500` | Internal Server Error | Server error |
| `503` | Service Unavailable | MCP servers unavailable |

### Error Response Examples

#### Validation Error (422)
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"invalid": "request"}'
```

Response:
```json
{
  "detail": [
    {
      "loc": ["body", "query"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ],
  "error_code": "VALIDATION_ERROR",
  "timestamp": 1758768459.837
}
```

#### Server Error (500)
```bash
# Response when MCP server is unavailable
{
  "detail": "Internal server error: MCP server connection failed",
  "error_code": "MCP_CONNECTION_ERROR",
  "timestamp": 1758768459.837,
  "request_id": "req_12345"
}
```

#### Rate Limit Error (429)
```bash
# Response when rate limit is exceeded
{
  "detail": "Rate limit exceeded: 100 requests per hour",
  "error_code": "RATE_LIMIT_EXCEEDED",
  "retry_after": 3600,
  "timestamp": 1758768459.837
}
```

---

## 🔄 Rate Limiting

### Default Limits
- **100 requests per hour** per IP address
- **Burst limit**: 10 requests per minute

### Rate Limit Headers
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1758772059
```

### Handling Rate Limits
```bash
# Check rate limit status
curl -I http://127.0.0.1:8000/query

# Example with rate limit headers
HTTP/1.1 200 OK
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1758772059
```

---

## 🎯 Best Practices

### 1. Error Handling
```bash
# Always check HTTP status codes
response=$(curl -s -w "%{http_code}" http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}')

http_code=${response: -3}
if [ $http_code -eq 200 ]; then
  echo "Success!"
else
  echo "Error: HTTP $http_code"
fi
```

### 2. Timeout Handling
```bash
# Set appropriate timeouts for long-running queries
curl --max-time 30 -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Complex analysis query"}'
```

### 3. JSON Processing
```bash
# Use jq for JSON parsing
curl -s http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}' \
  | jq '.response'
```

### 4. Logging and Monitoring
```bash
# Log requests for debugging
curl -v http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}' \
  2>&1 | tee request.log
```

---

**🌐 Your REST API is ready for professional integration!**