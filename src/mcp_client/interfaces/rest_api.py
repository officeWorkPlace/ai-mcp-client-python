"""
REST API interface for MCP Client
Provides HTTP endpoints to interact with MCP services
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import from new structure
from ..core import GlobalMCPClient, Config, setup_logging
from ..core.exceptions import GlobalMCPClientError


# Pydantic models for API requests/responses
class QueryRequest(BaseModel):
    query: str = Field(..., description="The query to process")
    context: Optional[Dict[str, Any]] = Field(None, description="Optional context")


class QueryResponse(BaseModel):
    query: str
    response: str
    processing_time: float
    tools_used: List[str] = []
    timestamp: float


class ToolCallRequest(BaseModel):
    tool_name: str = Field(..., description="Name of the tool to call")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")


class ToolCallResponse(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    result: Any
    processing_time: float
    timestamp: float


class ServerInfo(BaseModel):
    connected_servers: List[str]
    available_tools: List[Dict[str, Any]]
    total_servers: int
    total_tools: int
    status: str


class HealthStatus(BaseModel):
    status: str
    servers: Dict[str, Dict[str, Any]]
    total_servers: int
    healthy_servers: int
    total_tools: int
    timestamp: float


class MCPRestAPI:
    """REST API server for MCP Client"""

    def __init__(self, config: Optional[Config] = None, host: str = "127.0.0.1", port: int = 8000):
        self.config = config or Config()
        self.host = host
        self.port = port
        self.mcp_client: Optional[GlobalMCPClient] = None
        self.app: Optional[FastAPI] = None

        # Setup logging
        self.logger = setup_logging(
            log_level=self.config.log_level,
            log_file=self.config.log_file,
            enable_file_logging=self.config.enable_file_logging,
            enable_console_logging=self.config.enable_console_logging,
            development_mode=self.config.development_mode,
        )

        # Create FastAPI app
        self.app = self._create_app()

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """Application lifespan manager"""
        # Startup
        self.logger.info("Starting MCP REST API server")
        await self._initialize_mcp_client()
        yield
        # Shutdown
        await self._cleanup()
        self.logger.info("MCP REST API server stopped")

    def _create_app(self) -> FastAPI:
        """Create FastAPI application"""
        app = FastAPI(
            title="MCP Client REST API",
            description="HTTP interface for Model Context Protocol Client",
            version="1.0.0",
            lifespan=self.lifespan
        )

        # Add routes
        self._setup_routes(app)
        return app

    def _setup_routes(self, app: FastAPI):
        """Setup API routes"""

        @app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint"""
            return {
                "message": "MCP Client REST API",
                "version": "1.0.0",
                "status": "active"
            }

        @app.post("/query", response_model=QueryResponse)
        async def process_query(request: QueryRequest):
            """Process a query using MCP client"""
            if not self.mcp_client:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="MCP client not initialized"
                )

            start_time = time.time()

            try:
                response = await self.mcp_client.process_query(request.query)
                processing_time = time.time() - start_time

                return QueryResponse(
                    query=request.query,
                    response=response,
                    processing_time=processing_time,
                    timestamp=time.time()
                )

            except GlobalMCPClientError as e:
                self.logger.error(f"MCP client error: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Query processing failed: {str(e)}"
                )
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error"
                )

        @app.post("/tools/call", response_model=ToolCallResponse)
        async def call_tool(request: ToolCallRequest):
            """Call a specific tool directly"""
            if not self.mcp_client:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="MCP client not initialized"
                )

            start_time = time.time()

            try:
                result = await self.mcp_client.call_tool(request.tool_name, request.arguments)
                processing_time = time.time() - start_time

                return ToolCallResponse(
                    tool_name=request.tool_name,
                    arguments=request.arguments,
                    result=result,
                    processing_time=processing_time,
                    timestamp=time.time()
                )

            except Exception as e:
                self.logger.error(f"Tool call error: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Tool call failed: {str(e)}"
                )

        @app.get("/server/info", response_model=ServerInfo)
        async def get_server_info():
            """Get server information"""
            if not self.mcp_client:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="MCP client not initialized"
                )

            try:
                info = self.mcp_client.get_server_info()
                return ServerInfo(
                    connected_servers=info["connected_servers"],
                    available_tools=info["available_tools"],
                    total_servers=info["total_servers"],
                    total_tools=info["total_tools"],
                    status="active"
                )

            except Exception as e:
                self.logger.error(f"Error getting server info: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to retrieve server information"
                )

        @app.get("/server/health", response_model=HealthStatus)
        async def health_check():
            """Check server health"""
            if not self.mcp_client:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="MCP client not initialized"
                )

            try:
                health = await self.mcp_client.health_check()
                return HealthStatus(
                    status="healthy" if health["healthy_servers"] > 0 else "unhealthy",
                    servers=health["servers"],
                    total_servers=health["total_servers"],
                    healthy_servers=health["healthy_servers"],
                    total_tools=health["total_tools"],
                    timestamp=time.time()
                )

            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Health check failed"
                )

        @app.get("/tools/list")
        async def list_tools():
            """List available tools"""
            if not self.mcp_client:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="MCP client not initialized"
                )

            try:
                tools = self.mcp_client.available_tools
                return {
                    "tools": tools,
                    "count": len(tools),
                    "timestamp": time.time()
                }

            except Exception as e:
                self.logger.error(f"Error listing tools: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to list tools"
                )

        @app.post("/conversation/reset")
        async def reset_conversation():
            """Reset conversation history"""
            if not self.mcp_client:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="MCP client not initialized"
                )

            try:
                if hasattr(self.mcp_client, 'reset_conversation'):
                    self.mcp_client.reset_conversation()

                return {
                    "status": "conversation_reset",
                    "message": "Conversation history cleared",
                    "timestamp": time.time()
                }

            except Exception as e:
                self.logger.error(f"Error resetting conversation: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to reset conversation"
                )

    async def _initialize_mcp_client(self):
        """Initialize MCP client"""
        try:
            self.mcp_client = GlobalMCPClient(self.config)
            await self.mcp_client.connect_to_all_servers()
            self.logger.info("MCP client initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize MCP client: {e}")
            raise

    async def _cleanup(self):
        """Cleanup resources"""
        if self.mcp_client:
            try:
                await self.mcp_client.cleanup()
            except Exception as e:
                self.logger.error(f"Error during cleanup: {e}")

    def run(self):
        """Run the REST API server"""
        import uvicorn

        self.logger.info(f"Starting MCP REST API server on {self.host}:{self.port}")

        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level=self.config.log_level.lower() if self.config.log_level else "info"
        )


def main():
    """Main function to run the REST API server"""
    import argparse

    parser = argparse.ArgumentParser(description="MCP Client REST API Server")
    parser.add_argument("--host", default="127.0.0.1", help="API server host")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Create server
    config = Config()
    if args.debug:
        config.log_level = "DEBUG"
        config.development_mode = True

    api_server = MCPRestAPI(config, args.host, args.port)
    api_server.run()


if __name__ == "__main__":
    main()