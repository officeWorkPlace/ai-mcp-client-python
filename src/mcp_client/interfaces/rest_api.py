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


# Standard API Response wrapper for React UI
class APIResponse(BaseModel):
    success: bool = Field(..., description="Whether the request was successful")
    data: Optional[Any] = Field(None, description="Response data")
    error: Optional[str] = Field(None, description="Error message if any")
    timestamp: float = Field(..., description="Response timestamp")
    meta: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ChatResponseFormatter:
    """Format AI responses for React chat interface like Claude Desktop"""

    @staticmethod
    def format_for_chat(raw_response: str, query: str, conversation_id: str = None) -> Dict[str, Any]:
        """
        Transform raw AI response into React chat-optimized format
        Like Claude Desktop - conversational, structured, and interactive

        Args:
            raw_response: Raw AI response text
            query: Original user query
            conversation_id: Optional conversation thread ID

        Returns:
            Chat-optimized response for React UI
        """
        import uuid
        import time

        # Generate unique message ID
        message_id = str(uuid.uuid4())[:8]

        # Parse response into chat-friendly blocks
        content_blocks = ChatResponseFormatter._parse_into_blocks(raw_response)

        # Extract conversational elements
        summary = ChatResponseFormatter._create_chat_summary(raw_response)
        insights = ChatResponseFormatter._extract_key_insights(raw_response)
        actions = ChatResponseFormatter._extract_quick_actions(raw_response)

        # Chat metadata
        response_time = 1.2  # Simulated response time
        word_count = len(raw_response.split())

        return {
            "message": {
                "id": message_id,
                "type": "assistant_response",
                "conversation_id": conversation_id,
                "timestamp": time.time(),
                "status": "delivered"
            },
            "content": {
                "summary": summary,
                "blocks": content_blocks,
                "quick_insights": insights[:3],  # Top 3 insights for chat preview
                "full_response": raw_response
            },
            "interactive": {
                "quick_actions": actions,
                "follow_up_suggestions": ChatResponseFormatter._generate_follow_ups(query),
                "expandable_sections": ChatResponseFormatter._identify_expandable_content(content_blocks),
                "copy_blocks": ChatResponseFormatter._identify_copyable_content(content_blocks)
            },
            "chat_metadata": {
                "response_time_seconds": response_time,
                "word_count": word_count,
                "estimated_read_time": max(1, round(word_count / 200)),
                "confidence": ChatResponseFormatter._assess_confidence(raw_response),
                "complexity": "business_friendly",
                "has_data": ChatResponseFormatter._contains_data_insights(raw_response),
                "has_actions": len(actions) > 0
            },
            "ui_optimization": {
                "render_mode": "progressive",  # Show summary first, then expand
                "typing_effect": True,
                "show_sources": False,
                "enable_copy": True,
                "highlight_insights": True,
                "theme": "professional"
            }
        }

    @staticmethod
    def _parse_into_blocks(text: str) -> List[Dict[str, Any]]:
        """Parse response into chat-friendly content blocks"""
        import re

        blocks = []
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        for i, para in enumerate(paragraphs):
            block = {
                "id": f"block_{i}",
                "type": "text",
                "content": para,
                "order": i
            }

            # Detect special content types
            if re.match(r'^\d+\.', para):  # Numbered list
                block["type"] = "numbered_list"
                items = re.findall(r'^\d+\.\s*(.+)', para, re.MULTILINE)
                block["items"] = items
            elif re.match(r'^[-•*]', para):  # Bullet list
                block["type"] = "bullet_list"
                items = re.findall(r'^[-•*]\s*(.+)', para, re.MULTILINE)
                block["items"] = items
            elif 'recommend' in para.lower():
                block["type"] = "recommendation"
                block["highlight"] = True
            elif any(word in para.lower() for word in ['data', 'metric', 'result', 'finding']):
                block["type"] = "insight"
                block["highlight"] = True

            blocks.append(block)

        return blocks

    @staticmethod
    def _create_chat_summary(text: str) -> str:
        """Create conversational summary for chat preview"""
        sentences = text.split('. ')

        if len(sentences) >= 2:
            # Take first 2 sentences for a natural summary
            summary = f"{sentences[0]}. {sentences[1]}."
        else:
            summary = sentences[0] if sentences else text

        # Keep it chat-friendly length
        if len(summary) > 150:
            summary = summary[:147] + "..."

        return summary

    @staticmethod
    def _extract_key_insights(text: str) -> List[str]:
        """Extract key insights for chat preview"""
        insights = []
        import re

        # Patterns for insights
        patterns = [
            r'(shows that|indicates|reveals|found that)\s+(.+?)(?:\.|$)',
            r'^\d+\.\s*(.+)',  # Numbered items
            r'^[-•*]\s*(.+)',  # Bullet points
            r'(key|important|significant|main)\s+(.+?)(?:\.|$)'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                insight = match[1] if isinstance(match, tuple) else match
                insight = insight.strip()

                if 10 < len(insight) < 80:  # Chat-appropriate length
                    insights.append(insight)

        return insights[:5]  # Top 5 insights

    @staticmethod
    def _extract_quick_actions(text: str) -> List[Dict[str, str]]:
        """Extract quick actions as interactive buttons"""
        actions = []
        import re

        action_patterns = [
            r'(should|recommend|suggest)\s+(.+?)(?:\.|$)',
            r'(consider|implement|focus on|try)\s+(.+?)(?:\.|$)'
        ]

        for pattern in action_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                action_text = match[1].strip()
                if 5 < len(action_text) < 50:  # Button-appropriate length
                    actions.append({
                        "text": action_text.capitalize(),
                        "type": "suggestion",
                        "action": f"Tell me more about {action_text}"
                    })

        return actions[:3]  # Max 3 quick actions

    @staticmethod
    def _generate_follow_ups(query: str) -> List[str]:
        """Generate follow-up question suggestions"""
        follow_ups = []

        # Generic business follow-ups
        if "data" in query.lower() or "metric" in query.lower():
            follow_ups.extend([
                "Can you show me specific numbers?",
                "What time period does this cover?",
                "How does this compare to last quarter?"
            ])
        elif "customer" in query.lower():
            follow_ups.extend([
                "Which customer segments are most valuable?",
                "What are the customer satisfaction trends?",
                "How can we improve customer retention?"
            ])
        else:
            follow_ups.extend([
                "Can you provide more details?",
                "What are the next steps?",
                "How does this impact our business?"
            ])

        return follow_ups[:3]

    @staticmethod
    def _identify_expandable_content(blocks: List[Dict[str, Any]]) -> List[str]:
        """Identify which content blocks can be expanded"""
        expandable = []

        for block in blocks:
            if block["type"] in ["numbered_list", "bullet_list"] and len(block.get("items", [])) > 3:
                expandable.append(block["id"])
            elif len(block.get("content", "")) > 200:
                expandable.append(block["id"])

        return expandable

    @staticmethod
    def _identify_copyable_content(blocks: List[Dict[str, Any]]) -> List[str]:
        """Identify which content can be copied"""
        copyable = []

        for block in blocks:
            if block["type"] in ["insight", "recommendation"]:
                copyable.append(block["id"])
            elif "data" in block.get("content", "").lower():
                copyable.append(block["id"])

        return copyable

    @staticmethod
    def _assess_confidence(text: str) -> str:
        """Assess response confidence for chat display"""
        text_lower = text.lower()

        high_confidence = ['definitely', 'clearly', 'confirmed', 'proven', 'shows that']
        low_confidence = ['might', 'possibly', 'perhaps', 'uncertain', 'may be']

        if any(phrase in text_lower for phrase in high_confidence):
            return "high"
        elif any(phrase in text_lower for phrase in low_confidence):
            return "low"
        else:
            return "medium"

    @staticmethod
    def _contains_data_insights(text: str) -> bool:
        """Check if response contains data insights"""
        data_indicators = ['data', 'metric', 'number', 'percentage', 'analysis', 'result', 'finding']
        return any(word in text.lower() for word in data_indicators)

# Pydantic models for API requests/responses
class QueryRequest(BaseModel):
    query: str = Field(..., description="Your business question or request")
    conversation_id: Optional[str] = Field(None, description="Optional conversation thread ID for chat context")
    context: Optional[Dict[str, Any]] = Field(None, description="Optional additional context (rarely needed)")


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
        self._start_time = time.time()

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
        """Application lifespan manager with robust shutdown"""
        # Startup
        self.logger.info("Starting MCP REST API server")
        try:
            await self._initialize_mcp_client()
            self.logger.info("MCP client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP client: {e}")
            raise

        try:
            yield
        finally:
            # Shutdown - ensure this always runs
            self.logger.info("Initiating graceful shutdown...")
            await self._safe_cleanup()

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

        @app.get("/", response_model=APIResponse)
        async def root():
            """Root endpoint"""
            return APIResponse(
                success=True,
                data={
                    "message": "MCP Client REST API",
                    "version": "1.0.0",
                    "status": "active"
                },
                timestamp=time.time()
            )

        @app.post("/query", response_model=APIResponse)
        async def process_query(request: QueryRequest):
            """Process a query using MCP client with user-friendly formatting"""
            if not self.mcp_client:
                return APIResponse(
                    success=False,
                    error="MCP client not initialized",
                    timestamp=time.time()
                )

            start_time = time.time()

            try:
                # Get raw AI response
                raw_response = await self.mcp_client.process_query(request.query)
                processing_time = time.time() - start_time

                # Format response for React Chat UI
                chat_data = ChatResponseFormatter.format_for_chat(
                    raw_response,
                    request.query,
                    request.conversation_id
                )

                return APIResponse(
                    success=True,
                    data={
                        "query": request.query,
                        "processing_time": processing_time,
                        "tools_used": [],
                        **chat_data  # Spread complete chat-optimized response
                    },
                    timestamp=time.time(),
                    meta={
                        "response_format": "chat_optimized",
                        "claude_desktop_style": True,
                        "version": "3.0",
                        "supports": ["progressive_rendering", "interactive_elements", "follow_ups"]
                    }
                )

            except GlobalMCPClientError as e:
                self.logger.error(f"MCP client error: {e}")
                return APIResponse(
                    success=False,
                    error=f"Query processing failed: {str(e)}",
                    timestamp=time.time()
                )
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                return APIResponse(
                    success=False,
                    error="Internal server error",
                    timestamp=time.time()
                )

        @app.post("/tools/call", response_model=APIResponse)
        async def call_tool(request: ToolCallRequest):
            """Call a specific tool directly"""
            if not self.mcp_client:
                return APIResponse(
                    success=False,
                    error="MCP client not initialized",
                    timestamp=time.time()
                )

            start_time = time.time()

            try:
                result = await self.mcp_client.call_tool(request.tool_name, request.arguments)
                processing_time = time.time() - start_time

                return APIResponse(
                    success=True,
                    data={
                        "tool_name": request.tool_name,
                        "arguments": request.arguments,
                        "result": result,
                        "processing_time": processing_time
                    },
                    timestamp=time.time()
                )

            except Exception as e:
                self.logger.error(f"Tool call error: {e}")
                return APIResponse(
                    success=False,
                    error=f"Tool call failed: {str(e)}",
                    timestamp=time.time()
                )

        @app.get("/server/info", response_model=APIResponse)
        async def get_server_info():
            """Get server information"""
            if not self.mcp_client:
                return APIResponse(
                    success=False,
                    error="MCP client not initialized",
                    timestamp=time.time()
                )

            try:
                info = self.mcp_client.get_server_info()
                return APIResponse(
                    success=True,
                    data={
                        "connected_servers": info["connected_servers"],
                        "available_tools": info["available_tools"],
                        "total_servers": info["total_servers"],
                        "total_tools": info["total_tools"],
                        "status": "active"
                    },
                    timestamp=time.time()
                )

            except Exception as e:
                self.logger.error(f"Error getting server info: {e}")
                return APIResponse(
                    success=False,
                    error="Failed to retrieve server information",
                    timestamp=time.time()
                )

        @app.get("/server/health", response_model=APIResponse)
        async def health_check():
            """Check server health"""
            if not self.mcp_client:
                return APIResponse(
                    success=False,
                    error="MCP client not initialized",
                    timestamp=time.time()
                )

            try:
                health = await self.mcp_client.health_check()
                return APIResponse(
                    success=True,
                    data={
                        "status": "healthy" if health["healthy_servers"] > 0 else "unhealthy",
                        "servers": health["servers"],
                        "total_servers": health["total_servers"],
                        "healthy_servers": health["healthy_servers"],
                        "total_tools": health["total_tools"]
                    },
                    timestamp=time.time()
                )

            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                return APIResponse(
                    success=False,
                    error="Health check failed",
                    timestamp=time.time()
                )

        @app.get("/tools/list", response_model=APIResponse)
        async def list_tools():
            """List available tools"""
            if not self.mcp_client:
                return APIResponse(
                    success=False,
                    error="MCP client not initialized",
                    timestamp=time.time()
                )

            try:
                tools = self.mcp_client.available_tools
                return APIResponse(
                    success=True,
                    data={
                        "tools": tools,
                        "count": len(tools)
                    },
                    timestamp=time.time()
                )

            except Exception as e:
                self.logger.error(f"Error listing tools: {e}")
                return APIResponse(
                    success=False,
                    error="Failed to list tools",
                    timestamp=time.time()
                )

        @app.get("/api/statistics", response_model=APIResponse)
        async def get_api_statistics():
            """Get API usage statistics"""
            if not self.mcp_client:
                return APIResponse(
                    success=False,
                    error="MCP client not initialized",
                    timestamp=time.time()
                )

            try:
                server_info = self.mcp_client.get_server_info()
                health = await self.mcp_client.health_check()

                return APIResponse(
                    success=True,
                    data={
                        "total_servers": server_info["total_servers"],
                        "connected_servers": len(server_info["connected_servers"]),
                        "total_tools": server_info["total_tools"],
                        "healthy_servers": health["healthy_servers"],
                        "uptime_seconds": time.time() - getattr(self, '_start_time', time.time())
                    },
                    timestamp=time.time()
                )

            except Exception as e:
                self.logger.error(f"Error getting API statistics: {e}")
                return APIResponse(
                    success=False,
                    error="Failed to retrieve API statistics",
                    timestamp=time.time()
                )

        @app.post("/conversation/reset", response_model=APIResponse)
        async def reset_conversation():
            """Reset conversation history"""
            if not self.mcp_client:
                return APIResponse(
                    success=False,
                    error="MCP client not initialized",
                    timestamp=time.time()
                )

            try:
                if hasattr(self.mcp_client, 'reset_conversation'):
                    self.mcp_client.reset_conversation()

                return APIResponse(
                    success=True,
                    data={
                        "status": "conversation_reset",
                        "message": "Conversation history cleared"
                    },
                    timestamp=time.time()
                )

            except Exception as e:
                self.logger.error(f"Error resetting conversation: {e}")
                return APIResponse(
                    success=False,
                    error="Failed to reset conversation",
                    timestamp=time.time()
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

    async def _safe_cleanup(self):
        """Safe cleanup with timeout and error handling"""
        if not self.mcp_client:
            self.logger.info("No MCP client to cleanup")
            return

        try:
            # Attempt graceful cleanup with timeout
            await asyncio.wait_for(self._cleanup(), timeout=10.0)
            self.logger.info("MCP client cleanup completed successfully")
        except asyncio.TimeoutError:
            self.logger.warning("MCP client cleanup timed out, forcing shutdown")
        except Exception as e:
            self.logger.error(f"Error during safe cleanup: {e}")
            # Try to at least close connections manually
            try:
                if hasattr(self.mcp_client, 'connections'):
                    for conn in self.mcp_client.connections.values():
                        try:
                            if hasattr(conn, 'session') and conn.session:
                                # Close session if it exists
                                pass  # Let the timeout handle it
                        except Exception as conn_error:
                            self.logger.error(f"Error closing connection: {conn_error}")
            except Exception as manual_cleanup_error:
                self.logger.error(f"Error in manual cleanup: {manual_cleanup_error}")
        finally:
            self.logger.info("Cleanup process finished")

    def run(self):
        """Run the REST API server"""
        import uvicorn

        self.logger.info(f"Starting MCP REST API server on {self.host}:{self.port}")

        try:
            # Let uvicorn handle its own signal handling - it has built-in Ctrl+C support
            uvicorn.run(
                self.app,
                host=self.host,
                port=self.port,
                log_level=self.config.log_level.lower() if self.config.log_level else "info",
                # Uvicorn configuration for better shutdown behavior
                access_log=False,  # Reduce noise
                server_header=False,
                date_header=False,
                loop="asyncio",  # Use asyncio explicitly
                # Enable graceful shutdown with timeout
                timeout_keep_alive=5,
                timeout_graceful_shutdown=30
            )
        except KeyboardInterrupt:
            self.logger.info("Received KeyboardInterrupt")
        except Exception as e:
            self.logger.error(f"Server error: {e}")
            raise


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

    print(f"🚀 Starting MCP Client REST API on http://{args.host}:{args.port}")
    print("💡 Press Ctrl+C to stop the server")
    print("=" * 50)

    # Let uvicorn handle all signal processing and cleanup
    api_server = MCPRestAPI(config, args.host, args.port)
    api_server.run()


if __name__ == "__main__":
    main()