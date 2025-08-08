"""MCP Server Implementation for DevGuard.

This module implements the Model Context Protocol server that exposes
DevGuard's capabilities to IDEs and external tools.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    MCPCapabilities,
    MCPError,
    MCPInitialize,
    MCPInitializeResult,
    MCPRequest,
    MCPResponse,
)
from .tools import (
    BaseMCPTool,
    CodeContextTool,
    DependencyAnalysisTool,
    ImpactAnalysisTool,
    PatternSearchTool,
    RecommendationTool,
    SecurityScanTool,
)

if TYPE_CHECKING:
    from ..agents.base_agent import BaseAgent
    from ..memory.shared_memory import SharedMemory
    from ..memory.vector_db import VectorDatabase

logger = logging.getLogger(__name__)


class MCPServer:
    """Model Context Protocol server for DevGuard."""

    def __init__(
        self,
        shared_memory: SharedMemory | None = None,
        vector_db: VectorDatabase | None = None,
        agents: dict[str, BaseAgent] | None = None,
        host: str = "localhost",
        port: int = 8080,
    ):
        """Initialize MCP server."""
        self.shared_memory = shared_memory
        self.vector_db = vector_db
        self.agents = agents or {}
        self.host = host
        self.port = port
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="DevGuard MCP Server",
            description="Model Context Protocol server for DevGuard",
            version="1.0.0",
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize tools
        self.tools: dict[str, BaseMCPTool] = {}
        
        # Setup logger
        self.logger = logger
        
        # Initialize tools after logger is set
        self._initialize_tools()
        
        # Setup routes
        self._setup_routes()
        
        self.logger.info("MCP Server initialized")

    def _initialize_tools(self) -> None:
        """Initialize MCP tools."""
        tool_classes = [
            CodeContextTool,
            PatternSearchTool,
            DependencyAnalysisTool,
            ImpactAnalysisTool,
            SecurityScanTool,
            RecommendationTool,
        ]
        
        for tool_class in tool_classes:
            tool = tool_class(
                shared_memory=self.shared_memory,
                vector_db=self.vector_db,
                agents=self.agents,
            )
            self.tools[tool.name] = tool
            
        self.logger.info(f"Initialized {len(self.tools)} MCP tools")

    def _setup_routes(self) -> None:
        """Setup FastAPI routes."""
        
        @self.app.get("/")
        async def root():
            """Root endpoint."""
            return {
                "name": "DevGuard MCP Server",
                "version": "1.0.0",
                "protocol": "mcp/1.0",
                "capabilities": self.get_server_capabilities().to_dict(),
            }

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "tools": list(self.tools.keys()),
                "agents": list(self.agents.keys()),
            }

        @self.app.websocket("/mcp")
        async def mcp_websocket(websocket: WebSocket):
            """Main MCP WebSocket endpoint."""
            await self.handle_websocket(websocket)

        @self.app.post("/mcp/invoke")
        async def mcp_http_invoke(request: dict):
            """HTTP endpoint for MCP tool invocation."""
            try:
                mcp_request = MCPRequest(**request)
                response = await self.handle_request(mcp_request)
                return response.to_dict()
            except Exception as e:
                self.logger.error(f"HTTP invoke error: {e}")
                error = MCPError.internal_error(str(e))
                response = MCPResponse.error_response(
                    id=request.get("id"),
                    error=error
                )
                return response.to_dict()

    async def handle_websocket(self, websocket: WebSocket) -> None:
        """Handle WebSocket connections."""
        await websocket.accept()
        self.logger.info("WebSocket connection established")
        
        try:
            while True:
                # Receive message
                data = await websocket.receive_text()
                self.logger.debug(f"Received WebSocket message: {data}")
                
                try:
                    # Parse JSON-RPC request
                    request_data = json.loads(data)
                    mcp_request = MCPRequest(**request_data)
                    
                    # Handle request
                    response = await self.handle_request(mcp_request)
                    
                    # Send response
                    await websocket.send_text(response.to_json())
                    
                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON decode error: {e}")
                    error = MCPError.parse_error(str(e))
                    response = MCPResponse.error_response(
                        id=None,
                        error=error
                    )
                    await websocket.send_text(response.to_json())
                    
                except Exception as e:
                    self.logger.error(f"Request handling error: {e}")
                    error = MCPError.internal_error(str(e))
                    request_id = (
                        request_data.get("id")
                        if "request_data" in locals()
                        else None
                    )
                    response = MCPResponse.error_response(
                        id=request_id,
                        error=error
                    )
                    await websocket.send_text(response.to_json())
                    
        except WebSocketDisconnect:
            self.logger.info("WebSocket connection closed")

    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """Handle MCP request."""
        self.logger.debug(f"Handling request: {request.method}")
        
        try:
            if request.method == "initialize":
                return await self.handle_initialize(request)
            elif request.method == "tools/list":
                return await self.handle_tools_list(request)
            elif request.method == "tools/call":
                return await self.handle_tool_call(request)
            elif request.method == "resources/list":
                return await self.handle_resources_list(request)
            elif request.method == "prompts/list":
                return await self.handle_prompts_list(request)
            else:
                error = MCPError.method_not_found(request.method)
                return MCPResponse.error_response(request.id, error)
                
        except Exception as e:
            self.logger.error(f"Error handling {request.method}: {e}")
            error = MCPError.internal_error(str(e))
            return MCPResponse.error_response(request.id, error)

    async def handle_initialize(self, request: MCPRequest) -> MCPResponse:
        """Handle initialize request."""
        try:
            if request.params:
                initialize_params = MCPInitialize(**request.params)
                self.logger.info(
                    f"Client initialized: {initialize_params.client_info}"
                )
            
            result = MCPInitializeResult(
                protocol_version="2024-11-05",
                capabilities=self.get_server_capabilities(),
                server_info={
                    "name": "DevGuard MCP Server",
                    "version": "1.0.0",
                    "description": "DevGuard autonomous swarm MCP server",
                },
            )
            
            return MCPResponse.success(request.id, result.to_dict())
            
        except Exception as e:
            self.logger.error(f"Initialize error: {e}")
            error = MCPError.invalid_params(str(e))
            return MCPResponse.error_response(request.id, error)

    async def handle_tools_list(self, request: MCPRequest) -> MCPResponse:
        """Handle tools list request."""
        tools_list = []
        for tool in self.tools.values():
            tools_list.append(tool.to_mcp_tool().to_dict())
        
        result = {"tools": tools_list}
        return MCPResponse.success(request.id, result)

    async def handle_tool_call(self, request: MCPRequest) -> MCPResponse:
        """Handle tool call request."""
        if not request.params:
            error = MCPError.invalid_params("Missing tool call parameters")
            return MCPResponse.error_response(request.id, error)
        
        tool_name = request.params.get("name")
        tool_args = request.params.get("arguments", {})
        
        if not tool_name:
            error = MCPError.invalid_params("Missing tool name")
            return MCPResponse.error_response(request.id, error)
        
        if tool_name not in self.tools:
            error = MCPError.invalid_params(f"Unknown tool: {tool_name}")
            return MCPResponse.error_response(request.id, error)
        
        try:
            # Execute tool
            tool = self.tools[tool_name]
            result = await tool.execute(tool_args)
            
            # Wrap in MCP tool response format
            tool_response = {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, indent=2),
                    }
                ]
            }
            
            return MCPResponse.success(request.id, tool_response)
            
        except Exception as e:
            self.logger.error(f"Tool execution error: {e}")
            error = MCPError.internal_error(f"Tool execution failed: {str(e)}")
            return MCPResponse.error_response(request.id, error)

    async def handle_resources_list(self, request: MCPRequest) -> MCPResponse:
        """Handle resources list request."""
        # For now, return empty resources list
        # This could be extended to expose file system resources
        result = {"resources": []}
        return MCPResponse.success(request.id, result)

    async def handle_prompts_list(self, request: MCPRequest) -> MCPResponse:
        """Handle prompts list request."""
        # For now, return empty prompts list
        # This could be extended to expose predefined prompts
        result = {"prompts": []}
        return MCPResponse.success(request.id, result)

    def get_server_capabilities(self) -> MCPCapabilities:
        """Get server capabilities."""
        return MCPCapabilities(
            tools={
                "listChanged": True,
            },
            resources={
                "subscribe": False,
                "listChanged": False,
            },
            prompts={
                "listChanged": False,
            },
            logging={},
        )

    async def start(self) -> None:
        """Start the MCP server."""
        self.logger.info(f"Starting MCP server on {self.host}:{self.port}")
        
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
        )
        
        server = uvicorn.Server(config)
        await server.serve()

    def start_sync(self) -> None:
        """Start the MCP server synchronously."""
        asyncio.run(self.start())

    async def shutdown(self) -> None:
        """Shutdown the MCP server."""
        self.logger.info("Shutting down MCP server")
        # Cleanup resources if needed

    def add_tool(self, tool: BaseMCPTool) -> None:
        """Add a custom tool to the server."""
        self.tools[tool.name] = tool
        self.logger.info(f"Added custom tool: {tool.name}")

    def remove_tool(self, tool_name: str) -> None:
        """Remove a tool from the server."""
        if tool_name in self.tools:
            del self.tools[tool_name]
            self.logger.info(f"Removed tool: {tool_name}")

    def get_tool_info(self) -> dict[str, Any]:
        """Get information about available tools."""
        return {
            tool_name: {
                "name": tool.name,
                "description": tool.description,
                "parameters": [param.model_dump() for param in tool.parameters],
            }
            for tool_name, tool in self.tools.items()
        }
