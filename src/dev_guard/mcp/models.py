"""MCP Protocol Data Models.

This module defines the data models for the Model Context Protocol (MCP)
server implementation in DevGuard.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MCPErrorCode(Enum):
    """Standard MCP error codes."""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603


class MCPError(BaseModel):
    """MCP error response model."""
    code: int = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    data: dict[str, Any] | None = Field(
        default=None, description="Additional error data"
    )

    @classmethod
    def parse_error(cls, message: str = "Parse error") -> MCPError:
        """Create a parse error."""
        return cls(code=MCPErrorCode.PARSE_ERROR.value, message=message)

    @classmethod
    def invalid_request(cls, message: str = "Invalid request") -> MCPError:
        """Create an invalid request error."""
        return cls(code=MCPErrorCode.INVALID_REQUEST.value, message=message)

    @classmethod
    def method_not_found(cls, method: str) -> MCPError:
        """Create a method not found error."""
        return cls(
            code=MCPErrorCode.METHOD_NOT_FOUND.value,
            message=f"Method not found: {method}"
        )

    @classmethod
    def invalid_params(cls, message: str = "Invalid params") -> MCPError:
        """Create an invalid params error."""
        return cls(code=MCPErrorCode.INVALID_PARAMS.value, message=message)

    @classmethod
    def internal_error(cls, message: str = "Internal error") -> MCPError:
        """Create an internal error."""
        return cls(code=MCPErrorCode.INTERNAL_ERROR.value, message=message)


class MCPRequest(BaseModel):
    """MCP request model."""
    jsonrpc: str = Field(default="2.0", description="JSON-RPC version")
    id: str | int | None = Field(..., description="Request ID")
    method: str = Field(..., description="Method name")
    params: dict[str, Any] | None = Field(
        default=None, description="Method parameters"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data: dict[str, Any] = {
            "jsonrpc": self.jsonrpc,
            "id": self.id,
            "method": self.method,
        }
        if self.params:
            data["params"] = self.params
        return data


class MCPResponse(BaseModel):
    """MCP response model."""
    jsonrpc: str = Field(default="2.0", description="JSON-RPC version")
    id: str | int | None = Field(..., description="Request ID")
    result: dict[str, Any] | None = Field(
        default=None, description="Response result"
    )
    error: MCPError | None = Field(
        default=None, description="Response error"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data: dict[str, Any] = {
            "jsonrpc": self.jsonrpc,
            "id": self.id,
        }
        if self.result is not None:
            data["result"] = self.result
        if self.error:
            data["error"] = self.error.model_dump()
        return data

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def success(
        cls, id: str | int | None, result: dict[str, Any]
    ) -> MCPResponse:
        """Create a successful response."""
        return cls(id=id, result=result)

    @classmethod
    def error_response(
        cls, id: str | int | None, error: MCPError
    ) -> MCPResponse:
        """Create an error response."""
        return cls(id=id, error=error)


class MCPToolParameter(BaseModel):
    """MCP tool parameter definition."""
    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type")
    description: str | None = Field(
        default=None, description="Parameter description"
    )
    required: bool = Field(
        default=False, description="Whether parameter is required"
    )
    default: Any | None = Field(default=None, description="Default value")


class MCPTool(BaseModel):
    """MCP tool definition."""
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    parameters: list[MCPToolParameter] = Field(
        default_factory=list,
        description="Tool parameters"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        required_params = [
            param.name for param in self.parameters if param.required
        ]
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": {
                    param.name: {
                        "type": param.type,
                        "description": param.description or "",
                    }
                    for param in self.parameters
                },
                "required": required_params,
            },
        }


class MCPResource(BaseModel):
    """MCP resource definition."""
    uri: str = Field(..., description="Resource URI")
    name: str = Field(..., description="Resource name")
    description: str | None = Field(
        default=None, description="Resource description"
    )
    mime_type: str | None = Field(default=None, description="MIME type")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data: dict[str, Any] = {
            "uri": self.uri,
            "name": self.name,
        }
        if self.description:
            data["description"] = self.description
        if self.mime_type:
            data["mimeType"] = self.mime_type
        return data


class MCPPrompt(BaseModel):
    """MCP prompt definition."""
    name: str = Field(..., description="Prompt name")
    description: str = Field(..., description="Prompt description")
    arguments: list[MCPToolParameter] = Field(
        default_factory=list,
        description="Prompt arguments"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "arguments": [
                {
                    "name": arg.name,
                    "description": arg.description or "",
                    "required": arg.required,
                }
                for arg in self.arguments
            ],
        }


class MCPCapabilities(BaseModel):
    """MCP server capabilities."""
    experimental: dict[str, Any] | None = Field(
        default=None, description="Experimental capabilities"
    )
    logging: dict[str, Any] | None = Field(
        default=None, description="Logging capabilities"
    )
    prompts: dict[str, Any] | None = Field(
        default=None, description="Prompts capabilities"
    )
    resources: dict[str, Any] | None = Field(
        default=None, description="Resources capabilities"
    )
    tools: dict[str, Any] | None = Field(
        default=None, description="Tools capabilities"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data: dict[str, Any] = {}
        if self.experimental:
            data["experimental"] = self.experimental
        if self.logging:
            data["logging"] = self.logging
        if self.prompts:
            data["prompts"] = self.prompts
        if self.resources:
            data["resources"] = self.resources
        if self.tools:
            data["tools"] = self.tools
        return data


class MCPInitialize(BaseModel):
    """MCP initialize request parameters."""
    protocol_version: str = Field(..., description="Protocol version")
    capabilities: MCPCapabilities = Field(
        ..., description="Client capabilities"
    )
    client_info: dict[str, Any] = Field(..., description="Client information")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "protocolVersion": self.protocol_version,
            "capabilities": self.capabilities.to_dict(),
            "clientInfo": self.client_info,
        }


class MCPInitializeResult(BaseModel):
    """MCP initialize response result."""
    protocol_version: str = Field(..., description="Protocol version")
    capabilities: MCPCapabilities = Field(
        ..., description="Server capabilities"
    )
    server_info: dict[str, Any] = Field(..., description="Server information")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "protocolVersion": self.protocol_version,
            "capabilities": self.capabilities.to_dict(),
            "serverInfo": self.server_info,
        }
