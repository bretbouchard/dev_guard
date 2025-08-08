"""MCP (Model Context Protocol) Server Implementation for DevGuard.

This module provides the MCP server interface for exposing DevGuard's
agent capabilities to IDEs and external tools.
"""

from .models import (
    MCPError,
    MCPPrompt,
    MCPRequest,
    MCPResource,
    MCPResponse,
    MCPTool,
)
from .server import MCPServer
from .tools import (
    CodeContextTool,
    DependencyAnalysisTool,
    ImpactAnalysisTool,
    PatternSearchTool,
    RecommendationTool,
    SecurityScanTool,
)

__all__ = [
    "MCPServer",
    "MCPRequest",
    "MCPResponse",
    "MCPError",
    "MCPTool",
    "MCPResource",
    "MCPPrompt",
    "CodeContextTool",
    "PatternSearchTool",
    "DependencyAnalysisTool",
    "ImpactAnalysisTool",
    "SecurityScanTool",
    "RecommendationTool",
]
