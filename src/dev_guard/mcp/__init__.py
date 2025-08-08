"""MCP (Model Context Protocol) Server Implementation for DevGuard.

This module provides the MCP server interface for exposing DevGuard's
agent capabilities to IDEs and external tools.
"""

from .server import MCPServer
from .models import (
    MCPRequest,
    MCPResponse,
    MCPError,
    MCPTool,
    MCPResource,
    MCPPrompt,
)
from .tools import (
    CodeContextTool,
    PatternSearchTool,
    DependencyAnalysisTool,
    ImpactAnalysisTool,
    SecurityScanTool,
    RecommendationTool,
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
