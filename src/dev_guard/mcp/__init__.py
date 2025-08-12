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
# Re-export base tool class for tests that import it directly
from .tools import BaseMCPTool  # noqa: F401

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
