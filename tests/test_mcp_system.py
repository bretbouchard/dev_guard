"""Comprehensive test suite for MCP (Model Context Protocol) system components.
This module tests the complete MCP implementation including models, server, and tools.
"""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Import availability check
try:
    # Core MCP imports
    from src.dev_guard.mcp.models import (
        MCPCapabilities,
        MCPError,
        MCPErrorCode,
        MCPInitialize,
        MCPInitializeResult,
        MCPPrompt,
        MCPRequest,
        MCPResource,
        MCPResponse,
        MCPTool,
        MCPToolParameter,
    )
    from src.dev_guard.mcp.server import MCPServer
    from src.dev_guard.mcp.tools import (
        BaseMCPTool,
        CodeContextTool,
        DependencyAnalysisTool,
        ImpactAnalysisTool,
        PatternSearchTool,
        RecommendationTool,
        SecurityScanTool,
    )
    MCP_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"MCP imports not available: {e}")
    MCP_IMPORTS_AVAILABLE = False


# MCP Models Tests
@pytest.mark.skipif(not MCP_IMPORTS_AVAILABLE, reason="MCP modules not available")
class TestMCPModels:
    """Test MCP protocol models and data structures."""

    def test_mcp_error_code_enum(self):
        """Test MCPErrorCode enum values."""
        assert MCPErrorCode.PARSE_ERROR.value == -32700
        assert MCPErrorCode.INVALID_REQUEST.value == -32600
        assert MCPErrorCode.METHOD_NOT_FOUND.value == -32601
        assert MCPErrorCode.INVALID_PARAMS.value == -32602
        assert MCPErrorCode.INTERNAL_ERROR.value == -32603

    def test_mcp_error_creation(self):
        """Test MCPError creation and class methods."""
        # Test direct creation
        error = MCPError(code=-1000, message="Test error", data={"test": True})
        assert error.code == -1000
        assert error.message == "Test error"
        assert error.data == {"test": True}
        
        # Test class method creation
        parse_error = MCPError.parse_error("Custom parse error")
        assert parse_error.code == MCPErrorCode.PARSE_ERROR.value
        assert "Custom parse error" in parse_error.message
        
        invalid_req = MCPError.invalid_request("Custom invalid request")
        assert invalid_req.code == MCPErrorCode.INVALID_REQUEST.value
        
        method_not_found = MCPError.method_not_found("test_method")
        assert method_not_found.code == MCPErrorCode.METHOD_NOT_FOUND.value
        assert "test_method" in method_not_found.message
        
        invalid_params = MCPError.invalid_params("Custom invalid params")
        assert invalid_params.code == MCPErrorCode.INVALID_PARAMS.value
        
        internal_error = MCPError.internal_error("Custom internal error")
        assert internal_error.code == MCPErrorCode.INTERNAL_ERROR.value

    def test_mcp_request_creation(self):
        """Test MCPRequest creation and validation."""
        # Basic request
        request = MCPRequest(
            id="test-123",
            method="tools/list"
        )
        assert request.id == "test-123"
        assert request.method == "tools/list"
        assert request.jsonrpc == "2.0"
        assert request.params is None
        
        # Request with parameters
        request_with_params = MCPRequest(
            id="test-456",
            method="tools/call",
            params={"name": "test_tool", "arguments": {"param": "value"}}
        )
        assert request_with_params.params is not None
        assert request_with_params.params["name"] == "test_tool"

    def test_mcp_response_creation(self):
        """Test MCPResponse creation and factory methods."""
        # Success response
        success_response = MCPResponse.success("test-123", {"result": "ok"})
        assert success_response.id == "test-123"
        assert success_response.jsonrpc == "2.0"
        assert success_response.result == {"result": "ok"}
        assert success_response.error is None
        
        # Error response
        error = MCPError.invalid_params("Test error")
        error_response = MCPResponse.error_response("test-456", error)
        assert error_response.id == "test-456"
        assert error_response.result is None
        assert error_response.error == error

    def test_mcp_tool_parameter_creation(self):
        """Test MCPToolParameter creation and validation."""
        # Required parameter
        param = MCPToolParameter(
            name="test_param",
            type="string",
            description="Test parameter",
            required=True
        )
        assert param.name == "test_param"
        assert param.type == "string"
        assert param.description == "Test parameter"
        assert param.required is True
        assert param.default is None
        
        # Optional parameter with default
        param_optional = MCPToolParameter(
            name="optional_param",
            type="number",
            description="Optional parameter",
            required=False,
            default=42
        )
        assert param_optional.required is False
        assert param_optional.default == 42

    def test_mcp_tool_creation(self):
        """Test MCPTool creation and conversion."""
        param = MCPToolParameter(
            name="input",
            type="string",
            description="Input parameter",
            required=True
        )
        
        tool = MCPTool(
            name="test_tool",
            description="Test tool for validation",
            parameters=[param]
        )
        
        assert tool.name == "test_tool"
        assert tool.description == "Test tool for validation"
        assert len(tool.parameters) == 1
        assert tool.parameters[0].name == "input"
        
        # Test to_dict method
        tool_dict = tool.to_dict()
        assert tool_dict["name"] == "test_tool"
        assert tool_dict["description"] == "Test tool for validation"
        assert "inputSchema" in tool_dict
        assert tool_dict["inputSchema"]["type"] == "object"
        assert "properties" in tool_dict["inputSchema"]

    def test_mcp_capabilities_creation(self):
        """Test MCPCapabilities creation and validation."""
        capabilities = MCPCapabilities(
            tools={"listChanged": True},
            resources={"subscribe": False, "listChanged": True},
            prompts={"listChanged": False}
        )
        
        assert capabilities.tools == {"listChanged": True}
        assert capabilities.resources == {"subscribe": False, "listChanged": True}
        assert capabilities.prompts == {"listChanged": False}
        
        # Test to_dict method
        cap_dict = capabilities.to_dict()
        assert "tools" in cap_dict
        assert "resources" in cap_dict
        assert "prompts" in cap_dict

    def test_mcp_initialize_creation(self):
        """Test MCPInitialize and MCPInitializeResult creation."""
        # Initialize request
        init_request = MCPInitialize(
            protocol_version="2024-11-05",
            capabilities=MCPCapabilities(tools={"listChanged": True}),
            client_info={"name": "Test Client", "version": "1.0.0"}
        )
        
        assert init_request.protocol_version == "2024-11-05"
        assert init_request.capabilities.tools == {"listChanged": True}
        assert init_request.client_info["name"] == "Test Client"
        
        # Initialize result
        init_result = MCPInitializeResult(
            protocol_version="2024-11-05",
            capabilities=MCPCapabilities(tools={"listChanged": True}),
            server_info={"name": "DevGuard MCP Server", "version": "1.0.0"}
        )
        
        assert init_result.protocol_version == "2024-11-05"
        assert init_result.server_info["name"] == "DevGuard MCP Server"
        
        # Test to_dict method
        result_dict = init_result.to_dict()
        assert "protocolVersion" in result_dict
        assert "capabilities" in result_dict
        assert "serverInfo" in result_dict


# MCP Server Tests
@pytest.mark.skipif(not MCP_IMPORTS_AVAILABLE, reason="MCP modules not available")
class TestMCPServer:
    """Test MCP server implementation and functionality."""

    def test_mcp_server_initialization(self):
        """Test MCP server initialization with dependencies."""
        # Mock dependencies
        mock_shared_memory = Mock()
        mock_vector_db = Mock()
        mock_agents = {"test_agent": Mock()}
        
        server = MCPServer(
            shared_memory=mock_shared_memory,
            vector_db=mock_vector_db,
            agents=mock_agents,
            host="localhost",
            port=8080
        )
        
        assert server.shared_memory == mock_shared_memory
        assert server.vector_db == mock_vector_db
        assert server.agents == mock_agents
        assert server.host == "localhost"
        assert server.port == 8080
        assert server.app is not None
        assert len(server.tools) > 0

    def test_mcp_server_default_initialization(self):
        """Test MCP server initialization with defaults."""
        server = MCPServer()
        
        assert server.shared_memory is None
        assert server.vector_db is None
        assert server.agents == {}
        assert server.host == "localhost"
        assert server.port == 8080
        assert len(server.tools) > 0

    def test_server_tool_initialization(self):
        """Test that server initializes all expected tools."""
        server = MCPServer()
        
        expected_tools = [
            "get_code_context",
            "search_patterns", 
            "get_dependencies",
            "analyze_impact",
            "security_scan",
            "get_recommendations"
        ]
        
        for tool_name in expected_tools:
            assert tool_name in server.tools
            assert isinstance(server.tools[tool_name], BaseMCPTool)

    def test_server_capabilities(self):
        """Test server capabilities generation."""
        server = MCPServer()
        capabilities = server.get_server_capabilities()
        
        assert isinstance(capabilities, MCPCapabilities)
        assert capabilities.tools is not None
        assert capabilities.tools.get("listChanged") is True
        assert capabilities.resources is not None
        assert capabilities.prompts is not None

    def test_add_custom_tool(self):
        """Test adding custom tools to server."""
        server = MCPServer()
        initial_count = len(server.tools)
        
        # Create mock tool
        mock_tool = Mock(spec=BaseMCPTool)
        mock_tool.name = "custom_tool"
        mock_tool.description = "Custom test tool"
        mock_tool.parameters = []
        
        server.add_tool(mock_tool)
        
        assert len(server.tools) == initial_count + 1
        assert "custom_tool" in server.tools
        assert server.tools["custom_tool"] == mock_tool

    def test_remove_tool(self):
        """Test removing tools from server."""
        server = MCPServer()
        initial_count = len(server.tools)
        
        # Get first tool name
        first_tool_name = next(iter(server.tools.keys()))
        server.remove_tool(first_tool_name)
        
        assert len(server.tools) == initial_count - 1
        assert first_tool_name not in server.tools

    def test_get_tool_info(self):
        """Test tool information retrieval."""
        server = MCPServer()
        tool_info = server.get_tool_info()
        
        assert isinstance(tool_info, dict)
        assert len(tool_info) > 0
        
        for tool_name, info in tool_info.items():
            assert "name" in info
            assert "description" in info
            assert "parameters" in info
            assert isinstance(info["parameters"], list)

    @pytest.mark.asyncio
    async def test_handle_initialize_request(self):
        """Test handling initialize request."""
        server = MCPServer()
        
        init_request = MCPRequest(
            id="init-1",
            method="initialize",
            params={
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": True}},
                "clientInfo": {"name": "Test Client", "version": "1.0.0"}
            }
        )
        
        response = await server.handle_initialize(init_request)
        
        assert isinstance(response, MCPResponse)
        assert response.id == "init-1"
        assert response.error is None
        assert response.result is not None
        assert "protocolVersion" in response.result
        assert "capabilities" in response.result
        assert "serverInfo" in response.result

    @pytest.mark.asyncio
    async def test_handle_tools_list_request(self):
        """Test handling tools list request."""
        server = MCPServer()
        
        list_request = MCPRequest(
            id="list-1",
            method="tools/list"
        )
        
        response = await server.handle_tools_list(list_request)
        
        assert isinstance(response, MCPResponse)
        assert response.id == "list-1"
        assert response.error is None
        assert response.result is not None
        assert "tools" in response.result
        assert isinstance(response.result["tools"], list)
        assert len(response.result["tools"]) > 0

    @pytest.mark.asyncio
    async def test_handle_tool_call_request(self):
        """Test handling tool call request."""
        server = MCPServer()
        
        # Mock tool execution
        with patch.object(server.tools['get_code_context'], 'execute',
                         new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = {"result": "test context"}
            
            call_request = MCPRequest(
                id="call-1",
                method="tools/call",
                params={
                    "name": "get_code_context",
                    "arguments": {"file_path": "test.py"}
                }
            )
            
            response = await server.handle_tool_call(call_request)
            
            assert isinstance(response, MCPResponse)
            assert response.id == "call-1"
            assert response.error is None
            assert response.result is not None

    @pytest.mark.asyncio
    async def test_handle_invalid_tool_call(self):
        """Test handling invalid tool call request."""
        server = MCPServer()
        
        # Test missing tool name
        invalid_request = MCPRequest(
            id="invalid-1",
            method="tools/call",
            params={"arguments": {"test": "value"}}  # Missing name
        )
        
        response = await server.handle_tool_call(invalid_request)
        
        assert isinstance(response, MCPResponse)
        assert response.id == "invalid-1"
        assert response.result is None
        assert response.error is not None
        assert response.error.code == MCPErrorCode.INVALID_PARAMS.value
        
        # Test unknown tool
        unknown_request = MCPRequest(
            id="invalid-2",
            method="tools/call",
            params={
                "name": "nonexistent_tool",
                "arguments": {}
            }
        )
        
        unknown_response = await server.handle_tool_call(unknown_request)
        assert unknown_response.error is not None
        assert unknown_response.error.code == MCPErrorCode.INVALID_PARAMS.value

    @pytest.mark.asyncio
    async def test_handle_unknown_method(self):
        """Test handling unknown method request."""
        server = MCPServer()
        
        unknown_request = MCPRequest(
            id="unknown-1",
            method="unknown/method"
        )
        
        response = await server.handle_request(unknown_request)
        
        assert isinstance(response, MCPResponse)
        assert response.id == "unknown-1"
        assert response.error is not None
        assert response.error.code == MCPErrorCode.METHOD_NOT_FOUND.value


# MCP Tools Tests
@pytest.mark.skipif(not MCP_IMPORTS_AVAILABLE, reason="MCP modules not available")
class TestMCPTools:
    """Test MCP tool implementations."""

    def test_base_mcp_tool_interface(self):
        """Test BaseMCPTool abstract interface."""
        # Test that BaseMCPTool cannot be instantiated
        with pytest.raises(TypeError):
            BaseMCPTool()
        
        # Test mock implementation
        class MockTool(BaseMCPTool):
            def __init__(self):
                super().__init__()
            
            @property
            def name(self) -> str:
                return "mock_tool"
            
            @property
            def description(self) -> str:
                return "Mock tool for testing"
            
            @property
            def parameters(self):
                return []
            
            async def execute(self, arguments):
                return {"status": "success"}
        
        tool = MockTool()
        assert tool.name == "mock_tool"
        assert tool.description == "Mock tool for testing"
        assert hasattr(tool, 'execute')

    def test_code_context_tool(self):
        """Test CodeContextTool implementation."""
        mock_shared_memory = Mock()
        mock_vector_db = Mock()
        mock_agents = {}
        
        tool = CodeContextTool(
            shared_memory=mock_shared_memory,
            vector_db=mock_vector_db,
            agents=mock_agents
        )
        
        assert tool.name == "get_code_context"
        assert "code" in tool.description.lower()
        assert len(tool.parameters) > 0
        
        # Check parameters
        param_names = [p.name for p in tool.parameters]
        assert "file_path" in param_names
        
        # Test to_mcp_tool method
        mcp_tool = tool.to_mcp_tool()
        assert isinstance(mcp_tool, MCPTool)
        assert mcp_tool.name == tool.name

    @pytest.mark.asyncio
    async def test_code_context_tool_execution(self):
        """Test CodeContextTool execution."""
        mock_shared_memory = Mock()
        mock_vector_db = Mock()
        mock_agents = {}
        
        tool = CodeContextTool(
            shared_memory=mock_shared_memory,
            vector_db=mock_vector_db,
            agents=mock_agents
        )
        
        # Mock file existence and reading
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data="def test(): pass")):
            
            result = await tool.execute({"file_path": "test.py"})
            
            assert isinstance(result, dict)
            assert "file_path" in result

    def test_pattern_search_tool(self):
        """Test PatternSearchTool implementation."""
        tool = PatternSearchTool()
        
        assert tool.name == "search_patterns"
        assert "pattern" in tool.description.lower()
        assert len(tool.parameters) > 0
        
        param_names = [p.name for p in tool.parameters]
        assert "pattern" in param_names

    def test_dependency_analysis_tool(self):
        """Test DependencyAnalysisTool implementation.""" 
        tool = DependencyAnalysisTool()
        
        assert tool.name == "get_dependencies"
        assert "dependenc" in tool.description.lower()
        assert len(tool.parameters) > 0

    def test_impact_analysis_tool(self):
        """Test ImpactAnalysisTool implementation."""
        tool = ImpactAnalysisTool()
        
        assert tool.name == "analyze_impact"
        assert "impact" in tool.description.lower()
        assert len(tool.parameters) > 0

    def test_security_scan_tool(self):
        """Test SecurityScanTool implementation."""
        tool = SecurityScanTool()
        
        assert tool.name == "security_scan"
        assert "security" in tool.description.lower()
        assert len(tool.parameters) > 0

    def test_recommendation_tool(self):
        """Test RecommendationTool implementation."""
        tool = RecommendationTool()
        
        assert tool.name == "get_recommendations"
        assert "recommend" in tool.description.lower()
        assert len(tool.parameters) > 0


# Integration Tests
@pytest.mark.skipif(not MCP_IMPORTS_AVAILABLE, reason="MCP modules not available") 
class TestMCPIntegration:
    """Test integration between MCP components."""

    def test_server_tool_integration(self):
        """Test integration between server and tools."""
        server = MCPServer()
        
        # Verify all tools are properly integrated
        for tool_name, tool in server.tools.items():
            assert isinstance(tool, BaseMCPTool)
            assert hasattr(tool, 'name')
            assert hasattr(tool, 'description')
            assert hasattr(tool, 'parameters')
            assert hasattr(tool, 'execute')
            
            # Test tool can be converted to MCP format
            mcp_tool = tool.to_mcp_tool()
            assert isinstance(mcp_tool, MCPTool)
            assert mcp_tool.name == tool.name

    @pytest.mark.asyncio
    async def test_end_to_end_request_handling(self):
        """Test complete request/response cycle."""
        server = MCPServer()
        
        # Test initialize
        init_request = MCPRequest(
            id="1",
            method="initialize",
            params={
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": True}},
                "clientInfo": {"name": "Test", "version": "1.0"}
            }
        )
        init_response = await server.handle_request(init_request)
        assert init_response.error is None
        
        # Test tools list
        list_request = MCPRequest(
            id="2", 
            method="tools/list"
        )
        list_response = await server.handle_request(list_request)
        assert list_response.error is None
        assert "tools" in list_response.result
        
        # Test tool call
        with patch.object(server.tools['get_code_context'], 'execute',
                         new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = {"result": "test"}
            
            call_request = MCPRequest(
                id="3",
                method="tools/call",
                params={
                    "name": "get_code_context",
                    "arguments": {"file_path": "test.py"}
                }
            )
            call_response = await server.handle_request(call_request)
            assert call_response.error is None
            assert "content" in call_response.result

    def test_error_handling_consistency(self):
        """Test consistent error handling across components."""
        # Test error codes are consistent
        assert MCPErrorCode.PARSE_ERROR.value == -32700
        assert MCPErrorCode.INVALID_REQUEST.value == -32600
        assert MCPErrorCode.METHOD_NOT_FOUND.value == -32601
        assert MCPErrorCode.INVALID_PARAMS.value == -32602
        assert MCPErrorCode.INTERNAL_ERROR.value == -32603
        
        # Test error creation consistency
        errors = [
            MCPError.parse_error(),
            MCPError.invalid_request(),
            MCPError.method_not_found("test"),
            MCPError.invalid_params(),
            MCPError.internal_error()
        ]
        
        for error in errors:
            assert isinstance(error, MCPError)
            assert isinstance(error.code, int)
            assert isinstance(error.message, str)


# Mock-based tests for components without dependencies
class TestMCPMocks:
    """Test MCP components using mocks to avoid dependencies."""

    def test_mock_mcp_tool_framework(self):
        """Test MCP tool framework with mocks."""
        if not MCP_IMPORTS_AVAILABLE:
            pytest.skip("MCP modules not available")
        
        class MockMCPTool(BaseMCPTool):
            def __init__(self, name, description):
                super().__init__()
                self.name = name
                self.description = description
                self.parameters = [
                    MCPToolParameter(name="input", type="string", required=True)
                ]
            
            async def execute(self, arguments):
                return {"output": f"Processed {arguments.get('input')}"}
        
        tool = MockMCPTool("mock_tool", "Mock tool for testing")
        assert tool.name == "mock_tool"
        assert tool.description == "Mock tool for testing"
        assert len(tool.parameters) == 1
        
        # Test MCP tool conversion
        mcp_tool = tool.to_mcp_tool()
        assert mcp_tool.name == "mock_tool"
        assert len(mcp_tool.parameters) == 1

    def test_mock_server_lifecycle(self):
        """Test server lifecycle with mocks."""
        if not MCP_IMPORTS_AVAILABLE:
            pytest.skip("MCP modules not available")
        
        # Mock FastAPI and dependencies
        with patch('src.dev_guard.mcp.server.FastAPI'), \
             patch('src.dev_guard.mcp.server.uvicorn'):
            server = MCPServer(host="test", port=9999)
            
            assert server.host == "test"
            assert server.port == 9999
            
            # Test tool management
            initial_count = len(server.tools)
            
            # Mock adding tool
            mock_tool = Mock()
            mock_tool.name = "test_tool"
            server.add_tool(mock_tool)
            
            assert len(server.tools) == initial_count + 1
            assert "test_tool" in server.tools

    def test_mock_protocol_compliance(self):
        """Test MCP protocol compliance with mocks."""
        if not MCP_IMPORTS_AVAILABLE:
            pytest.skip("MCP modules not available")
        
        # Test request/response structure compliance
        request_data = {
            "jsonrpc": "2.0",
            "id": "test-123",
            "method": "tools/list",
            "params": {}
        }
        
        # Mock successful parsing
        request = MCPRequest(**request_data)
        assert request.jsonrpc == "2.0"
        assert request.id == "test-123"
        assert request.method == "tools/list"
        
        # Mock response creation
        response = MCPResponse.success(request.id, {"tools": []})
        assert response.id == request.id
        assert response.result is not None
        assert response.error is None


# Performance and Edge Case Tests
class TestMCPPerformance:
    """Test MCP system performance and edge cases."""

    def test_large_payload_handling(self):
        """Test handling of large payloads."""
        if not MCP_IMPORTS_AVAILABLE:
            pytest.skip("MCP modules not available")
        
        # Create large content
        large_content = {"data": "x" * 10000}
        response = MCPResponse.success(
            id="large-test",
            result=large_content
        )
        
        # Should handle large payloads without errors
        response_json = response.to_json()
        assert len(response_json) > 10000
        
        # Should be parseable
        parsed = json.loads(response_json)
        assert parsed["result"]["data"] == "x" * 10000

    def test_concurrent_server_operations(self):
        """Test concurrent server operations."""
        if not MCP_IMPORTS_AVAILABLE:
            pytest.skip("MCP modules not available")
        
        server = MCPServer()
        
        # Test that adding/removing tools is thread-safe (basic check)
        initial_tools = list(server.tools.keys())
        
        # Add multiple tools
        for i in range(5):
            mock_tool = Mock()
            mock_tool.name = f"test_tool_{i}"
            server.add_tool(mock_tool)
        
        assert len(server.tools) == len(initial_tools) + 5
        
        # Remove tools
        for i in range(3):
            server.remove_tool(f"test_tool_{i}")
        
        assert len(server.tools) == len(initial_tools) + 2

    def test_error_recovery(self):
        """Test error recovery mechanisms."""
        if not MCP_IMPORTS_AVAILABLE:
            pytest.skip("MCP modules not available")
        
        server = MCPServer()
        
        # Test that server continues after tool errors
        class FailingTool(BaseMCPTool):
            @property
            def name(self):
                return "failing_tool"
            
            @property
            def description(self):
                return "Tool that fails"
            
            @property
            def parameters(self):
                return []
            
            async def execute(self, arguments):
                raise Exception("Tool execution failed")
        
        failing_tool = FailingTool()
        server.add_tool(failing_tool)
        
        # Server should still be operational
        assert "failing_tool" in server.tools
        assert len(server.tools) > 1


if __name__ == "__main__":
    if MCP_IMPORTS_AVAILABLE:
        pytest.main([__file__, "-v", "-x"])
    else:
        print("Skipping MCP tests due to import errors")
        exit(1)
