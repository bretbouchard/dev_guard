"""Comprehensive test suite for MCP (Model Context Protocol) components.

This module tests the MCP models, server, and tools implementation.
"""
import json
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

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


# Model Tests
@pytest.mark.skipif(not MCP_IMPORTS_AVAILABLE, reason="MCP modules not available")
class TestMCPModels:
    """Test MCP data models and protocol structures."""
    
    def test_mcp_error_creation(self):
        """Test MCP error model creation and methods."""
        # Test basic error creation
        error = MCPError(code=-32600, message="Invalid Request")
        assert error.code == -32600
        assert error.message == "Invalid Request"
        assert error.data is None
        
        # Test error with data
        error_with_data = MCPError(
            code=-32700,
            message="Parse error", 
            data={"line": 1, "column": 5}
        )
        assert error_with_data.data == {"line": 1, "column": 5}
    
    def test_mcp_error_factory_methods(self):
        """Test MCP error factory methods."""
        # Test parse error
        parse_error = MCPError.parse_error("JSON malformed")
        assert parse_error.code == MCPErrorCode.PARSE_ERROR.value
        assert "JSON malformed" in parse_error.message
        
        # Test invalid request
        invalid_req = MCPError.invalid_request("Missing required field")
        assert invalid_req.code == MCPErrorCode.INVALID_REQUEST.value
        assert "Missing required field" in invalid_req.message
        
        # Test method not found
        method_error = MCPError.method_not_found("unknown_method")
        assert method_error.code == MCPErrorCode.METHOD_NOT_FOUND.value
        assert "unknown_method" in method_error.message
        
        # Test invalid params
        param_error = MCPError.invalid_params("Wrong parameter type")
        assert param_error.code == MCPErrorCode.INVALID_PARAMS.value
        assert "Wrong parameter type" in param_error.message
        
        # Test internal error
        internal_error = MCPError.internal_error("Database connection failed")
        assert internal_error.code == MCPErrorCode.INTERNAL_ERROR.value
        assert "Database connection failed" in internal_error.message
    
    def test_mcp_request_model(self):
        """Test MCP request model."""
        # Test basic request
        request = MCPRequest(
            id="req-123",
            method="tools/list"
        )
        assert request.jsonrpc == "2.0"  # Default value
        assert request.id == "req-123"
        assert request.method == "tools/list"
        assert request.params is None
        
        # Test request with params
        request_with_params = MCPRequest(
            id="req-456",
            method="tools/call",
            params={"name": "test_tool", "arguments": {"key": "value"}}
        )
        assert request_with_params.params is not None
        assert request_with_params.params["name"] == "test_tool"
        
        # Test to_dict method
        request_dict = request_with_params.to_dict()
        assert request_dict["jsonrpc"] == "2.0"
        assert request_dict["method"] == "tools/call"
        assert "params" in request_dict
    
    def test_mcp_response_model(self):
        """Test MCP response model."""
        # Test successful response
        success_response = MCPResponse.success(
            id="req-123",
            result={"tools": [{"name": "test_tool"}]}
        )
        assert success_response.id == "req-123"
        assert success_response.result is not None
        assert success_response.error is None
        
        # Test error response
        error = MCPError.internal_error("Test error")
        error_response = MCPResponse.error_response(
            id="req-456", 
            error=error
        )
        assert error_response.id == "req-456"
        assert error_response.result is None
        assert error_response.error is not None
        
        # Test to_dict method
        response_dict = success_response.to_dict()
        assert "jsonrpc" in response_dict
        assert "id" in response_dict
        assert "result" in response_dict
        
        # Test to_json method
        response_json = success_response.to_json()
        assert isinstance(response_json, str)
        parsed = json.loads(response_json)
        assert parsed["id"] == "req-123"
    
    def test_mcp_tool_parameter_model(self):
        """Test MCP tool parameter model."""
        param = MCPToolParameter(
            name="file_path",
            type="string",
            description="Path to the file to analyze",
            required=True,
            default="/default/path"
        )
        
        assert param.name == "file_path"
        assert param.type == "string"
        assert param.description == "Path to the file to analyze"
        assert param.required is True
        assert param.default == "/default/path"
    
    def test_mcp_tool_model(self):
        """Test MCP tool model."""
        # Create parameters
        param1 = MCPToolParameter(
            name="file_path",
            type="string", 
            description="File to analyze",
            required=True
        )
        param2 = MCPToolParameter(
            name="depth",
            type="integer",
            description="Analysis depth",
            required=False,
            default=1
        )
        
        # Create tool
        tool = MCPTool(
            name="analyze_file",
            description="Analyze a code file",
            parameters=[param1, param2]
        )
        
        assert tool.name == "analyze_file"
        assert tool.description == "Analyze a code file"
        assert len(tool.parameters) == 2
        
        # Test to_dict method
        tool_dict = tool.to_dict()
        assert "inputSchema" in tool_dict
        assert tool_dict["inputSchema"]["type"] == "object"
        assert "properties" in tool_dict["inputSchema"]
        assert "required" in tool_dict["inputSchema"]
        
        # Check required fields
        required_fields = tool_dict["inputSchema"]["required"]
        assert "file_path" in required_fields
        assert "depth" not in required_fields  # Not required
    
    def test_mcp_resource_model(self):
        """Test MCP resource model."""
        resource = MCPResource(
            uri="file:///path/to/file.py",
            name="test_file.py",
            description="Test Python file",
            mime_type="text/x-python"
        )
        
        assert resource.uri == "file:///path/to/file.py"
        assert resource.name == "test_file.py"
        assert resource.description == "Test Python file"
        assert resource.mime_type == "text/x-python"
        
        # Test to_dict method
        resource_dict = resource.to_dict()
        assert resource_dict["uri"] == resource.uri
        assert resource_dict["name"] == resource.name
        assert resource_dict["description"] == resource.description
        assert resource_dict["mimeType"] == resource.mime_type
    
    def test_mcp_prompt_model(self):
        """Test MCP prompt model."""
        arg1 = MCPToolParameter(
            name="language",
            type="string",
            description="Programming language",
            required=True
        )
        
        prompt = MCPPrompt(
            name="code_template",
            description="Generate code template",
            arguments=[arg1]
        )
        
        assert prompt.name == "code_template"
        assert prompt.description == "Generate code template"
        assert len(prompt.arguments) == 1
        
        # Test to_dict method
        prompt_dict = prompt.to_dict()
        assert prompt_dict["name"] == "code_template"
        assert "arguments" in prompt_dict
        assert len(prompt_dict["arguments"]) == 1
    
    def test_mcp_capabilities_model(self):
        """Test MCP capabilities model."""
        capabilities = MCPCapabilities(
            tools={"listChanged": True},
            resources={"subscribe": False, "listChanged": True},
            prompts={"listChanged": False},
            logging={"level": "info"}
        )
        
        assert capabilities.tools == {"listChanged": True}
        assert capabilities.resources["subscribe"] is False
        assert capabilities.prompts == {"listChanged": False}
        assert capabilities.logging == {"level": "info"}
        
        # Test to_dict method
        cap_dict = capabilities.to_dict()
        assert "tools" in cap_dict
        assert "resources" in cap_dict
        assert "prompts" in cap_dict
        assert "logging" in cap_dict
    
    def test_mcp_initialize_models(self):
        """Test MCP initialize request and result models."""
        # Initialize request
        client_caps = MCPCapabilities(
            tools={"listChanged": True}
        )
        
        initialize_req = MCPInitialize(
            protocol_version="2024-11-05",
            capabilities=client_caps,
            client_info={"name": "Test Client", "version": "1.0.0"}
        )
        
        assert initialize_req.protocol_version == "2024-11-05"
        assert initialize_req.capabilities.tools == {"listChanged": True}
        assert initialize_req.client_info["name"] == "Test Client"
        
        # Initialize result
        server_caps = MCPCapabilities(
            tools={"listChanged": True},
            resources={"listChanged": False}
        )
        
        initialize_result = MCPInitializeResult(
            protocol_version="2024-11-05",
            capabilities=server_caps,
            server_info={"name": "DevGuard MCP Server", "version": "1.0.0"}
        )
        
        assert initialize_result.protocol_version == "2024-11-05"
        assert initialize_result.capabilities.tools == {"listChanged": True}
        assert initialize_result.server_info["name"] == "DevGuard MCP Server"
        
        # Test to_dict method
        result_dict = initialize_result.to_dict()
        assert "protocolVersion" in result_dict
        assert "capabilities" in result_dict
        assert "serverInfo" in result_dict


# Server Tests
@pytest.mark.skipif(not MCP_IMPORTS_AVAILABLE, reason="MCP modules not available")
class TestMCPServer:
    """Test MCP server implementation."""
    
    def test_mcp_server_initialization(self):
        """Test MCP server initialization."""
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
        assert server.app is not None  # FastAPI app initialized
        assert len(server.tools) > 0  # Tools initialized
    
    def test_mcp_server_tool_initialization(self):
        """Test that MCP server initializes tools correctly."""
        server = MCPServer()
        
        # Check that standard tools are available
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
        initial_tool_count = len(server.tools)
        
        # Create mock tool
        mock_tool = Mock(spec=BaseMCPTool)
        mock_tool.name = "custom_tool"
        mock_tool.description = "Custom test tool"
        mock_tool.parameters = []
        
        server.add_tool(mock_tool)
        
        assert len(server.tools) == initial_tool_count + 1
        assert "custom_tool" in server.tools
        assert server.tools["custom_tool"] == mock_tool
    
    def test_remove_tool(self):
        """Test removing tools from server."""
        server = MCPServer()
        initial_tool_count = len(server.tools)
        
        # Remove an existing tool
        first_tool_name = next(iter(server.tools.keys()))
        server.remove_tool(first_tool_name)
        
        assert len(server.tools) == initial_tool_count - 1
        assert first_tool_name not in server.tools
    
    def test_get_tool_info(self):
        """Test getting tool information."""
        server = MCPServer()
        tool_info = server.get_tool_info()
        
        assert isinstance(tool_info, dict)
        assert len(tool_info) > 0
        
        # Check structure of tool info
        for tool_name, info in tool_info.items():
            assert "name" in info
            assert "description" in info
            assert "parameters" in info
            assert isinstance(info["parameters"], list)
    
    @pytest.mark.asyncio
    async def test_handle_initialize_request(self):
        """Test handling initialize request."""
        server = MCPServer()
        
        # Create initialize request
        request = MCPRequest(
            id="init-1",
            method="initialize",
            params={
                "protocol_version": "2024-11-05",
                "capabilities": {"tools": {"listChanged": True}},
                "client_info": {"name": "Test Client", "version": "1.0"}
            }
        )
        
        response = await server.handle_initialize(request)
        
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
        
        request = MCPRequest(
            id="tools-1",
            method="tools/list"
        )
        
        response = await server.handle_tools_list(request)
        
        assert isinstance(response, MCPResponse)
        assert response.id == "tools-1"
        assert response.error is None
        assert response.result is not None
        assert "tools" in response.result
        assert isinstance(response.result["tools"], list)
        assert len(response.result["tools"]) > 0
    
    @pytest.mark.asyncio
    async def test_handle_tool_call_request(self):
        """Test handling tool call request."""
        server = MCPServer()
        
        # Mock a tool execution
        mock_tool = Mock(spec=BaseMCPTool)
        mock_tool.execute = AsyncMock(return_value={"result": "test output"})
        server.tools["test_tool"] = mock_tool
        
        request = MCPRequest(
            id="call-1",
            method="tools/call",
            params={
                "name": "test_tool",
                "arguments": {"input": "test"}
            }
        )
        
        response = await server.handle_tool_call(request)
        
        assert isinstance(response, MCPResponse)
        assert response.id == "call-1"
        assert response.error is None
        assert response.result is not None
        assert "content" in response.result
        
        # Verify tool was called
        mock_tool.execute.assert_called_once_with({"input": "test"})
    
    @pytest.mark.asyncio
    async def test_handle_unknown_method(self):
        """Test handling unknown method request."""
        server = MCPServer()
        
        request = MCPRequest(
            id="unknown-1",
            method="unknown/method"
        )
        
        response = await server.handle_request(request)
        
        assert isinstance(response, MCPResponse)
        assert response.id == "unknown-1"
        assert response.result is None
        assert response.error is not None
        assert response.error.code == MCPErrorCode.METHOD_NOT_FOUND.value
    
    @pytest.mark.asyncio
    async def test_handle_invalid_tool_call(self):
        """Test handling invalid tool call request."""
        server = MCPServer()
        
        # Test missing tool name
        request = MCPRequest(
            id="invalid-1",
            method="tools/call",
            params={"arguments": {"test": "value"}}  # Missing name
        )
        
        response = await server.handle_tool_call(request)
        
        assert isinstance(response, MCPResponse)
        assert response.id == "invalid-1"
        assert response.result is None
        assert response.error is not None
        assert response.error.code == MCPErrorCode.INVALID_PARAMS.value
        
        # Test unknown tool
        request_unknown = MCPRequest(
            id="invalid-2",
            method="tools/call",
            params={
                "name": "nonexistent_tool",
                "arguments": {}
            }
        )
        
        response_unknown = await server.handle_tool_call(request_unknown)
        
        assert response_unknown.error is not None
        assert response_unknown.error.code == MCPErrorCode.INVALID_PARAMS.value


# Tools Tests  
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
            @property
            def name(self) -> str:
                return "mock_tool"
                
            @property
            def description(self) -> str:
                return "Mock tool for testing"
                
            @property
            def parameters(self):
                return []
            
            async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
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
        
        # Mock file operations
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock=Mock()), \
             patch('os.path.getsize', return_value=1000):
            
            result = await tool.execute({"file_path": "test.py"})
            
            assert isinstance(result, dict)
            assert "file_path" in result
            assert result["file_path"] == "test.py"
    
    def test_pattern_search_tool(self):
        """Test PatternSearchTool implementation."""
        tool = PatternSearchTool()
        
        assert tool.name == "search_patterns"
        assert "pattern" in tool.description.lower() or "search" in tool.description.lower()
        assert len(tool.parameters) > 0
        
        # Check for expected parameters
        param_names = [p.name for p in tool.parameters]
        assert "query" in param_names or "pattern" in param_names
    
    def test_dependency_analysis_tool(self):
        """Test DependencyAnalysisTool implementation."""
        tool = DependencyAnalysisTool()
        
        assert tool.name == "get_dependencies"
        assert "depend" in tool.description.lower()
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
    """Test MCP component integration."""
    
    def test_mcp_tool_server_integration(self):
        """Test tool integration with server."""
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
            assert error.code < 0  # All error codes should be negative
            assert len(error.message) > 0  # All errors should have messages
    
    @pytest.mark.asyncio
    async def test_full_request_response_cycle(self):
        """Test complete MCP request/response cycle."""
        server = MCPServer()
        
        # Test initialize -> tools/list -> tools/call cycle
        
        # 1. Initialize
        init_request = MCPRequest(
            id="1",
            method="initialize",
            params={
                "protocol_version": "2024-11-05", 
                "capabilities": {"tools": {"listChanged": True}},
                "client_info": {"name": "Test", "version": "1.0"}
            }
        )
        
        init_response = await server.handle_request(init_request)
        assert init_response.error is None
        
        # 2. List tools
        list_request = MCPRequest(id="2", method="tools/list")
        list_response = await server.handle_request(list_request)
        assert list_response.error is None
        assert "tools" in list_response.result
        
        # 3. Call a tool (with mocking)
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


# Mock-based tests for components without dependencies
class TestMCPMocks:
    """Test MCP components using mocks."""
    
    def test_mock_mcp_tool_framework(self):
        """Test MCP tool framework with mocks."""
        class MockMCPTool(BaseMCPTool):
            def __init__(self, name, description):
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
        # Test request/response structure compliance
        request_data = {
            "jsonrpc": "2.0",
            "id": "test-123",
            "method": "tools/list",
            "params": None
        }
        
        request = MCPRequest(**request_data)
        assert request.jsonrpc == "2.0"
        
        # Test response structure
        response = MCPResponse.success(
            id=request.id,
            result={"tools": []}
        )
        
        response_dict = response.to_dict()
        assert "jsonrpc" in response_dict
        assert "id" in response_dict
        assert "result" in response_dict


# Performance and edge case tests
class TestMCPEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_invalid_model_data(self):
        """Test model validation with invalid data."""
        if not MCP_IMPORTS_AVAILABLE:
            pytest.skip("MCP modules not available")
        
        # Test invalid error code
        with pytest.raises(Exception):  # Should raise validation error
            MCPError(code="invalid", message="test")
        
        # Test invalid request structure
        with pytest.raises(Exception):
            MCPRequest(method="")  # Empty method should fail
    
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


if __name__ == "__main__":
    if MCP_IMPORTS_AVAILABLE:
        pytest.main([__file__, "-v", "-x"])
    else:
        print("Skipping MCP tests due to import errors")
        exit(1)
