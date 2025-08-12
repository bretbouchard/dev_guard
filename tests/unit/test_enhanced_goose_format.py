"""Tests for enhanced Goose patch format implementation (Task 10.4)."""

from unittest.mock import MagicMock, patch

import pytest

from dev_guard.agents.code_agent import CodeAgent
from dev_guard.memory.shared_memory import SharedMemory


@pytest.fixture
def mock_shared_memory():
    """Mock shared memory for testing."""
    memory = MagicMock(spec=SharedMemory)
    memory.add_memory = MagicMock()
    return memory


@pytest.fixture
def code_agent(mock_shared_memory):
    """Create a CodeAgent instance for testing."""
    # Minimal config mock with required attributes
    with patch('dev_guard.agents.code_agent.BaseAgent.__init__', return_value=None):
        agent = CodeAgent(
            agent_id="code_agent_test",
            config=MagicMock(),
            shared_memory=mock_shared_memory,
            vector_db=MagicMock(),
            working_directory="/test"
        )
    agent.agent_id = "code_agent_test"
    agent.shared_memory = mock_shared_memory
    agent.logger = MagicMock()
    return agent


@pytest.fixture
def sample_task():
    """Sample task for testing."""
    return {
        "type": "refactor",
        "file_path": "/test/sample.py",
        "prompt": "Refactor this code to improve readability",
        "description": "Test refactoring task"
    }


@pytest.fixture
def sample_goose_result():
    """Sample Goose execution result with enhanced tool call format."""
    return {
        "command": "goose session start",
        "session_id": "test-session-123",
        "output": "Session started successfully",
        "error": None,
        "return_code": 0,
        "success": True,
        "quality_checks_applied": ["syntax_check", "lint_check"],
        "tool_call": {
            "type": "goose_cli",
            "function": "session",
            "arguments": {"command": "start"},
            "timestamp": "2024-01-01T10:00:00.000Z",
            "duration_seconds": 2.5,
            "metadata": {
                "working_directory": "/test",
                "command_line": ["goose", "session", "start"],
                "exit_code": 0,
                "output_truncated": False,
                "error_output": None
            }
        }
    }


class TestEnhancedGooseFormat:
    """Test enhanced Goose patch format alignment."""
    
    @pytest.mark.asyncio
    async def test_log_goose_result_enhanced_format(self, code_agent, sample_task, sample_goose_result):
        """Test that _log_goose_result creates enhanced format memory entries."""
        await code_agent._log_goose_result(sample_task, sample_goose_result)
        
        # Verify add_memory was called
        code_agent.shared_memory.add_memory.assert_called_once()
        
        # Get the memory entry that was added
        memory_entry = code_agent.shared_memory.add_memory.call_args[0][0]
        
        # Verify basic memory entry structure
        assert memory_entry.agent_id == code_agent.agent_id
        assert memory_entry.type == "result"
        assert memory_entry.parent_id is None
        assert memory_entry.ast_summary is None
        assert memory_entry.goose_strategy == "refactor"
        assert memory_entry.file_path == "/test/sample.py"
        
        # Verify tags
        expected_tags = {"goose", "code_generation", "refactor", "enhanced_format"}
        assert memory_entry.tags == expected_tags
        
        # Verify content structure
        content = memory_entry.content
        assert "task" in content
        assert "result" in content
        assert "success" in content
        assert content["success"] is True
        
        # Verify enhanced goose patch structure
        goose_patch = memory_entry.goose_patch
        assert goose_patch is not None
        
        # Core execution data
        assert goose_patch["command"] == "goose session start"
        assert goose_patch["session_id"] == "test-session-123"
        assert goose_patch["output"] == "Session started successfully"
        assert goose_patch["error"] is None
        assert goose_patch["return_code"] == 0
        
        # Enhanced tool call metadata
        tool_call = goose_patch["tool_call"]
        assert tool_call["type"] == "goose_cli"
        assert tool_call["function"] == "session"
        assert tool_call["arguments"] == {"command": "start"}
        assert tool_call["timestamp"] == "2024-01-01T10:00:00.000Z"
        assert tool_call["duration_seconds"] == 2.5
        
        # Tool call metadata
        metadata = tool_call["metadata"]
        assert metadata["working_directory"] == "/test"
        assert metadata["command_line"] == ["goose", "session", "start"]
        assert metadata["exit_code"] == 0
        
        # DevGuard-specific metadata
        devguard_metadata = goose_patch["devguard_metadata"]
        assert devguard_metadata["task_type"] == "refactor"
        assert devguard_metadata["agent_id"] == code_agent.agent_id
        assert devguard_metadata["working_directory"] == "/test"
        assert devguard_metadata["file_path"] == "/test/sample.py"
        
        # Execution context
        execution_context = devguard_metadata["execution_context"]
        assert execution_context["prompt_used"] == "Refactor this code to improve readability"
        assert execution_context["task_description"] == "Test refactoring task"
        assert execution_context["quality_checks_applied"] == ["syntax_check", "lint_check"]
        
        # Markdown export compatibility
        markdown_export = goose_patch["markdown_export"]
        assert markdown_export["format_version"] == "1.0"
        assert markdown_export["exportable"] is True
        assert "devguard" in markdown_export["session_name"]
        assert markdown_export["summary"].startswith("DevGuard refactor operation")
    
    @pytest.mark.asyncio 
    async def test_log_goose_result_minimal_data(self, code_agent):
        """Test _log_goose_result with minimal input data."""
        minimal_task = {"type": "generate"}
        minimal_result = {
            "command": "goose help",
            "output": "Help text",
            "success": False
        }
        
        await code_agent._log_goose_result(minimal_task, minimal_result)
        
        # Verify memory entry was created
        code_agent.shared_memory.add_memory.assert_called_once()
        memory_entry = code_agent.shared_memory.add_memory.call_args[0][0]
        
        # Verify goose patch handles missing data gracefully
        goose_patch = memory_entry.goose_patch
        assert goose_patch["command"] == "goose help"
        assert goose_patch["session_id"] is None
        assert goose_patch["output"] == "Help text"
        assert goose_patch["return_code"] == -1  # Default value
        
        # Verify tool call has defaults
        tool_call = goose_patch["tool_call"]
        assert tool_call["type"] == "goose_cli"
        assert tool_call["function"] == "session"
        assert tool_call["arguments"] == {}
        assert tool_call["duration_seconds"] == 0
        
        # Verify DevGuard metadata handles missing data
        devguard_metadata = goose_patch["devguard_metadata"]
        assert devguard_metadata["task_type"] == "generate"
        assert devguard_metadata["file_path"] is None
        assert devguard_metadata["working_directory"] is None
    
    @pytest.mark.asyncio
    async def test_log_goose_result_error_handling(self, code_agent, sample_task):
        """Test error handling in _log_goose_result."""
        # Mock add_memory to raise exception
        code_agent.shared_memory.add_memory.side_effect = Exception("Test error")
        
        invalid_result = {"command": "test"}
        
        # Should not raise exception
        await code_agent._log_goose_result(sample_task, invalid_result)
        
        # Verify error was logged
        code_agent.logger.error.assert_called_once()
        assert "Error logging enhanced Goose result" in str(code_agent.logger.error.call_args)
    
    @pytest.mark.asyncio
    async def test_run_goose_command_enhanced_metadata(self, code_agent):
        """Test that _run_goose_command produces enhanced metadata format."""
        with patch('subprocess.run') as mock_run:
            # Mock successful subprocess execution
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_process.stdout = "Test output"
            mock_process.stderr = ""
            mock_run.return_value = mock_process
            
            with patch('os.getcwd', return_value="/test/working/dir"):
                result = await code_agent._run_goose_command("test command", "session-123")
                
                # Verify enhanced tool call format is included
                assert "tool_call" in result
                tool_call = result["tool_call"]
                
                assert tool_call["type"] == "goose_cli"
                assert tool_call["function"] == "session"
                assert "arguments" in tool_call
                assert "timestamp" in tool_call
                assert "duration_seconds" in tool_call
                
                # Verify metadata structure
                metadata = tool_call["metadata"]
                assert metadata["working_directory"] == "/test/working/dir"
                assert metadata["command_line"] == ["goose", "test", "command"]
                assert metadata["exit_code"] == 0
                assert metadata["output_truncated"] is False
    
    def test_format_compatibility_with_goose_export(self, sample_goose_result):
        """Test that our enhanced format is compatible with expected Goose export structure."""
        tool_call = sample_goose_result["tool_call"]
        
        # Verify required Goose export fields
        required_fields = ["type", "function", "arguments", "timestamp", "metadata"]
        for field in required_fields:
            assert field in tool_call, f"Missing required field: {field}"
        
        # Verify timestamp format is ISO compliant
        timestamp = tool_call["timestamp"]
        assert timestamp.endswith("Z"), "Timestamp should be in UTC with Z suffix"
        
        # Verify metadata contains execution details
        metadata = tool_call["metadata"]
        execution_fields = ["working_directory", "command_line", "exit_code"]
        for field in execution_fields:
            assert field in metadata, f"Missing execution field: {field}"
    
    def test_markdown_export_compatibility(self, code_agent, sample_task, sample_goose_result):
        """Test markdown export compatibility fields."""
        # Create enhanced goose patch through _log_goose_result
        code_agent.shared_memory = MagicMock()
        
        # Call the method to generate enhanced format
        import asyncio
        asyncio.run(code_agent._log_goose_result(sample_task, sample_goose_result))
        
        # Get the generated goose patch
        memory_entry = code_agent.shared_memory.add_memory.call_args[0][0]
        goose_patch = memory_entry.goose_patch
        
        # Verify markdown export section
        markdown_export = goose_patch["markdown_export"]
        
        assert markdown_export["format_version"] == "1.0"
        assert markdown_export["exportable"] is True
        assert isinstance(markdown_export["session_name"], str)
        assert len(markdown_export["session_name"]) > 0
        assert isinstance(markdown_export["summary"], str)
        assert len(markdown_export["summary"]) > 0
        
        # Verify session name follows expected pattern
        session_name = markdown_export["session_name"]
        assert session_name.startswith("devguard-")
        assert "test-session" in session_name  # Should include truncated session ID


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
