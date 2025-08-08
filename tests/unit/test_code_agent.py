"""Tests for the Code Agent with Goose CLI integration."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

from dev_guard.agents.code_agent import CodeAgent
from dev_guard.core.config import Config
from dev_guard.memory.shared_memory import SharedMemory
from dev_guard.memory.vector_db import VectorDatabase


class TestCodeAgent:
    """Test suite for Code Agent."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = MagicMock(spec=Config)
        config.get_agent_config.return_value = {
            "max_retries": 3,
            "timeout": 300
        }
        return config

    @pytest.fixture
    def mock_shared_memory(self):
        """Create a mock shared memory."""
        memory = MagicMock(spec=SharedMemory)
        memory.update_agent_state = MagicMock()
        memory.add_memory = MagicMock(return_value="test-memory-id")
        return memory

    @pytest.fixture
    def mock_vector_db(self):
        """Create a mock vector database."""
        return MagicMock(spec=VectorDatabase)

    @pytest.fixture
    def code_agent(self, mock_config, mock_shared_memory, mock_vector_db):
        """Create a Code Agent instance for testing."""
        with patch('dev_guard.agents.code_agent.os.path.exists') as mock_exists:
            mock_exists.return_value = True
            
            agent = CodeAgent(
                agent_id="code_agent_test",
                config=mock_config,
                shared_memory=mock_shared_memory,
                vector_db=mock_vector_db,
                working_directory="/tmp/test"
            )
            
            # Mock the Goose path
            agent.goose_path = "/usr/local/bin/goose"
            return agent

    @pytest.mark.asyncio
    async def test_code_agent_initialization(self, code_agent):
        """Test Code Agent initialization."""
        assert code_agent.agent_id == "code_agent_test"
        assert code_agent.goose_path == "/usr/local/bin/goose"
        assert code_agent.working_directory == "/tmp/test"
        assert code_agent.session_id is None

    @pytest.mark.asyncio
    async def test_goose_cli_not_found(self, mock_config, mock_shared_memory, mock_vector_db):
        """Test error handling when Goose CLI is not found."""
        with patch('dev_guard.agents.code_agent.os.path.exists') as mock_exists, \
             patch('shutil.which') as mock_which:
            mock_exists.return_value = False
            mock_which.return_value = None
            
            with pytest.raises(RuntimeError, match="Goose CLI not found"):
                CodeAgent(
                    agent_id="code_agent_test",
                    config=mock_config,
                    shared_memory=mock_shared_memory,
                    vector_db=mock_vector_db
                )

    @pytest.mark.asyncio
    async def test_execute_with_dict_state(self, code_agent):
        """Test execute method with dictionary state."""
        task_state = {
            "type": "generate",
            "prompt": "Write a Python function to calculate fibonacci",
            "file_path": "fibonacci.py"
        }

        # Mock the Goose command execution
        with patch.object(code_agent, '_run_goose_command') as mock_run, \
             patch.object(code_agent, 'quality_check_and_format') as mock_quality, \
             patch.object(code_agent, 'analyze_code_structure') as mock_structure:
            
            mock_run.return_value = {
                "success": True,
                "output": "def fibonacci(n): ...",
                "error": None,
                "return_code": 0,
                "command": "goose session start",
                "session_id": "test-session-123"
            }
            
            # Mock quality check
            mock_quality.return_value = {"formatting_applied": [], "issues_found": []}
            
            # Mock structure analysis
            mock_structure.return_value = {"success": True}

            result = await code_agent.execute(task_state)

            # Verify result
            assert result["success"] is True
            assert result["agent"] == "code_agent"
            assert "task" in result
            assert "result" in result

            # Verify the Goose command was called (pattern search + generation)
            assert mock_run.call_count >= 1

    @pytest.mark.asyncio
    async def test_execute_with_string_state(self, code_agent):
        """Test execute method with string state."""
        task_state = "Write a sorting algorithm in Python"
        
        # Mock the Goose command execution
        with patch.object(code_agent, '_run_goose_command') as mock_run:
            mock_run.return_value = {
                "success": True,
                "output": "def bubble_sort(arr): ...",
                "error": None,
                "return_code": 0,
                "command": "goose session start",
                "session_id": "test-session-456"
            }
            
            result = await code_agent.execute(task_state)
            
            # Verify result
            assert result["success"] is True
            assert result["task"]["prompt"] == task_state
            assert result["task"]["type"] == "generate"

    @pytest.mark.asyncio
    async def test_execute_error_handling(self, code_agent):
        """Test execute method error handling."""
        task_state = {"type": "generate", "prompt": "test"}

        # Mock the Goose command to raise an exception
        with patch.object(code_agent, '_run_goose_command') as mock_run:
            mock_run.side_effect = Exception("Goose command failed")

            result = await code_agent.execute(task_state)

            # Verify error result
            assert result["success"] is False
            assert result["result"]["error"] == "Goose command failed"
            assert result["agent"] == "code_agent"

    @pytest.mark.asyncio
    async def test_generate_code(self, code_agent):
        """Test code generation functionality."""
        prompt = "Create a Python class for handling user authentication"
        file_path = "auth.py"

        with patch.object(code_agent, '_run_goose_command') as mock_run, \
             patch.object(code_agent, 'quality_check_and_format') as mock_quality, \
             patch.object(code_agent, 'analyze_code_structure') as mock_structure:
            
            mock_run.return_value = {
                "success": True,
                "output": "class UserAuth: ...",
                "error": None,
                "return_code": 0,
                "command": "goose session start --file auth.py",
                "session_id": "auth-session-123"
            }
            
            # Mock quality check
            mock_quality.return_value = {"formatting_applied": [], "issues_found": []}
            
            # Mock structure analysis
            mock_structure.return_value = {"success": True}

            result = await code_agent.generate_code(prompt, file_path)

            # Verify result
            assert result["success"] is True
            assert result["file_path"] == file_path
            assert "generated_code" in result
            assert "session_id" in result

            # Verify command was called (pattern search + generation)
            assert mock_run.call_count >= 1

    @pytest.mark.asyncio
    async def test_fix_code(self, code_agent):
        """Test code fixing functionality."""
        file_path = "buggy.py"
        error_description = "Fix the IndexError in list access"
        
        with patch.object(code_agent, '_run_goose_command') as mock_run:
            mock_run.return_value = {
                "success": True,
                "output": "Fixed the IndexError by adding bounds checking",
                "error": None,
                "return_code": 0,
                "command": "goose session start --file buggy.py",
                "session_id": "fix-session-456"
            }
            
            result = await code_agent.fix_code(file_path, error_description)
            
            # Verify result
            assert result["success"] is True
            assert result["file_path"] == file_path
            assert result["fix_description"] == error_description
            
            # Verify the fix prompt was constructed correctly
            expected_prompt = f"Fix the following issue in {file_path}: {error_description}"
            mock_run.assert_called_once_with(
                ["session", "start", "--file", file_path],
                input_text=expected_prompt
            )

    @pytest.mark.asyncio
    async def test_write_tests(self, code_agent):
        """Test test generation functionality."""
        file_path = "calculator.py"
        
        with patch.object(code_agent, '_run_goose_command') as mock_run:
            mock_run.return_value = {
                "success": True,
                "output": "Test file generated successfully",
                "error": None,
                "return_code": 0,
                "command": "goose session start --file calculator.py",
                "session_id": "test-session-789"
            }
            
            result = await code_agent.write_tests(file_path)
            
            # Verify result
            assert result["success"] is True
            assert result["source_file"] == file_path
            
            # Verify the test prompt was constructed correctly
            expected_prompt = f"Write comprehensive tests for the code in {file_path}"
            mock_run.assert_called_once_with(
                ["session", "start", "--file", file_path],
                input_text=expected_prompt
            )

    @pytest.mark.asyncio
    async def test_refactor_code(self, code_agent):
        """Test code refactoring functionality."""
        file_path = "legacy.py"
        refactor_description = "Extract methods and improve readability"

        with patch.object(code_agent, '_run_goose_command') as mock_run, \
             patch.object(code_agent, 'analyze_code_structure') as mock_analyze, \
             patch.object(code_agent, 'quality_check_and_format') as mock_quality:
            
            mock_run.return_value = {
                "success": True,
                "output": "Code refactored successfully",
                "error": None,
                "return_code": 0,
                "command": "goose session start --file legacy.py",
                "session_id": "refactor-session-101"
            }
            
            # Mock structure analysis
            mock_analyze.return_value = {"success": True, "complexity_metrics": {"cyclomatic_complexity": 5}}
            
            # Mock quality check
            mock_quality.return_value = {"formatting_applied": [], "issues_found": []}

            result = await code_agent.refactor_code(file_path, refactor_description)

            # Verify result
            assert result["success"] is True
            assert result["file_path"] == file_path
            assert result["refactor_description"] == refactor_description

            # Verify commands were called (pattern search + refactor + analysis)
            assert mock_run.call_count >= 1

    @pytest.mark.asyncio
    async def test_run_goose_session(self, code_agent):
        """Test generic Goose session execution."""
        prompt = "Generate a REST API endpoint"
        file_path = "api.py"
        goose_args = ["--model", "gpt-4"]
        
        with patch.object(code_agent, '_run_goose_command') as mock_run:
            mock_run.return_value = {
                "success": True,
                "output": "API endpoint created",
                "error": None,
                "return_code": 0,
                "command": "goose session start --model gpt-4 --file api.py",
                "session_id": "api-session-202"
            }
            
            result = await code_agent.run_goose_session(prompt, file_path, goose_args)
            
            # Verify result
            assert result["success"] is True
            assert result["prompt"] == prompt
            assert result["file_path"] == file_path
            
            # Verify command arguments
            expected_args = ["session", "start", "--model", "gpt-4", "--file", file_path]
            mock_run.assert_called_once_with(expected_args, input_text=prompt)

    @pytest.mark.asyncio
    async def test_run_goose_command_success(self, code_agent):
        """Test successful Goose command execution."""
        args = ["session", "start"]
        input_text = "Hello Goose"
        
        # Mock subprocess execution
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (
            b"Goose session started successfully",
            b""
        )
        
        with patch('dev_guard.agents.code_agent.asyncio.create_subprocess_exec') as mock_exec:
            mock_exec.return_value = mock_process
            
            result = await code_agent._run_goose_command(args, input_text)
            
            # Verify result
            assert result["success"] is True
            assert result["output"] == "Goose session started successfully"
            assert result["error"] == ""
            assert result["return_code"] == 0
            assert "session_id" in result
            
            # Verify subprocess was called correctly
            mock_exec.assert_called_once_with(
                "/usr/local/bin/goose", "session", "start",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd="/tmp/test"
            )

    @pytest.mark.asyncio
    async def test_run_goose_command_failure(self, code_agent):
        """Test failed Goose command execution."""
        args = ["invalid", "command"]
        
        # Mock subprocess execution with failure
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate.return_value = (
            b"",
            b"Unknown command: invalid"
        )
        
        with patch('dev_guard.agents.code_agent.asyncio.create_subprocess_exec') as mock_exec:
            mock_exec.return_value = mock_process
            
            result = await code_agent._run_goose_command(args)
            
            # Verify result
            assert result["success"] is False
            assert result["output"] == ""
            assert result["error"] == "Unknown command: invalid"
            assert result["return_code"] == 1

    @pytest.mark.asyncio
    async def test_run_goose_command_exception(self, code_agent):
        """Test exception handling in Goose command execution."""
        args = ["test"]
        
        with patch('dev_guard.agents.code_agent.asyncio.create_subprocess_exec') as mock_exec:
            mock_exec.side_effect = Exception("Process creation failed")
            
            result = await code_agent._run_goose_command(args)
            
            # Verify result
            assert result["success"] is False
            assert result["error"] == "Process creation failed"
            assert result["return_code"] == -1

    def test_parse_state_for_task_string(self, code_agent):
        """Test state parsing with string input."""
        state = "Write a Python function"
        
        task = code_agent._parse_state_for_task(state)
        
        assert task is not None
        assert task["type"] == "generate"
        assert task["prompt"] == "Write a Python function"
        assert task["file_path"] is None

    def test_parse_state_for_task_dict(self, code_agent):
        """Test state parsing with dictionary input."""
        state = {
            "type": "fix",
            "description": "Fix the bug",
            "file_path": "bug.py",
            "context": {"line": 42}
        }
        
        task = code_agent._parse_state_for_task(state)
        
        assert task is not None
        assert task["type"] == "fix"
        assert task["prompt"] == "Fix the bug"
        assert task["file_path"] == "bug.py"
        assert task["context"] == {"line": 42}

    def test_parse_state_for_task_invalid(self, code_agent):
        """Test state parsing with invalid input."""
        state = 12345  # Not string or dict-like
        
        task = code_agent._parse_state_for_task(state)
        
        assert task is None

    @pytest.mark.asyncio
    async def test_log_goose_result(self, code_agent):
        """Test Goose result logging to shared memory."""
        task = {
            "type": "generate",
            "prompt": "Test task",
            "file_path": "test.py"
        }
        
        result = {
            "success": True,
            "session_id": "test-session-303",
            "goose_output": {
                "command": "goose session start",
                "output": "Task completed",
                "error": None,
                "return_code": 0
            }
        }
        
        await code_agent._log_goose_result(task, result)
        
        # Verify memory entry was added
        code_agent.shared_memory.add_memory.assert_called_once()
        
        # Get the memory entry that was added
        call_args = code_agent.shared_memory.add_memory.call_args[0]
        memory_entry = call_args[0]
        
        # Verify memory entry content
        assert memory_entry.agent_id == "code_agent_test"
        assert memory_entry.type == "result"
        assert "goose" in memory_entry.tags
        assert "code_generation" in memory_entry.tags
        assert memory_entry.goose_patch is not None
        assert memory_entry.goose_strategy == "generate"
        assert memory_entry.file_path == "test.py"

    def test_get_capabilities(self, code_agent):
        """Test getting Code Agent capabilities."""
        capabilities = code_agent.get_capabilities()

        expected_capabilities = [
            "code_generation",
            "code_fixing",
            "test_writing",
            "code_refactoring",
            "goose_cli_integration",
            "code_formatting",
            "code_linting",
            "quality_checking",
            "auto_fixing",
            "ast_analysis",
            "pattern_matching",
            "goose_memory_search",
            "structural_similarity",
            "refactoring_impact_analysis"
        ]

        assert capabilities == expected_capabilities

    @pytest.mark.asyncio
    async def test_execute_task_types(self, code_agent):
        """Test different task types in _execute_code_task."""
        # Test generate task
        generate_task = {"type": "generate", "prompt": "test"}
        with patch.object(code_agent, 'generate_code') as mock_generate:
            mock_generate.return_value = {"success": True}
            result = await code_agent._execute_code_task(generate_task)
            mock_generate.assert_called_once()
        
        # Test fix task
        fix_task = {"type": "fix", "file_path": "test.py", "error_description": "error"}
        with patch.object(code_agent, 'fix_code') as mock_fix:
            mock_fix.return_value = {"success": True}
            result = await code_agent._execute_code_task(fix_task)
            mock_fix.assert_called_once()
        
        # Test test task
        test_task = {"type": "test", "file_path": "test.py"}
        with patch.object(code_agent, 'write_tests') as mock_test:
            mock_test.return_value = {"success": True}
            result = await code_agent._execute_code_task(test_task)
            mock_test.assert_called_once()
        
        # Test refactor task
        refactor_task = {"type": "refactor", "file_path": "test.py", "refactor_description": "refactor"}
        with patch.object(code_agent, 'refactor_code') as mock_refactor:
            mock_refactor.return_value = {"success": True}
            result = await code_agent._execute_code_task(refactor_task)
            mock_refactor.assert_called_once()
        
        # Test generic task
        generic_task = {"type": "custom", "prompt": "custom task"}
        with patch.object(code_agent, 'run_goose_session') as mock_session:
            mock_session.return_value = {"success": True}
            result = await code_agent._execute_code_task(generic_task)
            mock_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_id_generation(self, code_agent):
        """Test that session IDs are generated properly."""
        args = ["session", "start"]
        
        # Mock subprocess execution
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"success", b"")
        
        with patch('dev_guard.agents.code_agent.asyncio.create_subprocess_exec') as mock_exec:
            mock_exec.return_value = mock_process
            
            # First call should generate a session ID
            result1 = await code_agent._run_goose_command(args)
            assert result1["session_id"] is not None
            first_session_id = result1["session_id"]
            
            # Session ID should be persisted
            assert code_agent.session_id == first_session_id
            
            # Second call should use the same session ID
            result2 = await code_agent._run_goose_command(args)
            assert result2["session_id"] == first_session_id

    @pytest.mark.asyncio
    async def test_working_directory_handling(self, code_agent):
        """Test working directory handling in command execution."""
        args = ["test"]
        custom_cwd = "/custom/working/dir"
        
        # Mock subprocess execution
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"success", b"")
        
        with patch('dev_guard.agents.code_agent.asyncio.create_subprocess_exec') as mock_exec:
            mock_exec.return_value = mock_process
            
            # Test with custom working directory
            await code_agent._run_goose_command(args, cwd=custom_cwd)
            
            # Verify custom cwd was used
            mock_exec.assert_called_once()
            call_args = mock_exec.call_args
            assert call_args[1]['cwd'] == custom_cwd
            
            mock_exec.reset_mock()
            
            # Test with default working directory
            await code_agent._run_goose_command(args)
            
            # Verify default cwd was used
            mock_exec.assert_called_once()
            call_args = mock_exec.call_args
            assert call_args[1]['cwd'] == "/tmp/test"


class TestCodeAgentQualityIntegration:
    """Test suite for Code Agent quality checking and formatting integration."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = MagicMock(spec=Config)
        config.get_agent_config.return_value = {
            "max_retries": 3,
            "timeout": 300
        }
        return config

    @pytest.fixture
    def mock_shared_memory(self):
        """Create a mock shared memory."""
        memory = MagicMock(spec=SharedMemory)
        memory.update_agent_state = MagicMock()
        memory.add_memory = MagicMock(return_value="test-memory-id")
        return memory

    @pytest.fixture
    def mock_vector_db(self):
        """Create a mock vector database."""
        return MagicMock(spec=VectorDatabase)

    @pytest.fixture
    def code_agent(self, mock_config, mock_shared_memory, mock_vector_db):
        """Create a Code Agent instance for testing."""
        with patch('dev_guard.agents.code_agent.os.path.exists') as mock_exists:
            mock_exists.return_value = True
            
            agent = CodeAgent(
                agent_id="code_agent_test",
                config=mock_config,
                shared_memory=mock_shared_memory,
                vector_db=mock_vector_db,
                working_directory="/tmp/test"
            )
            return agent

    @pytest.mark.asyncio
    async def test_format_code_success(self, code_agent):
        """Test successful code formatting with black and isort."""
        test_file = "/tmp/test_file.py"
        
        with patch('dev_guard.agents.code_agent.Path') as mock_path:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            
            with patch.object(code_agent, '_run_command_simple') as mock_run:
                # Mock successful black and isort runs
                mock_run.side_effect = [
                    {"returncode": 0, "stdout": "formatted", "stderr": ""},
                    {"returncode": 0, "stdout": "imports sorted", "stderr": ""}
                ]
                
                result = await code_agent.format_code(test_file)
                
                assert result["success"] is True
                assert "black" in result["formatters_applied"]
                assert "isort" in result["formatters_applied"]
                assert len(result["errors"]) == 0
                
                # Verify both tools were called
                assert mock_run.call_count == 2
                mock_run.assert_any_call(["python", "-m", "black", test_file])
                mock_run.assert_any_call(["python", "-m", "isort", test_file])

    @pytest.mark.asyncio
    async def test_format_code_file_not_exists(self, code_agent):
        """Test formatting when file doesn't exist."""
        test_file = "/tmp/nonexistent.py"
        
        with patch('dev_guard.agents.code_agent.Path') as mock_path:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = False
            mock_path.return_value = mock_path_instance
            
            result = await code_agent.format_code(test_file)
            
            assert result["success"] is False
            assert "does not exist" in result["error"]

    @pytest.mark.asyncio
    async def test_format_code_partial_failure(self, code_agent):
        """Test formatting when one tool fails."""
        test_file = "/tmp/test_file.py"
        
        with patch('dev_guard.agents.code_agent.Path') as mock_path:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            
            with patch.object(code_agent, '_run_command_simple') as mock_run:
                # Mock black success, isort failure
                mock_run.side_effect = [
                    {"returncode": 0, "stdout": "formatted", "stderr": ""},
                    {"returncode": 1, "stdout": "", "stderr": "syntax error"}
                ]
                
                result = await code_agent.format_code(test_file)
                
                assert result["success"] is True  # Still success if one formatter works
                assert "black" in result["formatters_applied"]
                assert "isort" not in result["formatters_applied"]
                assert len(result["errors"]) == 1
                assert "isort error" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_lint_code_success(self, code_agent):
        """Test successful code linting with ruff and mypy."""
        test_file = "/tmp/test_file.py"
        
        with patch('dev_guard.agents.code_agent.Path') as mock_path:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            
            with patch.object(code_agent, '_run_command_simple') as mock_run:
                # Mock ruff with JSON output and mypy with text output
                ruff_output = '[{"filename": "test.py", "location": {"row": 1, "column": 1}, "code": "E302", "message": "expected 2 blank lines"}]'
                mypy_output = "test.py:5:10: error: Incompatible types [assignment]"
                
                mock_run.side_effect = [
                    {"returncode": 1, "stdout": ruff_output, "stderr": ""},
                    {"returncode": 1, "stdout": mypy_output, "stderr": ""}
                ]
                
                result = await code_agent.lint_code(test_file)
                
                assert result["success"] is True
                assert "ruff" in result["linters_run"]
                assert "mypy" in result["linters_run"]
                assert len(result["issues"]) == 2
                
                # Check ruff issue format
                ruff_issue = next(i for i in result["issues"] if i["type"] == "ruff")
                assert ruff_issue["code"] == "E302"
                assert ruff_issue["line"] == 1
                assert ruff_issue["column"] == 1
                
                # Check mypy issue format
                mypy_issue = next(i for i in result["issues"] if i["type"] == "mypy")
                assert mypy_issue["line"] == 5
                assert mypy_issue["column"] == 10

    @pytest.mark.asyncio
    async def test_lint_code_no_issues(self, code_agent):
        """Test linting when no issues are found."""
        test_file = "/tmp/clean_file.py"
        
        with patch('dev_guard.agents.code_agent.Path') as mock_path:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            
            with patch.object(code_agent, '_run_command_simple') as mock_run:
                # Mock clean runs
                mock_run.side_effect = [
                    {"returncode": 0, "stdout": "[]", "stderr": ""},
                    {"returncode": 0, "stdout": "", "stderr": ""}
                ]
                
                result = await code_agent.lint_code(test_file)
                
                assert result["success"] is True
                assert len(result["issues"]) == 0
                assert len(result["errors"]) == 0

    @pytest.mark.asyncio
    async def test_quality_check_and_format_complete_pipeline(self, code_agent):
        """Test complete quality check and format pipeline."""
        test_file = "/tmp/test_file.py"
        
        with patch('dev_guard.agents.code_agent.Path') as mock_path:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            
            with patch.object(code_agent, 'format_code') as mock_format, \
                 patch.object(code_agent, 'lint_code') as mock_lint, \
                 patch.object(code_agent, '_auto_fix_issues') as mock_auto_fix:
                
                # Mock successful formatting
                mock_format.return_value = {
                    "success": True,
                    "formatters_applied": ["black", "isort"],
                    "errors": []
                }
                
                # Mock linting with fixable issues
                mock_lint.return_value = {
                    "success": True,
                    "issues": [
                        {"type": "ruff", "code": "F401", "message": "unused import"}
                    ],
                    "errors": []
                }
                
                # Mock successful auto-fix
                mock_auto_fix.return_value = {
                    "success": True,
                    "fixes_applied": ["ruff_auto_fix", "black_reformat"]
                }
                
                result = await code_agent.quality_check_and_format(test_file, auto_fix=True)
                
                assert result["success"] is True
                assert "formatting" in result["steps_completed"]
                assert "linting" in result["steps_completed"]
                assert "auto_fix" in result["steps_completed"]
                assert result["formatting_applied"] == ["black", "isort"]
                assert len(result["issues_found"]) == 1
                assert result["auto_fixes_applied"] == ["ruff_auto_fix", "black_reformat"]

    @pytest.mark.asyncio
    async def test_quality_check_and_format_no_auto_fix(self, code_agent):
        """Test quality pipeline without auto-fix."""
        test_file = "/tmp/test_file.py"
        
        with patch.object(code_agent, 'format_code') as mock_format, \
             patch.object(code_agent, 'lint_code') as mock_lint:
            
            mock_format.return_value = {"success": True, "formatters_applied": ["black"]}
            mock_lint.return_value = {"success": True, "issues": [{"type": "ruff"}]}
            
            result = await code_agent.quality_check_and_format(test_file, auto_fix=False)
            
            assert result["success"] is True
            assert "auto_fix" not in result["steps_completed"]
            assert len(result["auto_fixes_applied"]) == 0

    @pytest.mark.asyncio
    async def test_auto_fix_issues(self, code_agent):
        """Test automatic fixing of code issues."""
        test_file = "/tmp/test_file.py"
        issues = [
            {"type": "ruff", "code": "F401", "message": "unused import"}
        ]
        
        with patch.object(code_agent, '_run_command_simple') as mock_run, \
             patch.object(code_agent, '_run_black') as mock_black, \
             patch.object(code_agent, '_run_isort') as mock_isort:
            
            # Mock ruff auto-fix success
            mock_run.return_value = {"returncode": 0, "stdout": "", "stderr": ""}
            mock_black.return_value = {"success": True}
            mock_isort.return_value = {"success": True}
            
            result = await code_agent._auto_fix_issues(test_file, issues)
            
            assert result["success"] is True
            assert "ruff_auto_fix" in result["fixes_applied"]
            assert "black_reformat" in result["fixes_applied"]
            assert "isort_reformat" in result["fixes_applied"]
            
            # Verify ruff fix command was called
            mock_run.assert_called_once_with(["python", "-m", "ruff", "check", test_file, "--fix"])

    @pytest.mark.asyncio
    async def test_generate_code_with_quality_integration(self, code_agent):
        """Test code generation with automatic quality checking."""
        test_prompt = "Generate a simple Python function"
        test_file = "/tmp/generated.py"
        
        with patch.object(code_agent, '_run_goose_command') as mock_goose, \
             patch.object(code_agent, 'quality_check_and_format') as mock_quality, \
             patch('dev_guard.agents.code_agent.Path') as mock_path:
            
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            
            # Mock successful Goose generation
            mock_goose.return_value = {
                "success": True,
                "output": "def hello(): return 'world'"
            }
            
            # Mock quality check
            mock_quality.return_value = {
                "formatting_applied": ["black"],
                "issues_found": [],
                "auto_fixes_applied": []
            }
            
            result = await code_agent.generate_code(test_prompt, test_file)
            
            assert result["success"] is True
            assert "quality_check" in result
            assert "formatting_applied" in result
            mock_quality.assert_called_once_with(test_file, auto_fix=True)

    @pytest.mark.asyncio
    async def test_fix_code_with_quality_integration(self, code_agent):
        """Test code fixing with automatic quality checking."""
        test_file = "/tmp/buggy.py"
        error_desc = "Fix syntax error"
        
        with patch.object(code_agent, '_run_goose_command') as mock_goose, \
             patch.object(code_agent, 'quality_check_and_format') as mock_quality, \
             patch('dev_guard.agents.code_agent.Path') as mock_path:
            
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            
            mock_goose.return_value = {"success": True}
            mock_quality.return_value = {
                "formatting_applied": ["black", "isort"],
                "issues_found": [],
                "auto_fixes_applied": ["ruff_auto_fix"]
            }
            
            result = await code_agent.fix_code(test_file, error_desc)
            
            assert result["success"] is True
            assert result["quality_check"]["formatting_applied"] == ["black", "isort"]
            assert result["auto_fixes_applied"] == ["ruff_auto_fix"]

    @pytest.mark.asyncio
    async def test_refactor_code_with_quality_integration(self, code_agent):
        """Test code refactoring with automatic quality checking."""
        test_file = "/tmp/messy.py"
        refactor_desc = "Extract method and improve readability"
        
        with patch.object(code_agent, '_run_goose_command') as mock_goose, \
             patch.object(code_agent, 'quality_check_and_format') as mock_quality, \
             patch('dev_guard.agents.code_agent.Path') as mock_path:
            
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            
            mock_goose.return_value = {"success": True}
            mock_quality.return_value = {
                "formatting_applied": ["black"],
                "issues_found": [{"type": "mypy", "message": "type hint missing"}],
                "auto_fixes_applied": []
            }
            
            result = await code_agent.refactor_code(test_file, refactor_desc)
            
            assert result["success"] is True
            assert len(result["issues_found"]) == 1
            assert result["issues_found"][0]["type"] == "mypy"

    def test_updated_capabilities(self, code_agent):
        """Test that capabilities include new AST and memory features."""
        capabilities = code_agent.get_capabilities()
        
        assert "code_formatting" in capabilities
        assert "code_linting" in capabilities
        assert "quality_checking" in capabilities
        assert "auto_fixing" in capabilities
        assert "goose_cli_integration" in capabilities
        assert "ast_analysis" in capabilities
        assert "pattern_matching" in capabilities
        assert "goose_memory_search" in capabilities
        assert "structural_similarity" in capabilities
        assert "refactoring_impact_analysis" in capabilities


class TestCodeAgentASTIntegration:
    """Test suite for Code Agent AST analysis and Goose memory integration."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = MagicMock(spec=Config)
        config.get_agent_config.return_value = {
            "max_retries": 3,
            "timeout": 300
        }
        return config

    @pytest.fixture
    def mock_shared_memory(self):
        """Create a mock shared memory."""
        memory = MagicMock(spec=SharedMemory)
        memory.update_agent_state = MagicMock()
        memory.add_memory = MagicMock(return_value="test-memory-id")
        return memory

    @pytest.fixture
    def mock_vector_db(self):
        """Create a mock vector database."""
        db = MagicMock(spec=VectorDatabase)
        db.search = MagicMock(return_value={
            "documents": [
                {"content": "sample code", "metadata": {"file_path": "/test.py"}, "score": 0.8}
            ]
        })
        return db

    @pytest.fixture
    def code_agent(self, mock_config, mock_shared_memory, mock_vector_db):
        """Create a Code Agent instance for testing."""
        with patch('dev_guard.agents.code_agent.os.path.exists') as mock_exists:
            mock_exists.return_value = True
            
            agent = CodeAgent(
                agent_id="code_agent_test",
                config=mock_config,
                shared_memory=mock_shared_memory,
                vector_db=mock_vector_db,
                working_directory="/tmp/test"
            )
            return agent

    @pytest.fixture
    def sample_python_code(self):
        """Sample Python code for testing."""
        return '''
import os
from pathlib import Path

class Calculator:
    """A simple calculator class."""
    
    def __init__(self):
        self.history = []
    
    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers."""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result

def create_calculator() -> Calculator:
    """Factory function for Calculator."""
    return Calculator()
'''

    @pytest.mark.asyncio
    async def test_analyze_code_structure(self, code_agent, sample_python_code):
        """Test AST analysis of Python code structure."""
        test_file = "/tmp/test_calculator.py"
        
        with patch('dev_guard.agents.code_agent.Path') as mock_path, \
             patch('builtins.open', mock_open(read_data=sample_python_code)):
            
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            
            result = await code_agent.analyze_code_structure(test_file)
            
            assert result["success"] is True
            assert result["file_path"] == test_file
            
            # Check classes detection
            assert len(result["classes"]) == 1
            calculator_class = result["classes"][0]
            assert calculator_class["name"] == "Calculator"
            assert "add" in calculator_class["methods"]
            assert "multiply" in calculator_class["methods"]
            
            # Check functions detection
            functions = [f["name"] for f in result["functions"]]
            assert "create_calculator" in functions
            
            # Check imports detection
            imports = result["imports"]
            assert len(imports) == 2
            import_modules = [imp.get("module", "") for imp in imports]
            assert "os" in import_modules
            assert "pathlib" in import_modules
            
            # Check complexity metrics
            metrics = result["complexity_metrics"]
            assert metrics["classes_count"] == 1
            assert metrics["functions_count"] >= 1  # At least 1 function
            assert metrics["cyclomatic_complexity"] > 0

    @pytest.mark.asyncio
    async def test_analyze_code_structure_file_not_exists(self, code_agent):
        """Test AST analysis when file doesn't exist."""
        test_file = "/tmp/nonexistent.py"
        
        with patch('dev_guard.agents.code_agent.Path') as mock_path:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = False
            mock_path.return_value = mock_path_instance
            
            result = await code_agent.analyze_code_structure(test_file)
            
            assert result["success"] is False
            assert "does not exist" in result["error"]

    @pytest.mark.asyncio
    async def test_analyze_code_structure_syntax_error(self, code_agent):
        """Test AST analysis with syntax errors."""
        test_file = "/tmp/syntax_error.py"
        bad_code = "def broken_function(\n    return 42"  # Missing closing parenthesis
        
        with patch('dev_guard.agents.code_agent.Path') as mock_path, \
             patch('builtins.open', mock_open(read_data=bad_code)):
            
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            
            result = await code_agent.analyze_code_structure(test_file)
            
            assert result["success"] is False
            assert "Syntax error" in result["error"]

    @pytest.mark.asyncio
    async def test_search_similar_patterns(self, code_agent):
        """Test pattern search using Goose memory and vector DB."""
        query = "calculator class with add method"
        
        with patch.object(code_agent, '_search_goose_memory') as mock_goose, \
             patch.object(code_agent, '_search_vector_db') as mock_vector:
            
            # Mock Goose memory results
            mock_goose.return_value = {
                "success": True,
                "matches": [
                    {"code": "class Calculator:", "confidence": 0.9, "context": "math operations"}
                ]
            }
            
            # Mock vector DB results
            mock_vector.return_value = {
                "success": True,
                "matches": [
                    {"content": "def add(a, b): return a + b", "similarity_score": 0.8}
                ]
            }
            
            result = await code_agent.search_similar_patterns(query)
            
            assert result["success"] is True
            assert len(result["goose_matches"]) == 1
            assert len(result["recommended_patterns"]) > 0
            assert result["recommended_patterns"][0]["source"] == "goose_memory"

    @pytest.mark.asyncio
    async def test_search_similar_patterns_goose_fallback_to_vector(self, code_agent):
        """Test fallback to vector DB when Goose memory returns few results."""
        query = "simple function"
        
        with patch.object(code_agent, '_search_goose_memory') as mock_goose, \
             patch.object(code_agent, '_search_vector_db') as mock_vector:
            
            # Mock Goose with minimal results (< 3)
            mock_goose.return_value = {
                "success": True,
                "matches": [{"code": "def func():", "confidence": 0.7}]
            }
            
            # Mock vector DB results
            mock_vector.return_value = {
                "success": True,
                "matches": [
                    {"content": "def example(): pass", "similarity_score": 0.8}
                ]
            }
            
            result = await code_agent.search_similar_patterns(query)
            
            assert result["success"] is True
            assert len(result["goose_matches"]) == 1
            assert len(result["vector_matches"]) == 1
            mock_vector.assert_called_once()  # Should fallback to vector search

    @pytest.mark.asyncio
    async def test_search_goose_memory(self, code_agent):
        """Test Goose memory search functionality."""
        query = "refactor method extraction"
        
        with patch.object(code_agent, '_run_goose_command') as mock_goose_cmd:
            mock_goose_cmd.return_value = {
                "success": True,
                "output": """Memory entry: Method extraction pattern
Code: def extract_method(self, lines): ...
Context: Common refactoring pattern
Confidence: 0.85"""
            }
            
            result = await code_agent._search_goose_memory(query)
            
            assert result["success"] is True
            assert len(result["matches"]) == 1
            assert result["matches"][0]["confidence"] == 0.85
            
            # Verify correct command was called
            mock_goose_cmd.assert_called_once()
            args = mock_goose_cmd.call_args[0][0]
            assert "session" in args
            assert "Search memory for patterns similar to:" in " ".join(args)

    @pytest.mark.asyncio
    async def test_calculate_structural_similarity(self, code_agent):
        """Test structural similarity calculation between code structures."""
        struct1 = {
            "success": True,
            "classes": [{"name": "Calculator", "methods": ["add", "subtract"]}],
            "functions": [{"name": "main"}],
            "imports": [{"module": "math"}],
            "complexity_metrics": {"cyclomatic_complexity": 5}
        }
        
        struct2 = {
            "success": True,
            "classes": [{"name": "Calculator", "methods": ["add", "multiply"]}],
            "functions": [{"name": "helper"}],
            "imports": [{"module": "math"}],
            "complexity_metrics": {"cyclomatic_complexity": 4}
        }
        
        similarity = code_agent._calculate_structural_similarity(struct1, struct2)
        
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.3  # Should have some similarity due to shared class and import

    @pytest.mark.asyncio
    async def test_generate_code_with_pattern_integration(self, code_agent):
        """Test code generation with pattern matching integration."""
        prompt = "Create a calculator class"
        file_path = "/tmp/calculator.py"
        
        with patch.object(code_agent, 'search_similar_patterns') as mock_search, \
             patch.object(code_agent, '_run_goose_command') as mock_goose, \
             patch.object(code_agent, 'quality_check_and_format') as mock_quality, \
             patch.object(code_agent, 'analyze_code_structure') as mock_structure, \
             patch('dev_guard.agents.code_agent.Path') as mock_path:
            
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.suffix = ".py"
            mock_path.return_value = mock_path_instance
            
            # Mock pattern search results
            mock_search.return_value = {
                "success": True,
                "recommended_patterns": [
                    {"source": "goose_memory", "confidence": 0.9, "pattern": "class Calculator:", "reason": "Similar class"}
                ]
            }
            
            # Mock Goose command success
            mock_goose.return_value = {"success": True, "output": "Generated calculator class"}
            
            # Mock quality check
            mock_quality.return_value = {"formatting_applied": ["black"], "issues_found": []}
            
            # Mock structure analysis
            mock_structure.return_value = {"success": True, "classes": [{"name": "Calculator"}]}
            
            result = await code_agent.generate_code(prompt, file_path)
            
            assert result["success"] is True
            assert "pattern_analysis" in result
            assert "structure_analysis" in result
            
            # Verify pattern search was called
            mock_search.assert_called_once_with(prompt, file_path)
            
            # Verify enhanced prompt was used (should contain pattern context)
            goose_call_args = mock_goose.call_args
            assert "Similar patterns found:" in goose_call_args[1]["input_text"]

    @pytest.mark.asyncio
    async def test_refactor_code_with_impact_analysis(self, code_agent, sample_python_code):
        """Test refactoring with before/after structure analysis."""
        file_path = "/tmp/calculator.py"
        refactor_desc = "Extract method for logging"
        
        with patch.object(code_agent, 'analyze_code_structure') as mock_analyze, \
             patch.object(code_agent, 'search_similar_patterns') as mock_search, \
             patch.object(code_agent, '_run_goose_command') as mock_goose, \
             patch.object(code_agent, 'quality_check_and_format') as mock_quality, \
             patch('dev_guard.agents.code_agent.Path') as mock_path:
            
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            
            # Mock structure analysis (before and after)
            structure_before = {"success": True, "complexity_metrics": {"cyclomatic_complexity": 8}}
            structure_after = {"success": True, "complexity_metrics": {"cyclomatic_complexity": 6}}
            
            mock_analyze.side_effect = [structure_before, structure_after]
            
            # Mock pattern search
            mock_search.return_value = {"success": True, "recommended_patterns": []}
            
            # Mock Goose refactoring
            mock_goose.return_value = {"success": True, "output": "Refactored code"}
            
            # Mock quality check
            mock_quality.return_value = {"formatting_applied": ["black"], "issues_found": []}
            
            result = await code_agent.refactor_code(file_path, refactor_desc)
            
            assert result["success"] is True
            assert "structure_before" in result
            assert "structure_after" in result
            assert "refactoring_impact" in result
            
            # Check impact analysis
            impact = result["refactoring_impact"]
            assert impact["success"] is True
            assert impact["complexity_change"]["improvement"] == 2  # 8 - 6
            assert impact["quality_assessment"] == "improved"

    @pytest.mark.asyncio
    async def test_analyze_refactoring_impact(self, code_agent):
        """Test refactoring impact analysis calculation."""
        structure_before = {
            "success": True,
            "complexity_metrics": {
                "cyclomatic_complexity": 10,
                "classes_count": 2,
                "functions_count": 5,
                "total_nodes": 100
            }
        }
        
        structure_after = {
            "success": True,
            "complexity_metrics": {
                "cyclomatic_complexity": 7,
                "classes_count": 3,
                "functions_count": 6,
                "total_nodes": 95
            }
        }
        
        impact = code_agent._analyze_refactoring_impact(structure_before, structure_after)
        
        assert impact["success"] is True
        assert impact["complexity_change"]["improvement"] == 3  # 10 - 7
        assert impact["quality_assessment"] == "improved"
        assert impact["structure_changes"]["classes_change"] == 1  # 3 - 2
        assert impact["structure_changes"]["functions_change"] == 1  # 6 - 5

    def test_build_pattern_context(self, code_agent):
        """Test building pattern context string from recommendations."""
        patterns = [
            {
                "source": "goose_memory",
                "confidence": 0.9,
                "pattern": "class Calculator:",
                "reason": "Historical pattern"
            },
            {
                "source": "vector_search",
                "confidence": 0.8,
                "pattern": "def add(self, a, b):",
                "reason": "Similar method"
            }
        ]
        
        context = code_agent._build_pattern_context(patterns)
        
        assert "Pattern 1 (from goose_memory, confidence: 0.90):" in context
        assert "class Calculator:" in context
        assert "Pattern 2 (from vector_search, confidence: 0.80):" in context
        assert "def add(self, a, b):" in context
