"""Test cases for Goose CLI integration in QA Test Agent."""

from unittest.mock import Mock, patch

import pytest

from src.dev_guard.agents.qa_agent import QATestAgent
from src.dev_guard.core.config import AgentConfig, Config, LLMConfig, VectorDBConfig
from src.dev_guard.memory.shared_memory import SharedMemory
from src.dev_guard.memory.vector_db import VectorDatabase


@pytest.fixture
def config():
    """Create a test configuration."""
    return Config(
        agents=AgentConfig(
            max_retries=1,
            retry_delay=0.1
        ),
        llm=LLMConfig(
            provider="openrouter",
            model="meta-llama/llama-3.2-3b-instruct:free"
        ),
        vector_db=VectorDBConfig(
            provider="chroma",
            collection_name="test_collection"
        )
    )


@pytest.fixture
def memory():
    """Create a test memory instance."""
    return Mock(spec=SharedMemory)


@pytest.fixture
def vector_db():
    """Create a test vector database instance."""
    return Mock(spec=VectorDatabase)


@pytest.fixture
def qa_agent(config, memory, vector_db):
    """Create a QA Test Agent instance for testing."""
    with patch('src.dev_guard.agents.qa_test.QATestAgent._find_goose_executable', return_value="/usr/local/bin/goose"):
        agent = QATestAgent(
            agent_id="test-qa",
            config=config,
            shared_memory=memory,
            vector_db=vector_db
        )
        return agent


class TestGooseIntegration:
    """Test Goose CLI integration functionality."""

    def test_goose_capabilities_included(self, qa_agent):
        """Test that Goose-specific capabilities are included."""
        capabilities = qa_agent.get_capabilities()
        
        # Check for new Goose-specific capabilities
        assert "goose_fix_command" in capabilities
        assert "goose_write_tests" in capabilities
        assert "automated_qa_pipeline" in capabilities
        assert "code_repair_automation" in capabilities
        assert "intelligent_bug_fixing" in capabilities
        
        # Verify total capability count (26 including new ones)
        assert len(capabilities) >= 26

    @pytest.mark.asyncio
    @patch('subprocess.run')
    async def test_goose_fix_command_success(self, mock_subprocess, qa_agent):
        """Test successful Goose fix command execution."""
        # Mock successful subprocess execution
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.stdout = "Session started successfully\nFixed issues in target file"
        mock_process.stderr = ""
        mock_subprocess.return_value = mock_process
        
        # Mock file operations
        with patch('pathlib.Path.exists', return_value=True), \
             patch.object(qa_agent, '_find_test_file_for_target', return_value="test_target.py"), \
             patch.object(qa_agent, '_run_tests', return_value={"success": True}):
            
            task = {
                "target_file": "src/target.py",
                "error_description": "Bug in function logic",
                "fix_prompt": "Fix the logical error in the calculate function"
            }
            
            result = await qa_agent._goose_fix_command(task)
            
            assert result["success"] is True
            assert result["target_file"] == "src/target.py"
            assert result["fix_applied"] is True
            assert "Fixed issues" in result["goose_output"]
            assert result["verification_results"]["success"] is True

    @pytest.mark.asyncio
    @patch('subprocess.run')
    async def test_goose_fix_command_failure(self, mock_subprocess, qa_agent):
        """Test failed Goose fix command execution."""
        # Mock failed subprocess execution
        mock_process = Mock()
        mock_process.returncode = 1
        mock_process.stdout = ""
        mock_process.stderr = "Goose session failed to start"
        mock_subprocess.return_value = mock_process
        
        task = {
            "target_file": "src/nonexistent.py",
            "error_description": "File not found"
        }
        
        result = await qa_agent._goose_fix_command(task)
        
        assert result["success"] is False
        assert "Failed to start Goose session" in result["error"]

    @pytest.mark.asyncio
    @patch('subprocess.run')
    async def test_goose_write_tests_command_success(self, mock_subprocess, qa_agent):
        """Test successful Goose write-tests command execution."""
        # Mock successful subprocess execution
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.stdout = """
Test generation completed successfully.

```python
import pytest
from src.target import Calculator

def test_calculator_add():
    calc = Calculator()
    assert calc.add(2, 3) == 5

def test_calculator_divide():
    calc = Calculator()
    assert calc.divide(10, 2) == 5
```
"""
        mock_process.stderr = ""
        mock_subprocess.return_value = mock_process
        
        # Mock file operations
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.write_text') as mock_write, \
             patch('pathlib.Path.mkdir'), \
             patch.object(qa_agent, '_run_tests', return_value={"success": True}), \
             patch.object(qa_agent, '_analyze_coverage', return_value={"coverage_percentage": 85}):
            
            task = {
                "target_file": "src/target.py",
                "test_type": "unit",
                "coverage_target": "comprehensive",
                "test_framework": "pytest"
            }
            
            result = await qa_agent._goose_write_tests_command(task)
            
            assert result["success"] is True
            assert result["target_file"] == "src/target.py"
            assert result["tests_generated"] is True
            assert result["tests_saved"] is True
            assert result["test_framework"] == "pytest"
            assert "test_calculator_add" in result["generated_test_code"]
            
            # Verify test file was written
            mock_write.assert_called_once()

    @pytest.mark.asyncio
    @patch('subprocess.run')
    async def test_goose_write_tests_no_target_file(self, mock_subprocess, qa_agent):
        """Test write-tests command with missing target file."""
        task = {}  # No target_file specified
        
        result = await qa_agent._goose_write_tests_command(task)
        
        assert result["success"] is False
        assert "No target file specified" in result["error"]

    @pytest.mark.asyncio
    @patch('subprocess.run')
    async def test_automated_qa_pipeline_comprehensive(self, mock_subprocess, qa_agent):
        """Test comprehensive automated QA pipeline."""
        # Mock successful subprocess executions
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.stdout = "Pipeline operation successful"
        mock_process.stderr = ""
        mock_subprocess.return_value = mock_process
        
        # Mock file operations and methods
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.write_text'), \
             patch('pathlib.Path.mkdir'), \
             patch.object(qa_agent, '_run_tests', return_value={"success": True}), \
             patch.object(qa_agent, '_analyze_coverage', return_value={"coverage_percentage": 90}):
            
            task = {
                "target_files": ["src/file1.py", "src/file2.py"],
                "pipeline_type": "comprehensive",
                "fix_issues": True,
                "generate_tests": True
            }
            
            result = await qa_agent._run_automated_qa_pipeline(task)
            
            assert result["success"] is True
            assert result["files_processed"] == 2
            assert result["fixes_applied"] >= 0
            assert result["tests_generated"] >= 0
            assert "recommendations" in result
            assert "overall_coverage" in result

    @pytest.mark.asyncio
    async def test_automated_qa_pipeline_no_files(self, qa_agent):
        """Test QA pipeline with no target files."""
        task = {
            "target_files": [],  # Empty list
            "pipeline_type": "comprehensive"
        }
        
        result = await qa_agent._run_automated_qa_pipeline(task)
        
        assert result["success"] is False
        assert "No target files specified" in result["error"]

    def test_extract_code_changes_from_output(self, qa_agent):
        """Test code change extraction from Goose output."""
        output = """
Modified calculator.py
```python
def add(self, a, b):
    return a + b  # Fixed the logic
```

File: utils.py
Changes: Updated validation logic
"""
        
        changes = qa_agent._extract_code_changes_from_output(output)
        
        assert len(changes) >= 1
        # Should extract at least one change
        change_files = [change["file"] for change in changes if "file" in change]
        assert any("calculator.py" in f for f in change_files) or any("utils.py" in f for f in change_files)

    def test_create_goose_test_prompt(self, qa_agent):
        """Test Goose test prompt generation."""
        prompt = qa_agent._create_goose_test_prompt(
            "src/calculator.py",
            "unit",
            "comprehensive",
            "pytest"
        )
        
        assert "src/calculator.py" in prompt
        assert "pytest" in prompt
        assert "comprehensive" in prompt
        assert "Test type: unit" in prompt
        assert "fixtures and setup/teardown" in prompt

    def test_find_test_file_for_target(self, qa_agent):
        """Test finding corresponding test files."""
        # Mock file existence
        with patch('pathlib.Path.exists') as mock_exists:
            # First call returns False (test_calculator.py doesn't exist)
            # Second call returns True (calculator_test.py exists)
            mock_exists.side_effect = [False, True]
            
            result = qa_agent._find_test_file_for_target("src/calculator.py")
            assert result is not None
            assert "calculator" in result

    def test_determine_test_file_path(self, qa_agent):
        """Test determination of test file paths."""
        with patch('pathlib.Path.exists', return_value=False), \
             patch('pathlib.Path.mkdir') as mock_mkdir:
            
            result = qa_agent._determine_test_file_path("src/calculator.py", "pytest")
            
            assert "test_calculator.py" in result
            assert "tests" in result
            mock_mkdir.assert_called_once()

    def test_generate_qa_pipeline_recommendations(self, qa_agent):
        """Test QA pipeline recommendation generation."""
        pipeline_results = {
            "files_processed": 3,
            "fixes_applied": 2,
            "tests_generated": 1,
            "overall_coverage": {"coverage_percentage": 95},
            "file_results": {
                "file1.py": {"success": True},
                "file2.py": {"success": False}
            }
        }
        
        recommendations = qa_agent._generate_qa_pipeline_recommendations(pipeline_results)
        
        assert len(recommendations) > 0
        assert any("2 automated fixes" in rec for rec in recommendations)
        assert any("1 test suites" in rec for rec in recommendations)
        assert any("Excellent test coverage" in rec for rec in recommendations)

    @pytest.mark.asyncio
    async def test_task_routing_goose_commands(self, qa_agent):
        """Test task routing for Goose-specific commands."""
        # Mock the Goose command methods
        with patch.object(qa_agent, '_goose_fix_command', return_value={"success": True}) as mock_fix, \
             patch.object(qa_agent, '_goose_write_tests_command', return_value={"success": True}) as mock_write_tests, \
             patch.object(qa_agent, '_run_automated_qa_pipeline', return_value={"success": True}) as mock_pipeline:
            
            # Test goose_fix task routing
            fix_task = {"type": "goose_fix", "target_file": "test.py"}
            await qa_agent.execute_task(fix_task)
            mock_fix.assert_called_once_with(fix_task)
            
            # Test goose_write_tests task routing
            write_tests_task = {"type": "goose_write_tests", "target_file": "test.py"}
            await qa_agent.execute_task(write_tests_task)
            mock_write_tests.assert_called_once_with(write_tests_task)
            
            # Test automated_qa_pipeline task routing
            pipeline_task = {"type": "automated_qa_pipeline", "target_files": ["test.py"]}
            await qa_agent.execute_task(pipeline_task)
            mock_pipeline.assert_called_once_with(pipeline_task)
