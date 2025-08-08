"""Tests for enhanced QA Test Agent (Task 11.1)."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from dev_guard.agents.qa_agent import QATestAgent
from dev_guard.core.config import Config
from dev_guard.memory.shared_memory import SharedMemory
from dev_guard.memory.vector_db import VectorDatabase


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = MagicMock(spec=Config)
    config.get_agent_config.return_value = {
        "max_retries": 3,
        "timeout": 300
    }
    return config


@pytest.fixture
def mock_shared_memory():
    """Create a mock shared memory."""
    memory = MagicMock(spec=SharedMemory)
    memory.update_agent_state = MagicMock()
    memory.add_memory = MagicMock(return_value="test-memory-id")
    return memory


@pytest.fixture
def mock_vector_db():
    """Create a mock vector database."""
    return MagicMock(spec=VectorDatabase)


@pytest.fixture
def qa_agent(mock_config, mock_shared_memory, mock_vector_db):
    """Create a QA Test Agent instance for testing."""
    with patch('shutil.which') as mock_which:
        mock_which.return_value = "/usr/local/bin/goose"
        
        agent = QATestAgent(
            agent_id="qa_test_agent",
            config=mock_config,
            shared_memory=mock_shared_memory,
            vector_db=mock_vector_db
        )
        return agent


@pytest.fixture
def sample_test_file():
    """Create a sample test file."""
    return '''
import pytest
from unittest.mock import Mock, patch

def test_sample_function():
    """Test basic functionality."""
    assert True

def test_edge_case():
    """Test edge cases."""
    with pytest.raises(ValueError):
        raise ValueError("Test error")

@pytest.fixture
def sample_fixture():
    return {"data": "test"}

def test_with_mock():
    """Test with mocking."""
    with patch('module.function') as mock_func:
        mock_func.return_value = "mocked"
        assert True
'''


class TestQATestAgent:
    """Test suite for enhanced QA Test Agent."""
    
    def test_qa_agent_initialization(self, qa_agent):
        """Test QA Test Agent initialization."""
        assert qa_agent.agent_id == "qa_test_agent"
        assert qa_agent.goose_path == "/usr/local/bin/goose"
        assert qa_agent.coverage_threshold == 80
        assert "pytest" in qa_agent.test_frameworks
    
    def test_get_capabilities(self, qa_agent):
        """Test QA agent capabilities."""
        capabilities = qa_agent.get_capabilities()
        
        expected_capabilities = [
            "automated_testing",
            "test_generation",
            "coverage_analysis", 
            "quality_assessment",
            "performance_testing",
            "security_scanning",
            "test_execution",
            "test_validation",
            "goose_test_generation",
            "comprehensive_reporting",
            "test_optimization",
            "framework_support_pytest",
            "framework_support_unittest",
            "ci_cd_integration",
            "test_maintenance"
        ]
        
        for capability in expected_capabilities:
            assert capability in capabilities
    
    @pytest.mark.asyncio
    async def test_run_tests_comprehensive(self, qa_agent):
        """Test comprehensive test execution."""
        task = {
            "test_path": "tests/",
            "framework": "pytest",
            "coverage": True
        }
        
        with patch.object(qa_agent, '_execute_test_suite') as mock_execute, \
             patch.object(qa_agent, '_detailed_coverage_analysis') as mock_coverage, \
             patch.object(qa_agent, '_analyze_test_performance') as mock_performance, \
             patch.object(qa_agent, '_assess_test_quality') as mock_quality:
            
            # Mock successful test execution
            mock_execute.return_value = {
                "success": True,
                "test_results": {"passed": 15, "failed": 1, "total": 16},
                "execution_time": 30.5
            }
            
            # Mock coverage analysis
            mock_coverage.return_value = {
                "success": True,
                "analysis": {
                    "total_coverage": 85.2,
                    "recommendations": ["Improve coverage for module.py"]
                }
            }
            
            # Mock performance analysis
            mock_performance.return_value = {
                "success": True,
                "performance": {
                    "total_execution_time": 30.5,
                    "performance_score": 85,
                    "recommendations": []
                }
            }
            
            # Mock quality assessment
            mock_quality.return_value = {
                "success": True,
                "quality": {"overall_score": 82}
            }
            
            result = await qa_agent._run_tests(task)
            
            assert result["success"] is True
            assert "test_results" in result
            assert "overall_score" in result
            
            # Verify all analysis methods were called
            mock_execute.assert_called_once()
            mock_coverage.assert_called_once()
            mock_performance.assert_called_once()
            mock_quality.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_tests_with_goose(self, qa_agent, sample_test_file):
        """Test test generation using Goose CLI."""
        task = {
            "target_file": "src/calculator.py",
            "test_type": "comprehensive",
            "coverage_target": 90
        }
        
        with patch('pathlib.Path.exists') as mock_exists, \
             patch.object(qa_agent, '_run_goose_command') as mock_goose:
            
            mock_exists.return_value = True
            mock_goose.return_value = {
                "success": True,
                "output": f"Generated comprehensive tests:\n```python\n{sample_test_file}\n```",
                "session_id": "test-session-123"
            }
            
            with patch('builtins.open', mock_open(read_data="def calculate(): return 42")):
                result = await qa_agent._generate_tests_with_goose("src/calculator.py", task)
                
                assert result["success"] is True
                assert result["method"] == "goose_cli_extracted"
                assert "test_file" in result
                
                # Verify Goose command was called with appropriate prompt
                mock_goose.assert_called_once()
                args, kwargs = mock_goose.call_args
                assert "session" in args[0]
                assert "start" in args[0]
                assert kwargs["input_text"] is not None
    
    def test_create_test_generation_prompt(self, qa_agent):
        """Test test generation prompt creation."""
        prompt = qa_agent._create_test_generation_prompt(
            "src/module.py", "comprehensive", 90
        )
        
        assert "comprehensive" in prompt
        assert "90%" in prompt
        assert "pytest" in prompt
        assert "edge case" in prompt
        assert "mock" in prompt
        assert "fixtures" in prompt
    
    @pytest.mark.asyncio
    async def test_validate_generated_tests(self, qa_agent, sample_test_file):
        """Test validation of generated tests."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(sample_test_file)
            f.flush()
            
            result = await qa_agent._validate_generated_tests(f.name, "src/test.py")
            
            assert result["syntax_valid"] is True
            assert result["test_count"] >= 3  # Sample has 3 test functions
            assert result["has_fixtures"] is True
            assert result["has_mocks"] is True
            assert result["quality_score"] > 60
            
            Path(f.name).unlink()  # Cleanup
    
    def test_extract_test_code_from_output(self, qa_agent, sample_test_file):
        """Test extraction of test code from Goose output."""
        goose_output = f"""
Here are the comprehensive tests for your module:

```python
{sample_test_file}
```

The tests cover all major functionality and edge cases.
"""
        
        extracted_code = qa_agent._extract_test_code_from_output(goose_output)
        
        assert extracted_code is not None
        assert "def test_sample_function" in extracted_code
        assert "pytest" in extracted_code
    
    @pytest.mark.asyncio
    async def test_detailed_coverage_analysis(self, qa_agent):
        """Test detailed coverage analysis."""
        coverage_data = {
            "totals": {"percent_covered": 85.5},
            "files": {
                "src/module.py": {
                    "summary": {"covered_lines": 17, "num_statements": 20},
                    "missing_lines": [15, 18, 22]
                },
                "src/utils.py": {
                    "summary": {"covered_lines": 8, "num_statements": 10},
                    "missing_lines": [5, 9]
                }
            }
        }
        
        with patch('pathlib.Path.exists') as mock_exists, \
             patch('builtins.open', mock_open(read_data=json.dumps(coverage_data))):
            
            mock_exists.return_value = True
            
            result = await qa_agent._detailed_coverage_analysis("tests/")
            
            assert result["success"] is True
            analysis = result["analysis"]
            assert analysis["total_coverage"] == 85.5
            assert len(analysis["files_coverage"]) == 2
            assert "recommendations" in analysis
    
    @pytest.mark.asyncio
    async def test_analyze_test_performance(self, qa_agent):
        """Test test performance analysis."""
        test_result = {
            "execution_time": 45.2,
            "test_results": {"tests_run": 20},
            "output": "test_slow::test_method [10s]\ntest_fast::test_other [0.1s]"
        }
        
        result = await qa_agent._analyze_test_performance(test_result)
        
        assert result["success"] is True
        performance = result["performance"]
        assert performance["total_execution_time"] == 45.2
        assert performance["average_test_time"] == 45.2 / 20
        assert len(performance["slow_tests"]) > 0
        assert performance["slow_tests"][0]["test"] == "test_slow::test_method"
    
    @pytest.mark.asyncio
    async def test_assess_test_quality(self, qa_agent, sample_test_file):
        """Test test quality assessment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test_example.py"
            test_file.write_text(sample_test_file)
            
            result = await qa_agent._assess_test_quality(temp_dir)
            
            assert result["success"] is True
            quality = result["quality"]
            assert "overall_score" in quality
            assert quality["overall_score"] > 0
            
            stats = result["statistics"]
            assert stats["total_test_files"] == 1
            assert stats["total_tests"] >= 3
    
    def test_generate_test_summary(self, qa_agent):
        """Test comprehensive test summary generation."""
        execution_results = {
            "test_execution": {
                "test_results": {"passed": 18, "failed": 2}
            },
            "coverage_analysis": {
                "analysis": {
                    "total_coverage": 82.5,
                    "recommendations": ["Improve module.py coverage"]
                }
            },
            "performance_metrics": {
                "performance": {
                    "total_execution_time": 35.0,
                    "performance_score": 85
                }
            },
            "quality_assessment": {
                "quality": {"overall_score": 78}
            }
        }
        
        summary = qa_agent._generate_test_summary(execution_results)
        
        assert summary["overall_score"] > 0
        assert summary["test_success_rate"] == 90.0  # 18/20 * 100
        assert summary["coverage_percentage"] == 82.5
        assert summary["status"] in ["EXCELLENT", "GOOD", "NEEDS_IMPROVEMENT", "POOR"]
        assert len(summary["recommendations"]) > 0
    
    @pytest.mark.asyncio
    async def test_execute_test_suite_pytest(self, qa_agent):
        """Test pytest execution."""
        with patch.object(qa_agent, '_run_command') as mock_run, \
             patch.object(qa_agent, '_parse_test_output') as mock_parse:
            
            mock_run.return_value = {
                "success": True,
                "output": "20 passed, 0 failed",
                "return_code": 0
            }
            
            mock_parse.return_value = {
                "passed": 20,
                "failed": 0,
                "tests_run": 20
            }
            
            result = await qa_agent._execute_test_suite("tests/", "pytest", "test_*.py", True)
            
            assert result["success"] is True
            assert result["test_results"]["passed"] == 20
            
            # Verify pytest command with coverage
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert "pytest" in cmd
            assert "--cov=src" in cmd
    
    @pytest.mark.asyncio
    async def test_run_goose_command(self, qa_agent):
        """Test Goose CLI command execution."""
        with patch('subprocess.run') as mock_run:
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_process.stdout = "Goose session started successfully"
            mock_process.stderr = ""
            mock_run.return_value = mock_process
            
            result = await qa_agent._run_goose_command(
                ["session", "start"], 
                input_text="Generate tests for module.py"
            )
            
            assert result["success"] is True
            assert "Goose session started" in result["output"]
            assert result["return_code"] == 0
            
            # Verify command construction
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert call_args[0][0][0].endswith("goose")
            assert "session" in call_args[0][0]
    
    @pytest.mark.asyncio
    async def test_goose_command_not_found(self, qa_agent):
        """Test behavior when Goose CLI is not found."""
        qa_agent.goose_path = None
        
        result = await qa_agent._run_goose_command(["session", "start"])
        
        assert result["success"] is False
        assert "Goose CLI not found" in result["error"]
    
    @pytest.mark.asyncio
    async def test_execute_task_types(self, qa_agent):
        """Test different QA task types."""
        with patch.object(qa_agent, '_run_tests') as mock_run_tests, \
             patch.object(qa_agent, '_generate_tests') as mock_gen_tests, \
             patch.object(qa_agent, '_analyze_coverage') as mock_coverage, \
             patch.object(qa_agent, '_quality_check') as mock_quality:
            
            # Mock all methods to return success
            mock_run_tests.return_value = {"success": True}
            mock_gen_tests.return_value = {"success": True}
            mock_coverage.return_value = {"success": True}
            mock_quality.return_value = {"success": True}
            
            # Test run_tests task
            result = await qa_agent.execute_task({"type": "run_tests"})
            mock_run_tests.assert_called_once()
            
            # Test generate_tests task
            result = await qa_agent.execute_task({"type": "generate_tests"})
            mock_gen_tests.assert_called_once()
            
            # Test analyze_coverage task
            result = await qa_agent.execute_task({"type": "analyze_coverage"})
            mock_coverage.assert_called_once()
            
            # Test quality_check task
            result = await qa_agent.execute_task({"type": "quality_check"})
            mock_quality.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
