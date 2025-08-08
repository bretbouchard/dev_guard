"""Tests for Task 11.2 TDD Support in QA Test Agent."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

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
def tdd_qa_agent(mock_config, mock_shared_memory, mock_vector_db):
    """Create a QA Test Agent instance for TDD testing."""
    with patch('shutil.which') as mock_which:
        mock_which.return_value = "/usr/local/bin/goose"
        
        agent = QATestAgent(
            agent_id="tdd_qa_agent",
            config=mock_config,
            shared_memory=mock_shared_memory,
            vector_db=mock_vector_db
        )
        return agent


@pytest.fixture
def sample_python_file():
    """Create a sample Python file for testing."""
    return '''
def calculator(a, b, operation):
    """Simple calculator function."""
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("Division by zero")
        return a / b
    else:
        raise ValueError("Unknown operation")
'''


class TestTDDSupport:
    """Test suite for TDD support in QA Test Agent."""
    
    def test_tdd_capabilities_added(self, tdd_qa_agent):
        """Test that TDD capabilities are properly added."""
        capabilities = tdd_qa_agent.get_capabilities()
        
        tdd_capabilities = [
            "tdd_support",
            "test_driven_development",
            "red_green_refactor",
            "behavior_driven_development",
            "test_templates",
            "advanced_test_patterns"
        ]
        
        for capability in tdd_capabilities:
            assert capability in capabilities, f"Missing TDD capability: {capability}"
    
    def test_tdd_initialization(self, tdd_qa_agent):
        """Test TDD-specific initialization."""
        assert tdd_qa_agent.tdd_enabled is True
        assert tdd_qa_agent.tdd_cycle_state == "red"
        assert "unit" in tdd_qa_agent.test_patterns
        assert "pytest" in tdd_qa_agent.test_templates
        assert "unittest" in tdd_qa_agent.test_templates
        assert "bdd" in tdd_qa_agent.test_templates
    
    def test_pytest_template_generation(self, tdd_qa_agent):
        """Test pytest template generation."""
        template = tdd_qa_agent._get_pytest_template()
        
        # Verify template contains key pytest elements
        assert "import pytest" in template
        assert "@pytest.fixture" in template
        assert "@pytest.mark.parametrize" in template
        assert "def test_" in template
        assert "assert" in template
        assert "with patch(" in template
    
    def test_unittest_template_generation(self, tdd_qa_agent):
        """Test unittest template generation."""
        template = tdd_qa_agent._get_unittest_template()
        
        # Verify template contains key unittest elements
        assert "import unittest" in template
        assert "class Test" in template
        assert "def setUp(self)" in template
        assert "def tearDown(self)" in template
        assert "self.assertEqual" in template
        assert "self.assertRaises" in template
    
    def test_bdd_template_generation(self, tdd_qa_agent):
        """Test BDD template generation."""
        template = tdd_qa_agent._get_bdd_template()
        
        # Verify template contains key BDD elements
        assert "from pytest_bdd import" in template
        assert "@given" in template
        assert "@when" in template
        assert "@then" in template
        assert "scenarios(" in template
    
    @pytest.mark.asyncio
    async def test_tdd_cycle_execution(self, tdd_qa_agent):
        """Test complete TDD cycle execution."""
        task = {
            "type": "tdd_cycle",
            "target_file": "src/calculator.py",
            "requirements": "Implement basic arithmetic operations",
            "test_type": "unit"
        }
        
        with patch.object(tdd_qa_agent, '_tdd_red_phase') as mock_red, \
             patch.object(tdd_qa_agent, '_tdd_green_phase') as mock_green, \
             patch.object(tdd_qa_agent, '_tdd_refactor_phase') as mock_refactor:
            
            # Mock successful phases
            mock_red.return_value = {
                "success": True,
                "test_file": "test_calculator.py",
                "phase": "red"
            }
            
            mock_green.return_value = {
                "success": True,
                "test_results": {"passed": 0, "failed": 0},
                "phase": "green"
            }
            
            mock_refactor.return_value = {
                "success": True,
                "quality_improved": True,
                "phase": "refactor"
            }
            
            result = await tdd_qa_agent._run_tdd_cycle(task)
            
            assert result["success"] is True
            assert result["cycle_completed"] is True
            assert result["final_state"] == "refactor"
            assert "phases" in result
            assert "red" in result["phases"]
            assert "green" in result["phases"]
            assert "refactor" in result["phases"]
    
    @pytest.mark.asyncio
    async def test_tdd_red_phase(self, tdd_qa_agent, sample_python_file):
        """Test TDD Red phase (failing test creation)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            target_file = Path(temp_dir) / "calculator.py"
            target_file.write_text(sample_python_file)
            
            with patch.object(tdd_qa_agent, '_generate_tests_with_goose') as mock_generate, \
                 patch.object(tdd_qa_agent, '_execute_test_suite') as mock_execute:
                
                # Mock test generation
                mock_generate.return_value = {
                    "success": True,
                    "test_file": str(temp_dir / "test_calculator.py")
                }
                
                # Mock failing test execution (expected in RED phase)
                mock_execute.return_value = {
                    "test_results": {
                        "passed": 0,
                        "failed": 3,
                        "errors": 0
                    }
                }
                
                result = await tdd_qa_agent._tdd_red_phase(
                    str(target_file),
                    "Create failing tests for calculator",
                    "unit"
                )
                
                assert result["success"] is True  # Success means test failed as expected
                assert result["phase"] == "red"
                assert "test_file" in result
    
    @pytest.mark.asyncio
    async def test_tdd_green_phase(self, tdd_qa_agent):
        """Test TDD Green phase (minimal implementation)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            target_file = Path(temp_dir) / "calculator.py"
            test_file = Path(temp_dir) / "test_calculator.py"
            
            test_content = '''
import pytest
from calculator import add

def test_add_basic():
    assert add(2, 3) == 5

def test_add_negative():
    assert add(-1, 1) == 0
'''
            test_file.write_text(test_content)
            
            with patch.object(tdd_qa_agent, '_run_goose_command') as mock_goose, \
                 patch.object(tdd_qa_agent, '_execute_test_suite') as mock_execute:
                
                mock_goose.return_value = {
                    "success": True,
                    "output": "Implementation generated"
                }
                
                # Mock passing tests (goal of GREEN phase)
                mock_execute.return_value = {
                    "test_results": {
                        "passed": 2,
                        "failed": 0,
                        "errors": 0
                    }
                }
                
                result = await tdd_qa_agent._tdd_green_phase(
                    str(target_file),
                    str(test_file)
                )
                
                assert result["success"] is True
                assert result["phase"] == "green"
                assert result["test_results"]["failed"] == 0
    
    @pytest.mark.asyncio
    async def test_tdd_refactor_phase(self, tdd_qa_agent):
        """Test TDD Refactor phase (quality improvement)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            target_file = Path(temp_dir) / "calculator.py"
            test_file = Path(temp_dir) / "test_calculator.py"
            
            with patch.object(tdd_qa_agent, '_assess_test_quality') as mock_quality, \
                 patch.object(tdd_qa_agent, '_run_goose_command') as mock_goose, \
                 patch.object(tdd_qa_agent, '_execute_test_suite') as mock_execute:
                
                # Mock quality assessment
                mock_quality.side_effect = [
                    {"quality": {"overall_score": 70}},  # Before refactoring
                    {"quality": {"overall_score": 85}}   # After refactoring
                ]
                
                mock_goose.return_value = {
                    "success": True,
                    "output": "Refactoring completed"
                }
                
                mock_execute.return_value = {
                    "test_results": {
                        "passed": 2,
                        "failed": 0,
                        "errors": 0
                    }
                }
                
                result = await tdd_qa_agent._tdd_refactor_phase(
                    str(target_file),
                    str(test_file)
                )
                
                assert result["success"] is True
                assert result["phase"] == "refactor"
                assert result["quality_improved"] is True
                assert result["quality_after"] > result["quality_before"]
    
    @pytest.mark.asyncio
    async def test_minimal_implementation_generation(self, tdd_qa_agent):
        """Test minimal implementation generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            target_file = Path(temp_dir) / "calculator.py"
            test_file = Path(temp_dir) / "test_calculator.py"
            
            test_content = '''
import pytest
from calculator import Calculator

class TestCalculator:
    def test_add_basic_functionality(self):
        calc = Calculator()
        assert calc.add(2, 3) == 5
    
    def test_subtract_basic_functionality(self):
        calc = Calculator()
        assert calc.subtract(5, 3) == 2
'''
            test_file.write_text(test_content)
            
            result = await tdd_qa_agent._generate_minimal_implementation(
                str(target_file),
                str(test_file)
            )
            
            assert result["success"] is True
            assert result["phase"] == "green"
            assert target_file.exists()
            
            # Check that minimal implementation was created
            implementation = target_file.read_text()
            assert "class Calculator" in implementation
            assert "def add" in implementation
            assert "def subtract" in implementation
    
    @pytest.mark.asyncio
    async def test_bdd_test_generation(self, tdd_qa_agent):
        """Test BDD test generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            target_file = Path(temp_dir) / "calculator.py"
            
            task = {
                "target_file": str(target_file),
                "feature_description": "Calculator functionality for basic arithmetic",
                "user_stories": [
                    "perform addition operations",
                    "perform subtraction operations",
                    "handle division by zero errors"
                ]
            }
            
            result = await tdd_qa_agent._generate_behavior_driven_tests(task)
            
            assert result["success"] is True
            assert result["method"] == "bdd_generated"
            assert "feature_file" in result
            assert "test_file" in result
            
            # Check that files were created
            feature_file = Path(result["feature_file"])
            test_file = Path(result["test_file"])
            
            assert feature_file.exists()
            assert test_file.exists()
            
            # Check feature file content
            feature_content = feature_file.read_text()
            assert "Feature: Calculator" in feature_content
            assert "Scenario:" in feature_content
            assert "Given I have access to" in feature_content
    
    def test_feature_file_creation(self, tdd_qa_agent):
        """Test Gherkin feature file creation."""
        feature_content = tdd_qa_agent._create_feature_file(
            "calculator.py",
            "Basic arithmetic calculator functionality",
            ["add two numbers", "subtract numbers", "multiply values"]
        )
        
        assert "Feature: Calculator" in feature_content
        assert "Basic arithmetic calculator functionality" in feature_content
        assert "Scenario: User Story 1" in feature_content
        assert "Scenario: User Story 2" in feature_content
        assert "Scenario: User Story 3" in feature_content
        assert "Given I have access to the Calculator functionality" in feature_content
        assert "When I add two numbers" in feature_content
    
    def test_step_definitions_creation(self, tdd_qa_agent):
        """Test BDD step definitions creation."""
        step_definitions = tdd_qa_agent._create_step_definitions(
            "calculator.py",
            ["perform calculations"]
        )
        
        assert "from pytest_bdd import" in step_definitions
        assert "@given" in step_definitions
        assert "@when" in step_definitions
        assert "@then" in step_definitions
        assert "Calculator" in step_definitions
    
    @pytest.mark.asyncio
    async def test_tdd_task_routing(self, tdd_qa_agent):
        """Test that TDD tasks are properly routed."""
        tasks = [
            {"type": "tdd_cycle"},
            {"type": "tdd_red"},
            {"type": "tdd_green"},
            {"type": "tdd_refactor"},
            {"type": "generate_bdd_tests"}
        ]
        
        for task in tasks:
            with patch.object(tdd_qa_agent, f'_{task["type"].replace("generate_", "generate_")}') as mock_method:
                if task["type"] == "tdd_cycle":
                    mock_method = patch.object(tdd_qa_agent, '_run_tdd_cycle')
                elif task["type"] == "generate_bdd_tests":
                    mock_method = patch.object(tdd_qa_agent, '_generate_behavior_driven_tests')
                elif task["type"] == "tdd_red":
                    mock_method = patch.object(tdd_qa_agent, '_tdd_red_phase')
                elif task["type"] == "tdd_green":
                    mock_method = patch.object(tdd_qa_agent, '_tdd_green_phase')
                elif task["type"] == "tdd_refactor":
                    mock_method = patch.object(tdd_qa_agent, '_tdd_refactor_phase')
                
                with mock_method as mock_handler:
                    mock_handler.return_value = {"success": True}
                    
                    result = await tdd_qa_agent.execute_task(task)
                    
                    # Verify the correct handler was called
                    mock_handler.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
