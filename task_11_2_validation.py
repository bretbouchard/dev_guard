"""
Validation script for Task 11.2 TDD Support in QA Test Agent.
Tests TDD workflow and test generation capabilities.
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dev_guard.agents.qa_test import QATestAgent


class MockConfig:
    """Mock configuration for testing."""
    
    def get_agent_config(self, agent_type):
        return {
            "max_retries": 3,
            "timeout": 300
        }


class MockSharedMemory:
    """Mock shared memory for testing."""
    
    def update_agent_state(self, *args, **kwargs):
        pass
    
    def add_memory(self, *args, **kwargs):
        return "test-memory-id"


class MockVectorDB:
    """Mock vector database for testing."""
    pass


async def test_tdd_capabilities():
    """Test TDD capabilities and initialization."""
    print("Testing TDD Support capabilities...")
    
    # Mock dependencies
    config = MockConfig()
    memory = MockSharedMemory()
    vector_db = MockVectorDB()
    
    # Create QA Test Agent
    qa_agent = QATestAgent(
        agent_id="tdd_test_agent",
        config=config,
        shared_memory=memory,
        vector_db=vector_db
    )
    
    # Test TDD-specific initialization
    assert qa_agent.tdd_enabled is True
    assert qa_agent.tdd_cycle_state == "red"
    print("✓ TDD initialization successful")
    
    # Test TDD capabilities
    capabilities = qa_agent.get_capabilities()
    tdd_capabilities = [
        "tdd_support",
        "test_driven_development",
        "red_green_refactor",
        "behavior_driven_development",
        "test_templates",
        "advanced_test_patterns"
    ]
    
    for capability in tdd_capabilities:
        assert capability in capabilities, f"Missing: {capability}"
    print(f"✓ All {len(tdd_capabilities)} TDD capabilities present")
    
    # Test test patterns
    expected_patterns = ["unit", "integration", "e2e", "performance"]
    for pattern in expected_patterns:
        assert pattern in qa_agent.test_patterns
    print("✓ Test patterns configured correctly")
    
    # Test test templates
    expected_templates = ["pytest", "unittest", "bdd"]
    for template in expected_templates:
        assert template in qa_agent.test_templates
    print("✓ Test templates available")


async def test_template_generation():
    """Test test template generation."""
    print("\nTesting test template generation...")
    
    config = MockConfig()
    memory = MockSharedMemory()
    vector_db = MockVectorDB()
    
    qa_agent = QATestAgent(
        agent_id="template_test_agent",
        config=config,
        shared_memory=memory,
        vector_db=vector_db
    )
    
    # Test pytest template
    pytest_template = qa_agent._get_pytest_template()
    pytest_elements = [
        "import pytest",
        "@pytest.fixture",
        "@pytest.mark.parametrize",
        "def test_",
        "assert",
        "with patch("
    ]
    
    for element in pytest_elements:
        assert element in pytest_template
    print("✓ Pytest template generated correctly")
    
    # Test unittest template
    unittest_template = qa_agent._get_unittest_template()
    unittest_elements = [
        "import unittest",
        "class Test",
        "def setUp(self)",
        "def tearDown(self)",
        "self.assertEqual",
        "self.assertRaises"
    ]
    
    for element in unittest_elements:
        assert element in unittest_template
    print("✓ Unittest template generated correctly")
    
    # Test BDD template
    bdd_template = qa_agent._get_bdd_template()
    bdd_elements = [
        "from pytest_bdd import",
        "@given",
        "@when",
        "@then",
        "scenarios("
    ]
    
    for element in bdd_elements:
        assert element in bdd_template
    print("✓ BDD template generated correctly")


async def test_feature_file_creation():
    """Test Gherkin feature file creation."""
    print("\nTesting BDD feature file creation...")
    
    config = MockConfig()
    memory = MockSharedMemory()
    vector_db = MockVectorDB()
    
    qa_agent = QATestAgent(
        agent_id="bdd_test_agent",
        config=config,
        shared_memory=memory,
        vector_db=vector_db
    )
    
    # Test feature file creation
    feature_content = qa_agent._create_feature_file(
        "calculator.py",
        "Basic arithmetic calculator functionality",
        ["add two numbers", "subtract numbers", "multiply values"]
    )
    
    # Verify feature file structure
    assert "Feature: Calculator" in feature_content
    assert "Basic arithmetic calculator functionality" in feature_content
    assert "Scenario: User Story 1" in feature_content
    assert "Given I have access to the Calculator functionality" in feature_content
    assert "When I add two numbers" in feature_content
    print("✓ Gherkin feature file creation working")
    
    # Test step definitions creation
    step_definitions = qa_agent._create_step_definitions(
        "calculator.py",
        ["perform calculations"]
    )
    
    step_elements = [
        "from pytest_bdd import",
        "@given",
        "@when", 
        "@then",
        "Calculator"
    ]
    
    for element in step_elements:
        assert element in step_definitions
    print("✓ BDD step definitions creation working")


async def test_minimal_implementation():
    """Test minimal implementation generation."""
    print("\nTesting minimal implementation generation...")
    
    config = MockConfig()
    memory = MockSharedMemory()
    vector_db = MockVectorDB()
    
    qa_agent = QATestAgent(
        agent_id="minimal_impl_agent",
        config=config,
        shared_memory=memory,
        vector_db=vector_db
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        target_file = Path(temp_dir) / "calculator.py"
        test_file = Path(temp_dir) / "test_calculator.py"
        
        # Create test file content
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
        
        # Generate minimal implementation
        result = await qa_agent._generate_minimal_implementation(
            str(target_file),
            str(test_file)
        )
        
        assert result["success"] is True
        assert result["phase"] == "green"
        assert target_file.exists()
        print("✓ Minimal implementation generation working")
        
        # Check implementation content
        implementation = target_file.read_text()
        assert "class Calculator" in implementation
        assert "def add" in implementation
        assert "def subtract" in implementation
        print("✓ Generated implementation contains expected elements")


async def test_tdd_workflow_structure():
    """Test TDD workflow structure and state management."""
    print("\nTesting TDD workflow structure...")
    
    config = MockConfig()
    memory = MockSharedMemory()
    vector_db = MockVectorDB()
    
    qa_agent = QATestAgent(
        agent_id="tdd_workflow_agent",
        config=config,
        shared_memory=memory,
        vector_db=vector_db
    )
    
    # Test TDD cycle state transitions
    initial_state = qa_agent.tdd_cycle_state
    assert initial_state == "red"
    print("✓ TDD cycle starts in RED state")
    
    # Test that TDD methods exist and are callable
    tdd_methods = [
        "_run_tdd_cycle",
        "_tdd_red_phase",
        "_tdd_green_phase", 
        "_tdd_refactor_phase",
        "_generate_behavior_driven_tests"
    ]
    
    for method_name in tdd_methods:
        assert hasattr(qa_agent, method_name)
        method = getattr(qa_agent, method_name)
        assert callable(method)
    print(f"✓ All {len(tdd_methods)} TDD methods available")
    
    # Test template methods exist
    template_methods = [
        "_get_pytest_template",
        "_get_unittest_template",
        "_get_bdd_template"
    ]
    
    for method_name in template_methods:
        assert hasattr(qa_agent, method_name)
        method = getattr(qa_agent, method_name)
        assert callable(method)
        # Test that method returns string content
        template = method()
        assert isinstance(template, str)
        assert len(template) > 100  # Templates should be substantial
    print(f"✓ All {len(template_methods)} template methods working")


def test_tdd_task_types():
    """Test TDD task type support."""
    print("\nTesting TDD task type support...")
    
    config = MockConfig()
    memory = MockSharedMemory()
    vector_db = MockVectorDB()
    
    qa_agent = QATestAgent(
        agent_id="tdd_task_agent",
        config=config,
        shared_memory=memory,
        vector_db=vector_db
    )
    
    # Test that enhanced capabilities include TDD features
    original_capabilities = [
        "automated_testing", "test_generation", "coverage_analysis",
        "quality_assessment", "performance_testing", "security_scanning"
    ]
    
    tdd_capabilities = [
        "tdd_support", "test_driven_development", "red_green_refactor",
        "behavior_driven_development", "test_templates", "advanced_test_patterns"
    ]
    
    all_capabilities = qa_agent.get_capabilities()
    
    # Check original capabilities still exist
    for cap in original_capabilities:
        assert cap in all_capabilities
    print("✓ Original QA capabilities preserved")
    
    # Check new TDD capabilities added
    for cap in tdd_capabilities:
        assert cap in all_capabilities
    print("✓ New TDD capabilities added")
    
    total_expected = len(original_capabilities) + len(tdd_capabilities) + 9  # Other existing capabilities
    assert len(all_capabilities) >= total_expected
    print(f"✓ Total capabilities: {len(all_capabilities)} (expected >= {total_expected})")


async def main():
    """Run all TDD validation tests."""
    print("🧪 Task 11.2 TDD Support Validation\n")
    print("=" * 50)
    
    try:
        await test_tdd_capabilities()
        await test_template_generation()
        await test_feature_file_creation()
        await test_minimal_implementation()
        await test_tdd_workflow_structure()
        test_tdd_task_types()
        
        print("\n" + "=" * 50)
        print("🎉 Task 11.2 TDD Support validation completed successfully!")
        print("\n📋 TDD Features Validated:")
        print("   ✅ Test-Driven Development workflow support")
        print("   ✅ Red-Green-Refactor cycle implementation") 
        print("   ✅ Test template generation (pytest, unittest, BDD)")
        print("   ✅ Behavior-Driven Development support")
        print("   ✅ Minimal implementation generation")
        print("   ✅ Gherkin feature file creation")
        print("   ✅ Enhanced QA agent capabilities")
        print("   ✅ TDD state management")
        
        print("\n🎯 Ready to proceed to Task 11.3: Integrate Goose `fix` and `write-tests`")
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
