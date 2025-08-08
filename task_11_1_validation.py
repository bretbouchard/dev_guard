"""
Validation script for Task 11.1 Enhanced QA Test Agent.
Tests core functionality and integration capabilities.
"""

import asyncio
import sys
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


async def test_qa_agent_capabilities():
    """Test QA Test Agent capabilities and initialization."""
    print("Testing QA Test Agent initialization and capabilities...")
    
    # Mock dependencies
    config = MockConfig()
    memory = MockSharedMemory()
    vector_db = MockVectorDB()
    
    # Create QA Test Agent
    qa_agent = QATestAgent(
        agent_id="test_qa_agent",
        config=config,
        shared_memory=memory,
        vector_db=vector_db
    )
    
    # Test basic initialization
    assert qa_agent.agent_id == "test_qa_agent"
    print("✓ QA agent initialized successfully")
    
    # Test capabilities
    capabilities = qa_agent.get_capabilities()
    expected_capabilities = [
        "automated_testing",
        "test_generation",
        "coverage_analysis",
        "performance_testing",
        "security_scanning",
        "code_quality_check",
        "lint_checking",
        "style_checking"
    ]
    
    for cap in expected_capabilities:
        assert cap in capabilities, f"Missing capability: {cap}"
    print(f"✓ All {len(expected_capabilities)} expected capabilities present")
    
    # Test test generation prompt creation
    prompt = qa_agent._create_test_generation_prompt(
        "src/calculator.py", "comprehensive", 90
    )
    
    # Verify prompt contains key elements
    assert "comprehensive" in prompt
    assert "90%" in prompt
    assert "pytest" in prompt
    print("✓ Test generation prompt created successfully")
    
    # Test test code extraction
    sample_output = """
Here are the tests:

```python
import pytest

def test_add():
    assert 2 + 2 == 4

def test_subtract():
    assert 5 - 3 == 2
```

These tests cover the basic functionality.
"""
    
    extracted_code = qa_agent._extract_test_code_from_output(sample_output)
    assert extracted_code is not None
    assert "def test_add" in extracted_code
    print("✓ Test code extraction working correctly")
    
    # Test summary generation
    mock_results = {
        "test_execution": {
            "test_results": {"passed": 18, "failed": 2, "total": 20}
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
    
    summary = qa_agent._generate_test_summary(mock_results)
    assert summary["overall_score"] > 0
    assert summary["test_success_rate"] == 90.0  # 18/20 * 100
    assert summary["status"] in ["EXCELLENT", "GOOD", "NEEDS_IMPROVEMENT", "POOR"]
    print("✓ Test summary generation working correctly")
    
    print("\n🎉 Task 11.1 Enhanced QA Test Agent validation completed successfully!")
    print(f"   Agent ID: {qa_agent.agent_id}")
    print(f"   Capabilities: {len(capabilities)} features")
    print(f"   Goose integration: {'✓' if qa_agent.goose_path else '✗'}")
    print(f"   Test frameworks supported: {len(qa_agent.test_frameworks)}")


async def test_goose_integration():
    """Test Goose CLI integration capabilities."""
    print("\nTesting Goose CLI integration...")
    
    config = MockConfig()
    memory = MockSharedMemory()
    vector_db = MockVectorDB()
    
    qa_agent = QATestAgent(
        agent_id="goose_test_agent",
        config=config,
        shared_memory=memory,
        vector_db=vector_db
    )
    
    # Test Goose executable detection
    goose_path = qa_agent._find_goose_executable()
    print(f"   Goose path detected: {goose_path or 'Not found'}")
    
    # Test Goose command without actually running it
    if not qa_agent.goose_path:
        result = await qa_agent._run_goose_command(["session", "start"])
        assert result["success"] is False
        assert "not found" in result["error"].lower()
        print("✓ Goose not found handled correctly")
    else:
        print("✓ Goose CLI detected and available")
    
    print("✓ Goose integration test completed")


if __name__ == "__main__":
    asyncio.run(test_qa_agent_capabilities())
    asyncio.run(test_goose_integration())
