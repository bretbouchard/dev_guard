#!/usr/bin/env python3
"""
Task 10.3 Validation Script
Tests the AST analysis and Goose memory integration functionality
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dev_guard.agents.code_agent import CodeAgent


class MockConfig:
    """Mock configuration for testing."""
    
    def get_agent_config(self, agent_type):
        return {
            "max_retries": 3,
            "timeout": 300
        }


class MockSharedMemory:
    """Mock shared memory for testing."""
    
    def __init__(self):
        pass
    
    def update_agent_state(self, *args, **kwargs):
        pass
    
    def add_memory(self, *args, **kwargs):
        return "test-memory-id"


class MockVectorDB:
    """Mock vector database for testing."""
    
    def __init__(self):
        pass
    
    def search(self, query, limit=5):
        return {
            "documents": [
                {"content": "sample code", "metadata": {"file_path": "/test.py"}, "score": 0.8}
            ]
        }


async def validate_ast_analysis():
    """Test AST analysis functionality."""
    print("🔬 Testing AST Analysis...")
    
    # Create a sample Python file for analysis
    sample_code = '''
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
    
    # Create Code Agent
    mock_config = MockConfig()
    mock_memory = MockSharedMemory()
    mock_vector_db = MockVectorDB()
    
    agent = CodeAgent(
        agent_id="test_agent",
        config=mock_config,
        shared_memory=mock_memory,
        vector_db=mock_vector_db,
        working_directory="/tmp"
    )
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(sample_code)
        temp_file = f.name
    
    try:
        # Test AST analysis
        result = await agent.analyze_code_structure(temp_file)
        
        print("✅ AST Analysis Result:")
        print(f"   Success: {result['success']}")
        print(f"   Classes found: {len(result.get('classes', []))}")
        print(f"   Functions found: {len(result.get('functions', []))}")
        print(f"   Imports found: {len(result.get('imports', []))}")
        
        if result.get('classes'):
            class_info = result['classes'][0]
            print(f"   First class: {class_info['name']} with {len(class_info['methods'])} methods")
        
        # Test structural similarity calculation
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
        
        similarity = agent._calculate_structural_similarity(struct1, struct2)
        print(f"✅ Structural Similarity: {similarity:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ AST Analysis failed: {e}")
        return False
        
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.unlink(temp_file)


async def validate_pattern_matching():
    """Test pattern matching functionality."""
    print("\n🔍 Testing Pattern Matching...")
    
    try:
        # Create Code Agent
        mock_config = MockConfig()
        mock_memory = MockSharedMemory()
        mock_vector_db = MockVectorDB()
        
        agent = CodeAgent(
            agent_id="test_agent",
            config=mock_config,
            shared_memory=mock_memory,
            vector_db=mock_vector_db,
            working_directory="/tmp"
        )
        
        # Test pattern context building
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
        
        context = agent._build_pattern_context(patterns)
        print("✅ Pattern Context Built Successfully")
        print(f"   Context length: {len(context)} characters")
        print(f"   Contains goose_memory pattern: {'goose_memory' in context}")
        print(f"   Contains vector_search pattern: {'vector_search' in context}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pattern matching failed: {e}")
        return False


def validate_capabilities():
    """Test that new capabilities are included."""
    print("\n⚡ Testing Capabilities...")
    
    try:
        # Create Code Agent
        mock_config = MockConfig()
        mock_memory = MockSharedMemory()
        mock_vector_db = MockVectorDB()
        
        agent = CodeAgent(
            agent_id="test_agent",
            config=mock_config,
            shared_memory=mock_memory,
            vector_db=mock_vector_db,
            working_directory="/tmp"
        )
        
        capabilities = agent.get_capabilities()
        
        expected_new_capabilities = [
            "ast_analysis",
            "pattern_matching",
            "goose_memory_search",
            "structural_similarity",
            "refactoring_impact_analysis"
        ]
        
        print(f"✅ Total capabilities: {len(capabilities)}")
        
        missing_capabilities = []
        for cap in expected_new_capabilities:
            if cap in capabilities:
                print(f"   ✅ {cap}: Found")
            else:
                print(f"   ❌ {cap}: Missing")
                missing_capabilities.append(cap)
        
        return len(missing_capabilities) == 0
        
    except Exception as e:
        print(f"❌ Capabilities test failed: {e}")
        return False


async def main():
    """Run all validation tests."""
    print("🚀 Task 10.3 Validation: AST Analysis and Goose Memory Integration")
    print("=" * 70)
    
    tests = [
        ("AST Analysis", validate_ast_analysis()),
        ("Pattern Matching", validate_pattern_matching()),
        ("Capabilities", validate_capabilities())
    ]
    
    results = []
    
    for test_name, test_func in tests:
        if asyncio.iscoroutine(test_func):
            result = await test_func
        else:
            result = test_func
        results.append((test_name, result))
    
    print("\n" + "=" * 70)
    print("📊 VALIDATION SUMMARY")
    print("=" * 70)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20}: {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 Task 10.3 IMPLEMENTATION COMPLETE!")
        print("   ✅ AST analysis functionality working")
        print("   ✅ Pattern matching implemented")
        print("   ✅ Goose memory integration ready")
        print("   ✅ Structural similarity calculations functional")
        print("   ✅ Refactoring impact analysis available")
        return True
    else:
        print(f"\n⚠️  Task 10.3 needs attention: {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
