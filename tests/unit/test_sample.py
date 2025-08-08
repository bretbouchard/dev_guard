"""
Sample unit tests demonstrating the testing infrastructure.
These tests validate the testing framework setup and provide examples.
"""

from datetime import datetime

import pytest

from tests.utils.assertions import (
    assert_agent_state_valid,
    assert_llm_response_valid,
    assert_memory_entry_valid,
    assert_task_status_valid,
)
from tests.utils.helpers import (
    MockLLMProvider,
    TestTimer,
    generate_random_string,
    generate_test_code,
)


class TestTestingInfrastructure:
    """Test the testing infrastructure itself"""
    
    def test_fixtures_available(self, test_config, mock_git_repo, mock_llm_responses):
        """Test that all fixtures are properly available"""
        assert test_config is not None
        assert "database" in test_config
        assert "llm" in test_config
        
        assert mock_git_repo is not None
        assert "path" in mock_git_repo
        assert "repo" in mock_git_repo
        
        assert mock_llm_responses is not None
        assert "code_generation" in mock_llm_responses
    
    def test_memory_entry_factory(self):
        """Test memory entry factory and validation"""
        from tests.conftest import MemoryEntryFactory
        
        entry = MemoryEntryFactory()
        entry_dict = {
            "id": entry.id,
            "agent_id": entry.agent_id,
            "timestamp": entry.timestamp,
            "type": entry.type,
            "content": entry.content
        }
        
        assert_memory_entry_valid(entry_dict)
    
    def test_task_status_factory(self):
        """Test task status factory and validation"""
        from tests.conftest import TaskStatusFactory
        
        task = TaskStatusFactory()
        task_dict = {
            "id": task.id,
            "agent_id": task.agent_id,
            "status": task.status,
            "description": task.description,
            "created_at": task.created_at,
            "updated_at": task.updated_at
        }
        
        assert_task_status_valid(task_dict)
    
    def test_agent_state_factory(self):
        """Test agent state factory and validation"""
        from tests.conftest import AgentStateFactory
        
        state = AgentStateFactory()
        state_dict = {
            "agent_id": state.agent_id,
            "status": state.status,
            "last_heartbeat": state.last_heartbeat
        }
        
        assert_agent_state_valid(state_dict)
    
    @pytest.mark.asyncio
    async def test_mock_shared_memory(self, mock_shared_memory):
        """Test mock shared memory functionality"""
        from tests.conftest import TestMemoryEntry
        
        # Create test memory entry
        entry = TestMemoryEntry(
            id="test_id",
            agent_id="test_agent",
            timestamp=datetime.now(),
            type="observation",
            content={"test": "data"}
        )
        
        # Test adding memory
        entry_id = await mock_shared_memory.add_memory(entry)
        assert entry_id == "test_id"
        
        # Test retrieving memories
        memories = await mock_shared_memory.get_memories("test_agent")
        assert len(memories) == 1
        assert memories[0].id == "test_id"
    
    @pytest.mark.asyncio
    async def test_mock_vector_db(self, mock_vector_db, temp_dir):
        """Test mock vector database functionality"""
        
        test_file = temp_dir / "test.py"
        test_content = "def hello(): return 'world'"
        
        # Test adding file content
        doc_ids = await mock_vector_db.add_file_content(
            test_file, test_content, {"type": "python"}
        )
        assert len(doc_ids) == 1
        
        # Test searching
        results = await mock_vector_db.search("hello")
        assert len(results) > 0
        assert "hello" in results[0]["content"]
    
    @pytest.mark.asyncio
    async def test_mock_llm_client(self, mock_llm_client):
        """Test mock LLM client functionality"""
        # Test generation
        response = await mock_llm_client.generate("write a fibonacci function")
        assert_llm_response_valid(response)
        assert "fibonacci" in response["content"]
        
        # Test availability
        available = await mock_llm_client.is_available()
        assert available is True
        
        # Test models
        models = await mock_llm_client.get_models()
        assert isinstance(models, list)
        assert len(models) > 0


class TestUtilityHelpers:
    """Test utility helper functions"""
    
    def test_generate_random_string(self):
        """Test random string generation"""
        string1 = generate_random_string(10)
        string2 = generate_random_string(10)
        
        assert len(string1) == 10
        assert len(string2) == 10
        assert string1 != string2  # Should be different
        assert string1.isalnum()
        assert string2.isalnum()
    
    def test_generate_test_code(self):
        """Test code generation for different languages and complexities"""
        # Test Python simple
        python_simple = generate_test_code("python", "simple")
        assert "def " in python_simple
        assert "python" in python_simple.lower() or "test" in python_simple.lower()
        
        # Test Python medium
        python_medium = generate_test_code("python", "medium")
        assert "class " in python_medium
        assert "def " in python_medium
        
        # Test JavaScript
        js_code = generate_test_code("javascript", "simple")
        assert "function" in js_code
        assert "export" in js_code
    
    def test_timer_context_manager(self):
        """Test timer context manager"""
        import time
        
        with TestTimer() as timer:
            time.sleep(0.1)
        
        assert timer.duration is not None
        assert timer.duration >= 0.1
        assert timer.start_time is not None
        assert timer.end_time is not None
    
    @pytest.mark.asyncio
    async def test_mock_llm_provider(self):
        """Test mock LLM provider"""
        responses = {
            "fibonacci": {
                "content": "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
                "usage": {"prompt_tokens": 10, "completion_tokens": 20}
            }
        }
        
        provider = MockLLMProvider(responses)
        
        # Test specific response
        response = await provider.generate("write a fibonacci function")
        assert "fibonacci" in response["content"]
        assert provider.get_call_count() == 1
        
        # Test default response
        response = await provider.generate("some other prompt")
        assert "Mock response" in response["content"]
        assert provider.get_call_count() == 2
        
        # Test call history
        history = provider.get_call_history()
        assert len(history) == 2
        assert "fibonacci" in history[0]["prompt"]


class TestAssertionHelpers:
    """Test custom assertion helpers"""
    
    def test_memory_entry_assertions(self):
        """Test memory entry assertion helpers"""
        valid_entry = {
            "id": "test_id",
            "agent_id": "test_agent",
            "timestamp": datetime.now(),
            "type": "observation",
            "content": {"data": "test"}
        }
        
        # Should not raise
        assert_memory_entry_valid(valid_entry)
        
        # Test invalid entry
        invalid_entry = valid_entry.copy()
        invalid_entry["type"] = "invalid_type"
        
        with pytest.raises(AssertionError):
            assert_memory_entry_valid(invalid_entry)
    
    def test_llm_response_assertions(self):
        """Test LLM response assertion helpers"""
        valid_response = {
            "content": "Generated code here",
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 30
            }
        }
        
        # Should not raise
        assert_llm_response_valid(valid_response)
        
        # Test invalid response
        invalid_response = {"invalid": "response"}
        
        with pytest.raises(AssertionError):
            assert_llm_response_valid(invalid_response)


@pytest.mark.slow
class TestPerformanceInfrastructure:
    """Test performance testing infrastructure"""
    
    def test_benchmark_data_fixture(self, benchmark_data):
        """Test benchmark data fixture"""
        assert "small_dataset" in benchmark_data
        assert "medium_dataset" in benchmark_data
        assert "large_dataset" in benchmark_data
        
        assert len(benchmark_data["small_dataset"]) == 100
        assert len(benchmark_data["medium_dataset"]) == 1000
        assert len(benchmark_data["large_dataset"]) == 10000
    
    @pytest.mark.benchmark
    def test_performance_measurement(self, benchmark):
        """Test performance measurement capabilities"""
        def sample_operation():
            return sum(range(1000))
        
        result = benchmark(sample_operation)
        assert result == sum(range(1000))


@pytest.mark.integration
class TestIntegrationInfrastructure:
    """Test integration testing infrastructure"""
    
    def test_multi_repo_fixture(self, mock_multi_repos):
        """Test multi-repository fixture"""
        assert "frontend" in mock_multi_repos
        assert "backend" in mock_multi_repos
        assert "shared" in mock_multi_repos
        
        for repo_name, repo_data in mock_multi_repos.items():
            assert "path" in repo_data
            assert "repo" in repo_data
            assert repo_data["path"].exists()
    
    def test_file_system_fixture(self, mock_file_system):
        """Test mock file system fixture"""
        assert mock_file_system.exists()
        assert (mock_file_system / "src" / "dev_guard").exists()
        assert (mock_file_system / "tests" / "unit").exists()


@pytest.mark.security
class TestSecurityInfrastructure:
    """Test security testing infrastructure"""
    
    def test_security_test_cases(self, security_test_cases):
        """Test security test cases fixture"""
        assert "sql_injection" in security_test_cases
        assert "command_injection" in security_test_cases
        assert "xss" in security_test_cases
        
        for vuln_type, cases in security_test_cases.items():
            assert "vulnerable" in cases
            assert "safe" in cases