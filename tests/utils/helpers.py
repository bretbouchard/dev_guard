"""
Test helper utilities for DevGuard testing infrastructure.
Provides common testing utilities, mock generators, and test data creation.
"""

import asyncio
import os
import random
import string
import time
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest


class TestTimer:
    """Context manager for timing test operations"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time


class MemoryTracker:
    """Track memory usage during tests"""
    
    def __init__(self):
        self.initial_memory = None
        self.peak_memory = None
        self.final_memory = None
    
    def __enter__(self):
        try:
            import psutil
            process = psutil.Process()
            self.initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = self.initial_memory
        except ImportError:
            pytest.skip("psutil not available for memory tracking")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            import psutil
            process = psutil.Process()
            self.final_memory = process.memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = max(self.peak_memory, self.final_memory)
        except ImportError:
            pass


def generate_random_string(length: int = 10) -> str:
    """Generate a random string of specified length"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def generate_test_code(language: str = "python", complexity: str = "simple") -> str:
    """Generate test code samples for different languages and complexity levels"""
    
    if language == "python":
        if complexity == "simple":
            return f"""
def test_function_{generate_random_string(5)}():
    '''Simple test function'''
    return "Hello, World!"

def add_numbers(a, b):
    '''Add two numbers'''
    return a + b
"""
        elif complexity == "medium":
            return f"""
class TestClass_{generate_random_string(5)}:
    '''Test class with multiple methods'''
    
    def __init__(self, name):
        self.name = name
        self.data = []
    
    def add_item(self, item):
        '''Add item to data'''
        self.data.append(item)
        return len(self.data)
    
    def get_summary(self):
        '''Get summary of data'''
        return {{
            'name': self.name,
            'count': len(self.data),
            'items': self.data
        }}
    
    async def async_operation(self):
        '''Async operation example'''
        await asyncio.sleep(0.1)
        return f"Processed {{len(self.data)}} items"
"""
        else:  # complex
            return f"""
import asyncio
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class DataModel_{generate_random_string(5)}:
    '''Complex data model'''
    id: str
    name: str
    metadata: Dict[str, Any]
    created_at: datetime
    
    def validate(self) -> bool:
        '''Validate data model'''
        return all([
            self.id and len(self.id) > 0,
            self.name and len(self.name) > 0,
            isinstance(self.metadata, dict),
            isinstance(self.created_at, datetime)
        ])

class AbstractProcessor_{generate_random_string(5)}(ABC):
    '''Abstract processor interface'''
    
    @abstractmethod
    async def process(self, data: DataModel_{generate_random_string(5)}) -> Dict[str, Any]:
        '''Process data model'''
        pass
    
    @abstractmethod
    def validate_input(self, data: Any) -> bool:
        '''Validate input data'''
        pass

class ConcreteProcessor(AbstractProcessor_{generate_random_string(5)}):
    '''Concrete processor implementation'''
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processed_count = 0
    
    async def process(self, data: DataModel_{generate_random_string(5)}) -> Dict[str, Any]:
        '''Process data with complex logic'''
        if not self.validate_input(data):
            raise ValueError("Invalid input data")
        
        # Simulate complex processing
        await asyncio.sleep(0.01)
        
        result = {{
            'processed_id': data.id,
            'original_name': data.name,
            'processed_name': data.name.upper(),
            'metadata_keys': list(data.metadata.keys()),
            'processing_time': datetime.now(),
            'processor_config': self.config
        }}
        
        self.processed_count += 1
        return result
    
    def validate_input(self, data: Any) -> bool:
        '''Validate input data'''
        return isinstance(data, DataModel_{generate_random_string(5)}) and data.validate()
"""
    
    elif language == "javascript":
        if complexity == "simple":
            return f"""
function testFunction_{generate_random_string(5)}() {{
    // Simple test function
    return "Hello, World!";
}}

function addNumbers(a, b) {{
    // Add two numbers
    return a + b;
}}

export {{ testFunction_{generate_random_string(5)}, addNumbers }};
"""
        elif complexity == "medium":
            return f"""
class TestClass_{generate_random_string(5)} {{
    constructor(name) {{
        this.name = name;
        this.data = [];
    }}
    
    addItem(item) {{
        this.data.push(item);
        return this.data.length;
    }}
    
    getSummary() {{
        return {{
            name: this.name,
            count: this.data.length,
            items: [...this.data]
        }};
    }}
    
    async asyncOperation() {{
        await new Promise(resolve => setTimeout(resolve, 100));
        return `Processed ${{this.data.length}} items`;
    }}
}}

export default TestClass_{generate_random_string(5)};
"""
    
    return "// Unsupported language or complexity"


def create_mock_git_history(repo_path: Path, num_commits: int = 5) -> list[dict[str, Any]]:
    """Create mock Git history with commits"""
    from git import Repo
    
    repo = Repo(repo_path)
    commits = []
    
    for i in range(num_commits):
        # Create or modify a file
        test_file = repo_path / f"file_{i}.txt"
        test_file.write_text(f"Content for commit {i}\n{generate_random_string(50)}")
        
        # Add and commit
        repo.index.add([str(test_file)])
        commit = repo.index.commit(f"Test commit {i}: {generate_random_string(10)}")
        
        commits.append({
            "hash": commit.hexsha,
            "message": commit.message,
            "author": str(commit.author),
            "timestamp": commit.committed_datetime,
            "files": [str(test_file)]
        })
    
    return commits


@contextmanager
def temporary_environment_variables(**env_vars) -> Generator[None, None, None]:
    """Temporarily set environment variables for testing"""
    original_values = {}
    
    # Store original values and set new ones
    for key, value in env_vars.items():
        original_values[key] = os.environ.get(key)
        os.environ[key] = str(value)
    
    try:
        yield
    finally:
        # Restore original values
        for key, original_value in original_values.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


@asynccontextmanager
async def mock_async_context(**kwargs) -> AsyncGenerator[dict[str, Any], None]:
    """Create an async context manager for testing async operations"""
    context_data = {
        "started_at": datetime.now(),
        "mock_data": kwargs,
        "operations": []
    }
    
    try:
        yield context_data
    finally:
        context_data["ended_at"] = datetime.now()
        context_data["duration"] = (
            context_data["ended_at"] - context_data["started_at"]
        ).total_seconds()


def create_test_database_url(backend: str = "sqlite") -> str:
    """Create a test database URL for different backends"""
    if backend == "sqlite":
        return "sqlite:///:memory:"
    elif backend == "postgresql":
        return "postgresql://test:test@localhost:5432/test_devguard"
    elif backend == "mysql":
        return "mysql://test:test@localhost:3306/test_devguard"
    else:
        raise ValueError(f"Unsupported database backend: {backend}")


def generate_mock_llm_conversation(num_exchanges: int = 3) -> list[dict[str, str]]:
    """Generate a mock conversation between user and LLM"""
    conversation = []
    
    for i in range(num_exchanges):
        # User message
        user_prompts = [
            "Can you help me write a function to calculate fibonacci numbers?",
            "How can I optimize this code for better performance?",
            "Please add error handling to this function",
            "Can you write unit tests for this code?",
            "How do I make this code more readable?"
        ]
        
        conversation.append({
            "role": "user",
            "content": random.choice(user_prompts)
        })
        
        # Assistant response
        assistant_responses = [
            "I'll help you create a fibonacci function with proper error handling and optimization.",
            "Here's an optimized version using memoization to improve performance.",
            "I've added comprehensive error handling with proper exception types.",
            "Here are comprehensive unit tests covering edge cases and normal operation.",
            "I've refactored the code with better variable names and documentation."
        ]
        
        conversation.append({
            "role": "assistant",
            "content": random.choice(assistant_responses)
        })
    
    return conversation


def create_mock_vector_embeddings(texts: list[str], dimension: int = 384) -> list[list[float]]:
    """Create mock vector embeddings for testing"""
    embeddings = []
    
    for text in texts:
        # Create deterministic but varied embeddings based on text content
        random.seed(hash(text) % (2**32))
        embedding = [random.uniform(-1, 1) for _ in range(dimension)]
        
        # Normalize the embedding
        magnitude = sum(x**2 for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        embeddings.append(embedding)
    
    return embeddings


def simulate_file_changes(repo_path: Path, num_changes: int = 3) -> list[dict[str, Any]]:
    """Simulate file changes in a repository"""
    changes = []
    
    for i in range(num_changes):
        change_type = random.choice(["create", "modify", "delete"])
        file_name = f"test_file_{i}_{generate_random_string(5)}.py"
        file_path = repo_path / file_name
        
        if change_type == "create":
            content = generate_test_code("python", "simple")
            file_path.write_text(content)
            changes.append({
                "type": "create",
                "file": str(file_path),
                "content": content
            })
        
        elif change_type == "modify" and file_path.exists():
            original_content = file_path.read_text()
            new_content = original_content + f"\n# Modified at {datetime.now()}"
            file_path.write_text(new_content)
            changes.append({
                "type": "modify",
                "file": str(file_path),
                "original_content": original_content,
                "new_content": new_content
            })
        
        elif change_type == "delete" and file_path.exists():
            original_content = file_path.read_text()
            file_path.unlink()
            changes.append({
                "type": "delete",
                "file": str(file_path),
                "original_content": original_content
            })
    
    return changes


class MockLLMProvider:
    """Mock LLM provider for testing"""
    
    def __init__(self, responses: dict[str, Any] | None = None):
        self.responses = responses or {}
        self.call_count = 0
        self.call_history = []
    
    async def generate(self, prompt: str, **kwargs) -> dict[str, Any]:
        """Generate mock response based on prompt"""
        self.call_count += 1
        self.call_history.append({
            "prompt": prompt,
            "kwargs": kwargs,
            "timestamp": datetime.now()
        })
        
        # Return predefined response if available
        for key, response in self.responses.items():
            if key.lower() in prompt.lower():
                return response
        
        # Default response
        return {
            "content": f"Mock response for: {prompt[:50]}...",
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": 20
            }
        }
    
    async def is_available(self) -> bool:
        """Check if provider is available"""
        return True
    
    def get_call_count(self) -> int:
        """Get number of calls made"""
        return self.call_count
    
    def get_call_history(self) -> list[dict[str, Any]]:
        """Get history of all calls"""
        return self.call_history.copy()


def create_performance_test_data(size: str = "medium") -> dict[str, Any]:
    """Create test data for performance testing"""
    sizes = {
        "small": {"files": 10, "lines_per_file": 50, "agents": 2},
        "medium": {"files": 100, "lines_per_file": 200, "agents": 5},
        "large": {"files": 1000, "lines_per_file": 500, "agents": 10}
    }
    
    config = sizes.get(size, sizes["medium"])
    
    return {
        "files": [
            {
                "name": f"file_{i}.py",
                "content": generate_test_code("python", "medium"),
                "size": config["lines_per_file"]
            }
            for i in range(config["files"])
        ],
        "agents": [
            {
                "id": f"agent_{i}",
                "type": random.choice(["code", "test", "docs", "security"]),
                "status": "idle"
            }
            for i in range(config["agents"])
        ],
        "tasks": [
            {
                "id": f"task_{i}",
                "description": f"Process file_{i % config['files']}.py",
                "type": "code_analysis",
                "priority": random.randint(1, 5)
            }
            for i in range(config["files"] // 2)
        ]
    }


async def wait_for_condition(
    condition_func,
    timeout: float = 5.0,
    interval: float = 0.1,
    error_message: str = "Condition not met within timeout"
) -> None:
    """Wait for a condition to become true within a timeout"""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if await condition_func() if asyncio.iscoroutinefunction(condition_func) else condition_func():
            return
        await asyncio.sleep(interval)
    
    raise TimeoutError(error_message)


def assert_json_schema(data: Any, schema: dict[str, Any]) -> None:
    """Assert that data matches a JSON schema (simplified validation)"""
    try:
        import jsonschema
        jsonschema.validate(data, schema)
    except ImportError:
        # Fallback to basic validation if jsonschema not available
        if "type" in schema:
            expected_type = schema["type"]
            if expected_type == "object":
                assert isinstance(data, dict)
            elif expected_type == "array":
                assert isinstance(data, list)
            elif expected_type == "string":
                assert isinstance(data, str)
            elif expected_type == "number":
                assert isinstance(data, (int, float))
            elif expected_type == "boolean":
                assert isinstance(data, bool)


class TestDataGenerator:
    """Generate various types of test data"""
    
    @staticmethod
    def create_agent_config(agent_type: str = "base") -> dict[str, Any]:
        """Create agent configuration for testing"""
        base_config = {
            "id": f"{agent_type}_agent_{generate_random_string(5)}",
            "type": agent_type,
            "max_retries": 3,
            "timeout": 30,
            "heartbeat_interval": 10,
            "memory_limit": 1000,
            "enabled": True
        }
        
        type_specific = {
            "commander": {"priority": 1, "oversight_interval": 60},
            "planner": {"max_tasks": 50, "load_balance": True},
            "code": {"goose_enabled": True, "auto_format": True},
            "qa": {"coverage_threshold": 95, "strict_mode": True},
            "security": {"scan_interval": 3600, "alert_threshold": "medium"}
        }
        
        if agent_type in type_specific:
            base_config.update(type_specific[agent_type])
        
        return base_config
    
    @staticmethod
    def create_repository_config(repo_name: str = None) -> dict[str, Any]:
        """Create repository configuration for testing"""
        return {
            "name": repo_name or f"test_repo_{generate_random_string(5)}",
            "path": f"/tmp/test_repos/{repo_name or generate_random_string(10)}",
            "branch": "main",
            "auto_commit": False,
            "auto_push": False,
            "ignore_patterns": ["*.pyc", "__pycache__", ".git"],
            "watch_files": ["*.py", "*.js", "*.ts", "*.md"],
            "scan_interval": 30
        }
    
    @staticmethod
    def create_task_data(task_type: str = "generic") -> dict[str, Any]:
        """Create task data for testing"""
        return {
            "id": str(random.randint(1000, 9999)),
            "type": task_type,
            "description": f"Test {task_type} task - {generate_random_string(20)}",
            "priority": random.randint(1, 5),
            "agent_id": f"agent_{generate_random_string(5)}",
            "status": "pending",
            "created_at": datetime.now(),
            "metadata": {
                "source": "test",
                "complexity": random.choice(["low", "medium", "high"]),
                "estimated_duration": random.randint(60, 3600)
            }
        }