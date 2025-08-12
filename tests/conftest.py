"""
Comprehensive test fixtures for DevGuard testing infrastructure.
Provides mock repositories, LLM responses, database operations, and shared test utilities.
"""

import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from factory import Factory, LazyAttribute
from factory import Faker as FactoryFaker
from faker import Faker
from git import Repo
from pydantic import BaseModel

# Test configuration
import os

# Ensure tests use local Ollama GPT-OSS by default
os.environ.setdefault("DEV_GUARD_LLM_PROVIDER", "ollama")
os.environ.setdefault("DEV_GUARD_LLM_MODEL", "gpt-oss:20b")
os.environ.setdefault("DEV_GUARD_LLM_BASE_URL", "http://localhost:11434")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

pytest_plugins = ["pytest_asyncio"]
fake = Faker()


# Base test models for consistent testing
class TestMemoryEntry(BaseModel):
    """Test model for memory entries"""
    id: str
    agent_id: str
    timestamp: datetime
    type: str
    content: dict[str, Any]
    tags: set[str] = set()
    parent_id: str | None = None
    context: dict[str, Any] = {}


class TestTaskStatus(BaseModel):
    """Test model for task status"""
    id: str
    agent_id: str
    status: str
    description: str
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = {}
    dependencies: list[str] = []
    result: dict[str, Any] | None = None
    error: str | None = None


class TestAgentState(BaseModel):
    """Test model for agent state"""
    agent_id: str
    status: str
    current_task: str | None = None
    last_heartbeat: datetime
    metadata: dict[str, Any] = {}


# Factory classes for test data generation
class MemoryEntryFactory(Factory):
    """Factory for generating test memory entries"""
    class Meta:
        model = TestMemoryEntry

    id = LazyAttribute(lambda obj: str(uuid.uuid4()))
    agent_id = FactoryFaker('word')
    timestamp = FactoryFaker('date_time')
    type = FactoryFaker('random_element', elements=['task', 'observation', 'decision', 'result', 'error'])
    content = LazyAttribute(lambda obj: {'data': fake.text(), 'value': fake.random_int()})
    tags = LazyAttribute(lambda obj: {fake.word() for _ in range(fake.random_int(1, 3))})
    context = LazyAttribute(lambda obj: {'source': fake.file_name(), 'line': fake.random_int(1, 100)})


class TaskStatusFactory(Factory):
    """Factory for generating test task status"""
    class Meta:
        model = TestTaskStatus

    id = LazyAttribute(lambda obj: str(uuid.uuid4()))
    agent_id = FactoryFaker('word')
    status = FactoryFaker('random_element', elements=['pending', 'running', 'completed', 'failed', 'cancelled'])
    description = FactoryFaker('sentence')
    created_at = FactoryFaker('date_time')
    updated_at = FactoryFaker('date_time')
    metadata = LazyAttribute(lambda obj: {'priority': fake.random_int(1, 5), 'category': fake.word()})
    dependencies = LazyAttribute(lambda obj: [str(uuid.uuid4()) for _ in range(fake.random_int(0, 2))])


class AgentStateFactory(Factory):
    """Factory for generating test agent state"""
    class Meta:
        model = TestAgentState

    agent_id = FactoryFaker('word')
    status = FactoryFaker('random_element', elements=['idle', 'busy', 'error', 'stopped'])
    last_heartbeat = FactoryFaker('date_time')
    metadata = LazyAttribute(lambda obj: {'version': fake.semantic_version(), 'uptime': fake.random_int(0, 86400)})


# Core fixtures
# Note: Using pytest-asyncio built-in event_loop fixture instead of custom one


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_config():
    """Provide test configuration dictionary"""
    return {
        "database": {
            "url": ":memory:",
            "echo": False
        },
        "vector_db": {
            "path": ":memory:",
            "collection_name": "test_collection"
        },
        "llm": {
            "provider": "mock",
            "model": "test-model",
            "temperature": 0.0,
            "max_tokens": 1000
        },
        "agents": {
            "max_retries": 3,
            "timeout": 30,
            "heartbeat_interval": 10
        },
        "repositories": [],
        "notifications": {
            "enabled": False
        }
    }


# Mock repository fixtures
@pytest.fixture
def mock_git_repo(temp_dir):
    """Create a mock Git repository for testing"""
    repo_path = temp_dir / "test_repo"
    repo_path.mkdir()

    # Initialize git repo
    repo = Repo.init(repo_path)

    # Create some test files
    (repo_path / "README.md").write_text("# Test Repository\n\nThis is a test repository.")
    (repo_path / "src").mkdir()
    (repo_path / "src" / "__init__.py").write_text("")
    (repo_path / "src" / "main.py").write_text("""
def hello_world():
    '''Simple hello world function'''
    return "Hello, World!"

if __name__ == "__main__":
    print(hello_world())
""")
    (repo_path / "tests").mkdir()
    (repo_path / "tests" / "test_main.py").write_text("""
import pytest
from src.main import hello_world

def test_hello_world():
    assert hello_world() == "Hello, World!"
""")

    # Create initial commit
    repo.index.add([str(f) for f in repo_path.rglob("*") if f.is_file()])
    repo.index.commit("Initial commit")

    return {
        "path": repo_path,
        "repo": repo,
        "files": {
            "readme": repo_path / "README.md",
            "main": repo_path / "src" / "main.py",
            "test": repo_path / "tests" / "test_main.py"
        }
    }


@pytest.fixture
def mock_multi_repos(temp_dir):
    """Create multiple mock repositories for cross-repo testing"""
    repos = {}

    for repo_name in ["frontend", "backend", "shared"]:
        repo_path = temp_dir / repo_name
        repo_path.mkdir()
        repo = Repo.init(repo_path)

        # Create repo-specific files
        if repo_name == "frontend":
            (repo_path / "package.json").write_text('{"name": "frontend", "version": "1.0.0"}')
            (repo_path / "src" / "components").mkdir(parents=True)
            (repo_path / "src" / "components" / "App.js").write_text("export default function App() { return <div>Hello</div>; }")
        elif repo_name == "backend":
            (repo_path / "requirements.txt").write_text("fastapi==0.104.1\nuvicorn==0.24.0")
            (repo_path / "src").mkdir()
            (repo_path / "src" / "main.py").write_text("from fastapi import FastAPI\napp = FastAPI()")
        else:  # shared
            (repo_path / "setup.py").write_text("from setuptools import setup\nsetup(name='shared')")
            (repo_path / "shared").mkdir()
            (repo_path / "shared" / "utils.py").write_text("def shared_function(): return 'shared'")

        repo.index.add([str(f) for f in repo_path.rglob("*") if f.is_file()])
        repo.index.commit(f"Initial {repo_name} commit")

        repos[repo_name] = {
            "path": repo_path,
            "repo": repo
        }

    return repos


# Mock LLM response fixtures
@pytest.fixture
def mock_llm_responses():
    """Provide mock LLM responses for different scenarios"""
    return {
        "code_generation": {
            "content": """
def calculate_fibonacci(n: int) -> int:
    '''Calculate the nth Fibonacci number'''
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
""",
            "reasoning": "Generated a recursive Fibonacci function with proper type hints and docstring",
            "confidence": 0.95
        },
        "code_review": {
            "content": "The code looks good but could benefit from memoization for better performance",
            "suggestions": [
                "Add @lru_cache decorator for memoization",
                "Consider iterative approach for large n values",
                "Add input validation for negative numbers"
            ],
            "confidence": 0.88
        },
        "documentation": {
            "content": """
# API Documentation

## Overview
This module provides utility functions for mathematical calculations.

## Functions

### calculate_fibonacci(n: int) -> int
Calculates the nth Fibonacci number using recursive approach.

**Parameters:**
- n (int): The position in the Fibonacci sequence

**Returns:**
- int: The nth Fibonacci number

**Example:**
```python
result = calculate_fibonacci(10)
print(result)  # Output: 55
```
""",
            "confidence": 0.92
        },
        "error_analysis": {
            "content": "RecursionError: maximum recursion depth exceeded",
            "cause": "Recursive function without proper base case handling for large inputs",
            "solution": "Implement iterative approach or add memoization",
            "confidence": 0.97
        },
        "security_scan": {
            "vulnerabilities": [
                {
                    "type": "CWE-78",
                    "severity": "HIGH",
                    "description": "OS Command Injection",
                    "file": "src/utils.py",
                    "line": 42,
                    "recommendation": "Use subprocess with shell=False and validate inputs"
                }
            ],
            "confidence": 0.91
        }
    }


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing"""
    client = AsyncMock()

    async def mock_generate(prompt: str, **kwargs) -> dict[str, Any]:
        # Simple mock response based on prompt keywords
        if "fibonacci" in prompt.lower():
            return {
                "content": "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
                "usage": {"prompt_tokens": 50, "completion_tokens": 30}
            }
        elif "test" in prompt.lower():
            return {
                "content": "def test_function(): assert True",
                "usage": {"prompt_tokens": 30, "completion_tokens": 20}
            }
        else:
            return {
                "content": "# Generated code based on prompt",
                "usage": {"prompt_tokens": 40, "completion_tokens": 25}
            }

    client.generate = mock_generate
    client.is_available = AsyncMock(return_value=True)
    client.get_models = AsyncMock(return_value=["test-model", "backup-model"])

    return client


# Database fixtures
@pytest.fixture
async def mock_shared_memory():
    """Mock shared memory system for testing"""
    memory = AsyncMock()
    memory.entries = {}
    memory.tasks = {}
    memory.agent_states = {}

    async def add_memory(entry: TestMemoryEntry) -> str:
        memory.entries[entry.id] = entry
        return entry.id

    async def get_memories(agent_id: str, memory_type: str = None) -> list[TestMemoryEntry]:
        results = []
        for entry in memory.entries.values():
            if entry.agent_id == agent_id:
                if memory_type is None or entry.type == memory_type:
                    results.append(entry)
        return results

    async def create_task(task: TestTaskStatus) -> str:
        memory.tasks[task.id] = task
        return task.id

    async def update_task(task_id: str, **updates) -> bool:
        if task_id in memory.tasks:
            for key, value in updates.items():
                setattr(memory.tasks[task_id], key, value)
            return True
        return False

    async def update_agent_state(state: TestAgentState) -> None:
        memory.agent_states[state.agent_id] = state

    memory.add_memory = add_memory
    memory.get_memories = get_memories
    memory.create_task = create_task
    memory.update_task = update_task
    memory.update_agent_state = update_agent_state

    return memory


@pytest.fixture
async def mock_vector_db():
    """Mock vector database for testing"""
    vector_db = AsyncMock()
    vector_db.documents = {}
    vector_db.embeddings = {}

    async def add_file_content(file_path: Path, content: str, metadata: dict) -> list[str]:
        doc_id = str(uuid.uuid4())
        vector_db.documents[doc_id] = {
            "id": doc_id,
            "content": content,
            "metadata": {**metadata, "file_path": str(file_path)},
            "source": str(file_path)
        }
        return [doc_id]

    async def search(query: str, n_results: int = 10, where: dict = None) -> list[dict]:
        # Simple mock search - return documents containing query terms
        results = []
        for doc in vector_db.documents.values():
            if any(term.lower() in doc["content"].lower() for term in query.split()):
                results.append(doc)
                if len(results) >= n_results:
                    break
        return results

    async def search_code(query: str, file_extensions: list[str] = None) -> list[dict]:
        results = []
        for doc in vector_db.documents.values():
            file_path = doc["metadata"].get("file_path", "")
            if file_extensions:
                if not any(file_path.endswith(ext) for ext in file_extensions):
                    continue
            if query.lower() in doc["content"].lower():
                results.append(doc)
        return results

    vector_db.add_file_content = add_file_content
    vector_db.search = search
    vector_db.search_code = search_code
    vector_db.get_collection = AsyncMock(return_value=MagicMock())

    return vector_db


# Agent fixtures
@pytest.fixture
def mock_base_agent():
    """Mock base agent for testing"""
    agent = AsyncMock()
    agent.agent_id = "test_agent"
    agent.status = "idle"
    agent.memory_entries = []
    agent.tasks = []

    async def log_observation(observation: str, data: dict = None) -> str:
        entry_id = str(uuid.uuid4())
        agent.memory_entries.append({
            "id": entry_id,
            "type": "observation",
            "content": observation,
            "data": data or {},
            "timestamp": datetime.now()
        })
        return entry_id

    async def log_decision(decision: str, reasoning: str) -> str:
        entry_id = str(uuid.uuid4())
        agent.memory_entries.append({
            "id": entry_id,
            "type": "decision",
            "content": decision,
            "reasoning": reasoning,
            "timestamp": datetime.now()
        })
        return entry_id

    async def create_task(description: str, task_type: str) -> str:
        task_id = str(uuid.uuid4())
        agent.tasks.append({
            "id": task_id,
            "description": description,
            "type": task_type,
            "status": "pending",
            "created_at": datetime.now()
        })
        return task_id

    agent.log_observation = log_observation
    agent.log_decision = log_decision
    agent.create_task = create_task
    agent.execute = AsyncMock(return_value={"status": "completed", "result": "success"})

    return agent


# Utility fixtures
@pytest.fixture
def sample_code_files():
    """Provide sample code files for testing"""
    return {
        "python": {
            "simple.py": """
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
""",
            "class.py": """
class Calculator:
    def __init__(self):
        self.history = []

    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

    def get_history(self):
        return self.history
""",
            "async.py": """
import asyncio

async def fetch_data(url):
    # Simulate async operation
    await asyncio.sleep(0.1)
    return f"Data from {url}"

async def process_urls(urls):
    tasks = [fetch_data(url) for url in urls]
    return await asyncio.gather(*tasks)
"""
        },
        "javascript": {
            "utils.js": """
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

export { debounce };
""",
            "api.js": """
class ApiClient {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }

    async get(endpoint) {
        const response = await fetch(`${this.baseUrl}${endpoint}`);
        return response.json();
    }

    async post(endpoint, data) {
        const response = await fetch(`${this.baseUrl}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        return response.json();
    }
}

export default ApiClient;
"""
        }
    }


@pytest.fixture
def mock_file_system(temp_dir):
    """Create a mock file system structure for testing"""
    structure = {
        "src": {
            "dev_guard": {
                "__init__.py": "",
                "agents": {
                    "__init__.py": "",
                    "base_agent.py": "# Base agent implementation",
                    "commander.py": "# Commander agent implementation"
                },
                "core": {
                    "__init__.py": "",
                    "config.py": "# Configuration management",
                    "swarm.py": "# Swarm orchestration"
                },
                "memory": {
                    "__init__.py": "",
                    "shared_memory.py": "# Shared memory implementation",
                    "vector_db.py": "# Vector database implementation"
                }
            }
        },
        "tests": {
            "__init__.py": "",
            "unit": {
                "__init__.py": "",
                "test_agents.py": "# Agent unit tests",
                "test_memory.py": "# Memory unit tests"
            },
            "integration": {
                "__init__.py": "",
                "test_swarm.py": "# Swarm integration tests"
            }
        },
        "docs": {
            "README.md": "# DevGuard Documentation",
            "api.md": "# API Documentation"
        }
    }

    def create_structure(base_path: Path, structure: dict):
        for name, content in structure.items():
            path = base_path / name
            if isinstance(content, dict):
                path.mkdir(exist_ok=True)
                create_structure(path, content)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)

    create_structure(temp_dir, structure)
    return temp_dir


# Performance and benchmark fixtures
@pytest.fixture
def benchmark_data():
    """Provide data for performance benchmarking"""
    return {
        "small_dataset": [fake.text() for _ in range(100)],
        "medium_dataset": [fake.text() for _ in range(1000)],
        "large_dataset": [fake.text() for _ in range(10000)],
        "code_samples": [
            fake.text() + "\ndef function():\n    pass"
            for _ in range(500)
        ]
    }


# Security testing fixtures
@pytest.fixture
def security_test_cases():
    """Provide security test cases and vulnerable code samples"""
    return {
        "sql_injection": {
            "vulnerable": "query = f'SELECT * FROM users WHERE id = {user_id}'",
            "safe": "query = 'SELECT * FROM users WHERE id = %s'"
        },
        "command_injection": {
            "vulnerable": "os.system(f'ls {user_input}')",
            "safe": "subprocess.run(['ls', user_input], check=True)"
        },
        "xss": {
            "vulnerable": "return f'<div>{user_content}</div>'",
            "safe": "return f'<div>{html.escape(user_content)}</div>'"
        },
        "path_traversal": {
            "vulnerable": "open(f'/files/{filename}', 'r')",
            "safe": "open(os.path.join('/files', os.path.basename(filename)), 'r')"
        }
    }


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_test_environment():
    """Automatically cleanup test environment after each test"""
    yield
    # Cleanup any global state, temporary files, etc.
    # This runs after each test
    pass


# Parametrized fixtures for comprehensive testing
@pytest.fixture(params=["sqlite", "memory"])
def database_backend(request):
    """Parametrized fixture for different database backends"""
    return request.param


@pytest.fixture(params=["local", "cloud", "hybrid"])
def llm_provider_type(request):
    """Parametrized fixture for different LLM provider types"""
    return request.param


@pytest.fixture(params=[1, 5, 10])
def agent_count(request):
    """Parametrized fixture for different agent counts"""
    return request.param