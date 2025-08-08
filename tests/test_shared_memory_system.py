"""Comprehensive test suite for Shared Memory system.
This module tests the SQLite-based shared memory system including
memory entries, task status tracking, and agent state management.
"""

import concurrent.futures
import json
import sqlite3
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Import availability check
try:
    from src.dev_guard.memory.shared_memory import AgentState, MemoryEntry, SharedMemory, TaskStatus
    SHARED_MEMORY_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Shared memory imports not available: {e}")
    SHARED_MEMORY_IMPORTS_AVAILABLE = False


@pytest.fixture
def temp_db_path():
    """Create a temporary database path for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        temp_path = f.name
    yield temp_path
    # Cleanup
    try:
        Path(temp_path).unlink(missing_ok=True)
    except Exception:
        pass


@pytest.fixture
def shared_memory(temp_db_path):
    """Create a SharedMemory instance for testing."""
    if not SHARED_MEMORY_IMPORTS_AVAILABLE:
        pytest.skip("Shared memory modules not available")
    
    memory = SharedMemory(db_path=temp_db_path)
    yield memory
    # Cleanup - SharedMemory doesn't have close method


# MemoryEntry Model Tests
@pytest.mark.skipif(not SHARED_MEMORY_IMPORTS_AVAILABLE, reason="Shared memory modules not available")
class TestMemoryEntry:
    """Test MemoryEntry model validation and functionality."""

    def test_memory_entry_creation(self):
        """Test basic MemoryEntry creation."""
        entry = MemoryEntry(
            agent_id="test_agent",
            type="observation",
            content={"message": "Test observation"}
        )
        
        assert entry.agent_id == "test_agent"
        assert entry.type == "observation"
        assert entry.content == {"message": "Test observation"}
        assert len(entry.id) == 36  # UUID format
        assert isinstance(entry.timestamp, datetime)
        assert entry.tags == set()
        assert entry.parent_id is None
        assert entry.context == {}

    def test_memory_entry_with_goose_metadata(self):
        """Test MemoryEntry with Goose patch and AST metadata."""
        entry = MemoryEntry(
            agent_id="code_agent",
            type="result",
            content={"code_change": "Added function"},
            goose_patch={
                "command": "fix",
                "working_dir": "/test/path",
                "output": {"status": "success"}
            },
            ast_summary={
                "functions_added": ["new_function"],
                "complexity": "low"
            },
            goose_strategy="refactor_for_clarity",
            file_path="/test/path/module.py"
        )
        
        assert entry.goose_patch["command"] == "fix"
        assert entry.ast_summary["functions_added"] == ["new_function"]
        assert entry.goose_strategy == "refactor_for_clarity"
        assert entry.file_path == "/test/path/module.py"

    def test_memory_entry_validation_errors(self):
        """Test MemoryEntry validation errors."""
        # Test empty agent_id
        with pytest.raises(ValueError, match="agent_id cannot be empty"):
            MemoryEntry(
                agent_id="",
                type="observation",
                content={"test": "value"}
            )
        
        # Test invalid type
        with pytest.raises(ValueError):
            MemoryEntry(
                agent_id="test_agent",
                type="invalid_type",
                content={"test": "value"}
            )
        
        # Test empty content
        with pytest.raises(ValueError, match="content must be a non-empty dictionary"):
            MemoryEntry(
                agent_id="test_agent",
                type="observation",
                content={}
            )

    def test_memory_entry_tags_validation(self):
        """Test MemoryEntry tags validation."""
        entry = MemoryEntry(
            agent_id="test_agent",
            type="observation",
            content={"test": "value"},
            tags={"tag1", "tag2", "", "  ", "tag3"}
        )
        
        # Empty and whitespace-only tags should be filtered out
        assert entry.tags == {"tag1", "tag2", "tag3"}

    def test_memory_entry_parent_child_relationship(self):
        """Test MemoryEntry parent-child relationships."""
        parent_entry = MemoryEntry(
            agent_id="test_agent",
            type="task",
            content={"task": "parent task"}
        )
        
        child_entry = MemoryEntry(
            agent_id="test_agent",
            type="observation",
            content={"observation": "child observation"},
            parent_id=parent_entry.id
        )
        
        assert child_entry.parent_id == parent_entry.id


# TaskStatus Model Tests
@pytest.mark.skipif(not SHARED_MEMORY_IMPORTS_AVAILABLE, reason="Shared memory modules not available")
class TestTaskStatus:
    """Test TaskStatus model validation and functionality."""

    def test_task_status_creation(self):
        """Test basic TaskStatus creation."""
        task = TaskStatus(
            agent_id="planner_agent",
            status="pending",
            description="Test task for validation"
        )
        
        assert task.agent_id == "planner_agent"
        assert task.status == "pending"
        assert task.description == "Test task for validation"
        assert len(task.id) == 36  # UUID format
        assert isinstance(task.created_at, datetime)
        assert isinstance(task.updated_at, datetime)
        assert task.metadata == {}
        assert task.dependencies == []
        assert task.result is None
        assert task.error is None

    def test_task_status_with_metadata(self):
        """Test TaskStatus with metadata and dependencies."""
        task = TaskStatus(
            agent_id="code_agent",
            status="running",
            description="Generate unit tests",
            metadata={
                "file_path": "/test/module.py",
                "test_type": "unit",
                "priority": "high"
            },
            dependencies=["task-1", "task-2"]
        )
        
        assert task.metadata["file_path"] == "/test/module.py"
        assert task.metadata["priority"] == "high"
        assert "task-1" in task.dependencies
        assert "task-2" in task.dependencies

    def test_task_status_completion(self):
        """Test TaskStatus with completion result."""
        task = TaskStatus(
            agent_id="test_agent",
            status="completed",
            description="Completed task",
            result={
                "output": "Task completed successfully",
                "metrics": {"duration": 5.2}
            }
        )
        
        assert task.status == "completed"
        assert task.result["output"] == "Task completed successfully"
        assert task.result["metrics"]["duration"] == 5.2

    def test_task_status_failure(self):
        """Test TaskStatus with error information."""
        task = TaskStatus(
            agent_id="test_agent",
            status="failed",
            description="Failed task",
            error="Task failed due to missing dependency"
        )
        
        assert task.status == "failed"
        assert "missing dependency" in task.error

    def test_task_status_validation(self):
        """Test TaskStatus validation errors."""
        # Test empty agent_id
        with pytest.raises(ValueError, match="agent_id cannot be empty"):
            TaskStatus(
                agent_id="",
                status="pending",
                description="Test task"
            )
        
        # Test invalid status
        with pytest.raises(ValueError):
            TaskStatus(
                agent_id="test_agent",
                status="invalid_status",
                description="Test task"
            )


# AgentState Model Tests  
@pytest.mark.skipif(not SHARED_MEMORY_IMPORTS_AVAILABLE, reason="Shared memory modules not available")
class TestAgentState:
    """Test AgentState model validation and functionality."""

    def test_agent_state_creation(self):
        """Test basic AgentState creation."""
        if not hasattr(globals().get('AgentState', type(None)), '__name__'):
            pytest.skip("AgentState not available in this version")
        
        state = AgentState(
            agent_id="commander_agent",
            status="active"
        )
        
        assert state.agent_id == "commander_agent"
        assert state.status == "active"


# SharedMemory System Tests
@pytest.mark.skipif(not SHARED_MEMORY_IMPORTS_AVAILABLE, reason="Shared memory modules not available")
class TestSharedMemorySystem:
    """Test SharedMemory system functionality."""

    def test_shared_memory_initialization(self, temp_db_path):
        """Test SharedMemory initialization."""
        memory = SharedMemory(db_path=temp_db_path)
        
        assert Path(temp_db_path).exists()
        assert memory.db_path == temp_db_path
        
        # Test database schema creation
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.cursor()
            
            # Check if tables exist
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name IN ('memories', 'tasks', 'agents')
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            assert 'memories' in tables
            assert 'tasks' in tables
        
        memory.close()

    def test_add_memory_entry(self, shared_memory):
        """Test adding memory entries."""
        entry = MemoryEntry(
            agent_id="test_agent",
            type="observation",
            content={"message": "Test memory entry"}
        )
        
        result_id = shared_memory.add_memory(entry)
        assert result_id == entry.id
        assert isinstance(result_id, str)
        
        # Verify entry was stored
        memories = shared_memory.get_memories(agent_id="test_agent")
        assert len(memories) == 1
        assert memories[0].content == {"message": "Test memory entry"}

    def test_get_memories_filtering(self, shared_memory):
        """Test memory retrieval with filtering."""
        # Add multiple entries
        entries = [
            MemoryEntry(
                agent_id="agent1",
                type="observation",
                content={"index": 1},
                tags={"tag1"}
            ),
            MemoryEntry(
                agent_id="agent2", 
                type="decision",
                content={"index": 2},
                tags={"tag2"}
            ),
            MemoryEntry(
                agent_id="agent1",
                type="result",
                content={"index": 3},
                tags={"tag1", "tag3"}
            )
        ]
        
        for entry in entries:
            shared_memory.add_memory(entry)
        
        # Test filtering by agent_id
        agent1_memories = shared_memory.get_memories(agent_id="agent1")
        assert len(agent1_memories) == 2
        
        # Test filtering by type
        observation_memories = shared_memory.get_memories(type="observation")
        assert len(observation_memories) == 1
        
        # Test filtering by tags
        tag1_memories = shared_memory.get_memories(tags=["tag1"])
        assert len(tag1_memories) == 2

    def test_memory_conversation_threading(self, shared_memory):
        """Test conversation threading functionality."""
        # Create parent memory
        parent_entry = MemoryEntry(
            agent_id="test_agent",
            type="task",
            content={"task": "main task"}
        )
        shared_memory.add_memory(parent_entry)
        
        # Create child memories
        child1 = MemoryEntry(
            agent_id="test_agent",
            type="observation",
            content={"observation": "first step"},
            parent_id=parent_entry.id
        )
        child2 = MemoryEntry(
            agent_id="test_agent",
            type="result",
            content={"result": "completed"},
            parent_id=parent_entry.id
        )
        
        shared_memory.add_memory(child1)
        shared_memory.add_memory(child2)
        
        # Test thread retrieval
        thread = shared_memory.get_conversation_thread(parent_entry.id)
        assert len(thread) >= 3  # parent + 2 children
        
        # Verify parent is in thread
        parent_in_thread = any(m.id == parent_entry.id for m in thread)
        assert parent_in_thread

    def test_add_task_status(self, shared_memory):
        """Test adding task status."""
        task = TaskStatus(
            agent_id="planner_agent",
            status="pending",
            description="Test task"
        )
        
        success = shared_memory.add_task(task)
        assert success is True
        
        # Verify task was stored
        tasks = shared_memory.get_tasks(agent_id="planner_agent")
        assert len(tasks) == 1
        assert tasks[0].description == "Test task"

    def test_update_task_status(self, shared_memory):
        """Test updating task status."""
        task = TaskStatus(
            agent_id="test_agent",
            status="pending",
            description="Task to update"
        )
        
        shared_memory.add_task(task)
        
        # Update task status
        success = shared_memory.update_task_status(
            task.id,
            "running",
            result={"progress": "50%"}
        )
        assert success is True
        
        # Verify update
        updated_tasks = shared_memory.get_tasks(task_id=task.id)
        assert len(updated_tasks) == 1
        assert updated_tasks[0].status == "running"
        assert updated_tasks[0].result == {"progress": "50%"}

    def test_memory_search(self, shared_memory):
        """Test memory search functionality."""
        # Add searchable entries
        entries = [
            MemoryEntry(
                agent_id="test_agent",
                type="observation",
                content={"message": "Python function implementation"}
            ),
            MemoryEntry(
                agent_id="test_agent", 
                type="result",
                content={"code": "def test_function(): return True"}
            ),
            MemoryEntry(
                agent_id="test_agent",
                type="observation",
                content={"message": "JavaScript async function"}
            )
        ]
        
        for entry in entries:
            shared_memory.add_memory(entry)
        
        # Search for Python-related memories
        python_memories = shared_memory.search_memories("Python")
        assert len(python_memories) >= 1
        
        # Search for function-related memories
        function_memories = shared_memory.search_memories("function")
        assert len(function_memories) >= 2

    def test_memory_cleanup(self, shared_memory):
        """Test memory cleanup and retention policies."""
        # Add old entries
        old_timestamp = datetime.now(timezone.utc) - timedelta(days=35)
        
        with patch('src.dev_guard.memory.shared_memory.datetime') as mock_datetime:
            mock_datetime.now.return_value = old_timestamp
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            old_entry = MemoryEntry(
                agent_id="test_agent",
                type="observation",
                content={"old": "data"}
            )
            old_entry.timestamp = old_timestamp
            shared_memory.add_memory(old_entry)
        
        # Add recent entries
        recent_entry = MemoryEntry(
            agent_id="test_agent",
            type="observation", 
            content={"recent": "data"}
        )
        shared_memory.add_memory(recent_entry)
        
        # Test cleanup (assuming 30-day retention)
        cleanup_count = shared_memory.cleanup_old_memories(retention_days=30)
        
        # Verify recent entry remains
        recent_memories = shared_memory.get_memories()
        recent_data = [m for m in recent_memories if "recent" in str(m.content)]
        assert len(recent_data) > 0

    def test_concurrent_memory_operations(self, shared_memory):
        """Test concurrent memory operations."""
        results = []
        errors = []
        
        def add_memory_worker(worker_id):
            """Worker function for concurrent testing."""
            try:
                for i in range(5):
                    entry = MemoryEntry(
                        agent_id=f"worker_{worker_id}",
                        type="observation",
                        content={"worker_id": worker_id, "iteration": i}
                    )
                    success = shared_memory.add_memory(entry)
                    if success:
                        results.append(f"worker_{worker_id}_iter_{i}")
                    time.sleep(0.001)  # Small delay to encourage concurrency
            except Exception as e:
                errors.append(str(e))
        
        # Run concurrent workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(add_memory_worker, i) for i in range(3)]
            concurrent.futures.wait(futures)
        
        # Check results
        assert len(errors) == 0, f"Concurrent errors occurred: {errors}"
        assert len(results) == 15  # 3 workers * 5 iterations each
        
        # Verify all entries were stored
        all_memories = shared_memory.get_memories()
        assert len(all_memories) >= 15

    def test_memory_limits(self, shared_memory):
        """Test behavior with memory limits and large datasets."""
        # Add many entries
        for i in range(100):
            entry = MemoryEntry(
                agent_id=f"agent_{i % 5}",  # 5 different agents
                type="observation",
                content={"index": i}
            )
            shared_memory.add_memory(entry)
        
        # Test limit enforcement
        limited_memories = shared_memory.get_memories(limit=10)
        assert len(limited_memories) == 10
        
        # Test agent-specific limits
        agent_memories = shared_memory.get_memories(agent_id="agent_0", limit=5)
        assert len(agent_memories) <= 5

    def test_goose_integration_metadata(self, shared_memory):
        """Test Goose integration metadata storage and retrieval."""
        entry = MemoryEntry(
            agent_id="code_agent",
            type="result",
            content={"refactoring": "completed"},
            goose_patch={
                "tool_calls": [
                    {
                        "command": "refactor",
                        "working_dir": "/project",
                        "files_modified": ["main.py", "utils.py"]
                    }
                ],
                "session_id": "goose-session-123",
                "export_format": "markdown"
            },
            ast_summary={
                "functions_modified": ["main", "helper"],
                "complexity_change": -2,
                "test_coverage_impact": "+5%"
            },
            goose_strategy="extract_function",
            file_path="/project/main.py"
        )
        
        shared_memory.add_memory(entry)
        
        # Retrieve and verify Goose metadata
        goose_memories = shared_memory.get_memories(
            agent_id="code_agent",
            type="result"
        )
        
        assert len(goose_memories) == 1
        retrieved = goose_memories[0]
        
        assert retrieved.goose_patch["session_id"] == "goose-session-123"
        assert retrieved.ast_summary["complexity_change"] == -2
        assert retrieved.goose_strategy == "extract_function"
        assert retrieved.file_path == "/project/main.py"

    def test_database_integrity(self, shared_memory):
        """Test database integrity and error handling."""
        # Test with invalid data
        entry = MemoryEntry(
            agent_id="test_agent",
            type="observation",
            content={"test": "data"}
        )
        
        # Modify entry to have invalid JSON (simulate corruption)
        with patch.object(shared_memory, '_serialize_content') as mock_serialize:
            mock_serialize.side_effect = json.JSONEncodeError("Test error", "doc", 0)
            
            success = shared_memory.add_memory(entry)
            # Should handle serialization errors gracefully
            assert success is False

    def test_database_connection_management(self, temp_db_path):
        """Test database connection lifecycle management."""
        memory = SharedMemory(db_path=temp_db_path)
        
        # Test connection is working
        assert memory._test_connection() is True
        
        # Test close
        memory.close()
        
        # Test operations after close
        entry = MemoryEntry(
            agent_id="test_agent",
            type="observation",
            content={"test": "data"}
        )
        
        # Should handle closed connection gracefully
        success = memory.add_memory(entry)
        assert success is False


# Mock-based tests for components without database dependencies
class TestSharedMemoryMocks:
    """Test shared memory components using mocks."""

    def test_mock_memory_operations(self):
        """Test memory operations with mocked database."""
        if not SHARED_MEMORY_IMPORTS_AVAILABLE:
            pytest.skip("Shared memory modules not available")
        
        with patch('sqlite3.connect') as mock_connect:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_connect.return_value.__enter__.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            
            # Mock successful operations
            mock_cursor.execute.return_value = None
            mock_cursor.fetchall.return_value = []
            
            memory = SharedMemory(db_path=":memory:")
            
            entry = MemoryEntry(
                agent_id="test_agent",
                type="observation",
                content={"test": "mock data"}
            )
            
            # This should work with mocked database
            result = memory.add_memory(entry)
            
            # Verify database calls were made
            assert mock_connect.called
            assert mock_cursor.execute.called

    def test_memory_entry_serialization(self):
        """Test memory entry serialization without database.""" 
        if not SHARED_MEMORY_IMPORTS_AVAILABLE:
            pytest.skip("Shared memory modules not available")
        
        entry = MemoryEntry(
            agent_id="test_agent",
            type="observation",
            content={"message": "test"},
            tags={"tag1", "tag2"}
        )
        
        # Test JSON serialization
        entry_json = entry.model_dump_json()
        assert isinstance(entry_json, str)
        
        # Test deserialization
        parsed = json.loads(entry_json)
        assert parsed["agent_id"] == "test_agent"
        assert parsed["type"] == "observation"

    def test_batch_memory_operations(self):
        """Test batch memory operations with mocks."""
        if not SHARED_MEMORY_IMPORTS_AVAILABLE:
            pytest.skip("Shared memory modules not available")
        
        entries = [
            MemoryEntry(
                agent_id=f"agent_{i}",
                type="observation",
                content={"index": i}
            )
            for i in range(10)
        ]
        
        # Test batch validation
        for entry in entries:
            assert entry.agent_id.startswith("agent_")
            assert entry.type == "observation"
            assert "index" in entry.content


if __name__ == "__main__":
    if SHARED_MEMORY_IMPORTS_AVAILABLE:
        pytest.main([__file__, "-v", "-x"])
    else:
        print("Skipping shared memory tests due to import errors")
        exit(1)
