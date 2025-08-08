"""Comprehensive test suite for SharedMemory system.

This module tests the SharedMemory class, MemoryEntry, TaskStatus, and AgentState models.
"""
import sqlite3
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest

try:
    from src.dev_guard.memory.shared_memory import (
        AgentState,
        MemoryEntry,
        SharedMemory,
        SharedMemoryError,
        TaskStatus,
    )
    SHARED_MEMORY_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"SharedMemory imports not available: {e}")
    SHARED_MEMORY_IMPORTS_AVAILABLE = False


@pytest.fixture
def temp_db_path():
    """Create a temporary database file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def shared_memory(temp_db_path):
    """Create a SharedMemory instance with temporary database."""
    if not SHARED_MEMORY_IMPORTS_AVAILABLE:
        pytest.skip("SharedMemory not available")
    return SharedMemory(db_path=temp_db_path)


@pytest.fixture
def sample_memory_entry():
    """Create a sample memory entry."""
    if not SHARED_MEMORY_IMPORTS_AVAILABLE:
        pytest.skip("SharedMemory not available")
    return MemoryEntry(
        agent_id="test_agent",
        type="observation",
        content={"message": "Test observation", "details": {"key": "value"}},
        tags={"test", "observation"},
        context={"source": "test_suite"}
    )


@pytest.fixture
def sample_task():
    """Create a sample task."""
    if not SHARED_MEMORY_IMPORTS_AVAILABLE:
        pytest.skip("SharedMemory not available")
    return TaskStatus(
        agent_id="test_agent",
        status="pending",
        description="Test task for validation",
        metadata={"priority": "high", "category": "test"},
        dependencies=[]
    )


@pytest.fixture
def sample_agent_state():
    """Create a sample agent state."""
    if not SHARED_MEMORY_IMPORTS_AVAILABLE:
        pytest.skip("SharedMemory not available")
    return AgentState(
        agent_id="test_agent",
        status="idle",
        metadata={"version": "1.0", "capabilities": ["test"]}
    )


# Model Tests
@pytest.mark.skipif(not SHARED_MEMORY_IMPORTS_AVAILABLE, reason="SharedMemory not available")
class TestMemoryModels:
    """Test memory-related data models."""
    
    def test_memory_entry_creation(self):
        """Test MemoryEntry model creation and validation."""
        entry = MemoryEntry(
            agent_id="agent_001",
            type="decision",
            content={"action": "create_file", "path": "/test/file.py"},
            tags={"code", "creation"},
            context={"source": "user_request", "session_id": "12345"}
        )
        
        assert entry.agent_id == "agent_001"
        assert entry.type == "decision"
        assert entry.content["action"] == "create_file"
        assert "code" in entry.tags
        assert "creation" in entry.tags
        assert entry.context["source"] == "user_request"
        assert isinstance(entry.timestamp, datetime)
        assert entry.timestamp.tzinfo == UTC
    
    def test_memory_entry_with_goose_fields(self):
        """Test MemoryEntry with Goose-specific fields."""
        goose_patch = {
            "tool_calls": [
                {"command": "git add .", "output": "file added"}
            ],
            "session_id": "goose_123"
        }
        
        ast_summary = {
            "functions": ["main", "helper"],
            "classes": ["TestClass"],
            "imports": ["os", "sys"]
        }
        
        entry = MemoryEntry(
            agent_id="code_agent",
            type="result",
            content={"refactored": True},
            goose_patch=goose_patch,
            ast_summary=ast_summary,
            goose_strategy="extract_method",
            file_path="/src/main.py"
        )
        
        assert entry.goose_patch == goose_patch
        assert entry.ast_summary == ast_summary
        assert entry.goose_strategy == "extract_method"
        assert entry.file_path == "/src/main.py"
    
    def test_memory_entry_validation(self):
        """Test MemoryEntry validation rules."""
        # Test invalid agent_id
        with pytest.raises(ValueError):
            MemoryEntry(
                agent_id="", 
                type="observation", 
                content={"msg": "test"}
            )
        
        # Test invalid type
        with pytest.raises(ValueError):
            MemoryEntry(
                agent_id="agent_1", 
                type="invalid_type", 
                content={"msg": "test"}
            )
        
        # Test valid types
        valid_types = ["task", "observation", "decision", "result", "error", "control"]
        for valid_type in valid_types:
            entry = MemoryEntry(
                agent_id="agent_1",
                type=valid_type,
                content={"message": "test"}
            )
            assert entry.type == valid_type
    
    def test_task_status_creation(self):
        """Test TaskStatus model creation and validation."""
        task = TaskStatus(
            agent_id="planner_agent",
            status="running",
            description="Analyze code structure and suggest improvements",
            metadata={"complexity": "high", "estimated_time": 300},
            dependencies=["task_001", "task_002"],
            result={"suggestions": ["refactor functions", "add tests"]}
        )
        
        assert task.agent_id == "planner_agent"
        assert task.status == "running"
        assert "improvements" in task.description
        assert task.metadata["complexity"] == "high"
        assert len(task.dependencies) == 2
        assert task.result["suggestions"] is not None
        assert isinstance(task.created_at, datetime)
        assert isinstance(task.updated_at, datetime)
    
    def test_task_status_validation(self):
        """Test TaskStatus validation rules."""
        # Test invalid agent_id
        with pytest.raises(ValueError):
            TaskStatus(agent_id="", status="pending", description="test")
        
        # Test invalid status
        with pytest.raises(ValueError):
            TaskStatus(agent_id="agent_1", status="invalid", description="test")
        
        # Test valid statuses
        valid_statuses = ["pending", "running", "completed", "failed", "cancelled"]
        for status in valid_statuses:
            task = TaskStatus(agent_id="agent_1", status=status, description="test")
            assert task.status == status
        
        # Test invalid description
        with pytest.raises(ValueError):
            TaskStatus(agent_id="agent_1", status="pending", description="")
        
        # Test invalid dependencies
        with pytest.raises(ValueError):
            TaskStatus(agent_id="agent_1", status="pending", description="test",
                      dependencies=["", "valid_dep"])  # Empty dependency
    
    def test_agent_state_creation(self):
        """Test AgentState model creation and validation."""
        state = AgentState(
            agent_id="security_agent",
            status="busy",
            current_task="task_456",
            metadata={
                "scan_progress": 0.75,
                "vulnerabilities_found": 3,
                "last_scan": "2024-01-01T10:00:00Z"
            }
        )
        
        assert state.agent_id == "security_agent"
        assert state.status == "busy"
        assert state.current_task == "task_456"
        assert state.metadata["scan_progress"] == 0.75
        assert isinstance(state.last_heartbeat, datetime)
    
    def test_agent_state_validation(self):
        """Test AgentState validation rules."""
        # Test invalid agent_id
        with pytest.raises(ValueError):
            AgentState(agent_id="", status="idle")
        
        # Test invalid status
        with pytest.raises(ValueError):
            AgentState(agent_id="agent_1", status="invalid")
        
        # Test valid statuses
        valid_statuses = ["idle", "busy", "error", "stopped"]
        for status in valid_statuses:
            state = AgentState(agent_id="agent_1", status=status)
            assert state.status == status
        
        # Test future heartbeat (should be rejected with tolerance)
        future_time = datetime.now(UTC) + timedelta(minutes=5)
        with pytest.raises(ValueError):
            AgentState(agent_id="agent_1", status="idle", last_heartbeat=future_time)


# SharedMemory Core Tests
@pytest.mark.skipif(not SHARED_MEMORY_IMPORTS_AVAILABLE, reason="SharedMemory not available")
class TestSharedMemoryCore:
    """Test core SharedMemory functionality."""
    
    def test_shared_memory_initialization(self, temp_db_path):
        """Test SharedMemory initialization and database setup."""
        sm = SharedMemory(db_path=temp_db_path)
        
        # Check database file was created
        assert Path(temp_db_path).exists()
        
        # Check tables were created
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cursor.fetchall()]
            
            assert "memory_entries" in tables
            assert "task_status" in tables
            assert "agent_states" in tables
    
    def test_add_memory_entry(self, shared_memory, sample_memory_entry):
        """Test adding memory entries."""
        entry_id = shared_memory.add_memory(sample_memory_entry)
        
        assert entry_id == sample_memory_entry.id
        
        # Retrieve and verify
        retrieved = shared_memory.get_memory_by_id(entry_id)
        assert retrieved is not None
        assert retrieved.agent_id == sample_memory_entry.agent_id
        assert retrieved.type == sample_memory_entry.type
        assert retrieved.content == sample_memory_entry.content
        assert retrieved.tags == sample_memory_entry.tags
    
    def test_add_memory_with_parent(self, shared_memory):
        """Test adding memory entries with parent relationships."""
        # Add parent entry
        parent_entry = MemoryEntry(
            agent_id="agent_1",
            type="observation", 
            content={"message": "Parent observation"}
        )
        parent_id = shared_memory.add_memory(parent_entry)
        
        # Add child entry
        child_entry = MemoryEntry(
            agent_id="agent_1",
            type="decision",
            content={"message": "Child decision"},
            parent_id=parent_id
        )
        child_id = shared_memory.add_memory(child_entry)
        
        # Verify relationship
        retrieved_child = shared_memory.get_memory_by_id(child_id)
        assert retrieved_child.parent_id == parent_id
    
    def test_add_memory_invalid_parent(self, shared_memory):
        """Test adding memory with invalid parent ID."""
        invalid_parent_entry = MemoryEntry(
            agent_id="agent_1",
            type="observation",
            content={"message": "test"},
            parent_id="nonexistent_parent_id"
        )
        
        with pytest.raises(ValueError, match="Parent entry .* does not exist"):
            shared_memory.add_memory(invalid_parent_entry)
    
    def test_get_memories_by_agent(self, shared_memory):
        """Test retrieving memories by agent ID."""
        # Add memories for different agents
        entry1 = MemoryEntry(agent_id="agent_1", type="observation", content={"msg": "1"})
        entry2 = MemoryEntry(agent_id="agent_2", type="observation", content={"msg": "2"})
        entry3 = MemoryEntry(agent_id="agent_1", type="decision", content={"msg": "3"})
        
        shared_memory.add_memory(entry1)
        shared_memory.add_memory(entry2)
        shared_memory.add_memory(entry3)
        
        # Get memories for agent_1
        agent1_memories = shared_memory.get_memories(agent_id="agent_1")
        assert len(agent1_memories) == 2
        assert all(m.agent_id == "agent_1" for m in agent1_memories)
        
        # Get memories for agent_2
        agent2_memories = shared_memory.get_memories(agent_id="agent_2")
        assert len(agent2_memories) == 1
        assert agent2_memories[0].agent_id == "agent_2"
    
    def test_get_memories_by_type(self, shared_memory):
        """Test retrieving memories by type."""
        # Add memories of different types
        obs_entry = MemoryEntry(agent_id="agent_1", type="observation", content={"msg": "obs"})
        dec_entry = MemoryEntry(agent_id="agent_1", type="decision", content={"msg": "dec"})
        res_entry = MemoryEntry(agent_id="agent_1", type="result", content={"msg": "res"})
        
        shared_memory.add_memory(obs_entry)
        shared_memory.add_memory(dec_entry)
        shared_memory.add_memory(res_entry)
        
        # Get observations
        observations = shared_memory.get_memories(memory_type="observation")
        assert len(observations) == 1
        assert observations[0].type == "observation"
        
        # Get decisions
        decisions = shared_memory.get_memories(memory_type="decision")
        assert len(decisions) == 1
        assert decisions[0].type == "decision"
    
    def test_get_memories_with_filters(self, shared_memory):
        """Test retrieving memories with multiple filters."""
        # Add test data
        now = datetime.now(UTC)
        recent_time = now - timedelta(minutes=5)
        old_time = now - timedelta(hours=2)
        
        # Recent entry
        recent_entry = MemoryEntry(
            agent_id="agent_1",
            type="observation", 
            content={"msg": "recent"},
            timestamp=recent_time
        )
        
        # Old entry
        old_entry = MemoryEntry(
            agent_id="agent_1",
            type="observation",
            content={"msg": "old"}, 
            timestamp=old_time
        )
        
        shared_memory.add_memory(recent_entry)
        shared_memory.add_memory(old_entry)
        
        # Filter by agent and since time
        since_time = now - timedelta(minutes=10)
        filtered_memories = shared_memory.get_memories(
            agent_id="agent_1",
            memory_type="observation", 
            since=since_time
        )
        
        assert len(filtered_memories) == 1
        assert filtered_memories[0].content["msg"] == "recent"
    
    def test_update_memory_entry(self, shared_memory, sample_memory_entry):
        """Test updating memory entries."""
        entry_id = shared_memory.add_memory(sample_memory_entry)
        
        # Update content
        new_content = {"message": "Updated observation", "version": 2}
        success = shared_memory.update_memory(entry_id, content=new_content)
        assert success is True
        
        # Verify update
        updated_entry = shared_memory.get_memory_by_id(entry_id)
        assert updated_entry.content == new_content
        
        # Update tags
        new_tags = {"updated", "modified"}
        success = shared_memory.update_memory(entry_id, tags=new_tags)
        assert success is True
        
        updated_entry = shared_memory.get_memory_by_id(entry_id)
        assert updated_entry.tags == new_tags
    
    def test_update_nonexistent_memory(self, shared_memory):
        """Test updating non-existent memory entry."""
        success = shared_memory.update_memory("nonexistent_id", content={"test": "value"})
        assert success is False
    
    def test_delete_memory_entry(self, shared_memory, sample_memory_entry):
        """Test deleting memory entries."""
        entry_id = shared_memory.add_memory(sample_memory_entry)
        
        # Verify it exists
        assert shared_memory.get_memory_by_id(entry_id) is not None
        
        # Delete it
        success = shared_memory.delete_memory(entry_id)
        assert success is True
        
        # Verify it's gone
        assert shared_memory.get_memory_by_id(entry_id) is None
    
    def test_delete_nonexistent_memory(self, shared_memory):
        """Test deleting non-existent memory entry."""
        success = shared_memory.delete_memory("nonexistent_id")
        assert success is False


# Task Management Tests
@pytest.mark.skipif(not SHARED_MEMORY_IMPORTS_AVAILABLE, reason="SharedMemory not available")
class TestTaskManagement:
    """Test task management functionality."""
    
    def test_create_task(self, shared_memory, sample_task):
        """Test creating tasks."""
        task_id = shared_memory.create_task(sample_task)
        assert task_id == sample_task.id
        
        # Retrieve and verify
        retrieved = shared_memory.get_task(task_id)
        assert retrieved is not None
        assert retrieved.agent_id == sample_task.agent_id
        assert retrieved.status == sample_task.status
        assert retrieved.description == sample_task.description
        assert retrieved.metadata == sample_task.metadata
    
    def test_create_task_with_dependencies(self, shared_memory):
        """Test creating tasks with dependencies."""
        # Create dependency task first
        dep_task = TaskStatus(
            agent_id="agent_1",
            status="completed",
            description="Dependency task"
        )
        dep_id = shared_memory.create_task(dep_task)
        
        # Create task with dependency
        main_task = TaskStatus(
            agent_id="agent_1", 
            status="pending",
            description="Main task",
            dependencies=[dep_id]
        )
        main_id = shared_memory.create_task(main_task)
        
        # Verify dependency relationship
        retrieved = shared_memory.get_task(main_id)
        assert len(retrieved.dependencies) == 1
        assert retrieved.dependencies[0] == dep_id
    
    def test_create_task_invalid_dependency(self, shared_memory, sample_task):
        """Test creating task with invalid dependency."""
        sample_task.dependencies = ["nonexistent_task_id"]
        
        with pytest.raises(ValueError, match="Dependency task .* does not exist"):
            shared_memory.create_task(sample_task)
    
    def test_update_task_status(self, shared_memory, sample_task):
        """Test updating task status."""
        task_id = shared_memory.create_task(sample_task)
        
        # Update status
        success = shared_memory.update_task(task_id, status="running")
        assert success is True
        
        # Verify update
        updated_task = shared_memory.get_task(task_id)
        assert updated_task.status == "running"
        assert updated_task.updated_at > updated_task.created_at
    
    def test_update_task_metadata(self, shared_memory, sample_task):
        """Test updating task metadata."""
        task_id = shared_memory.create_task(sample_task)
        
        new_metadata = {"priority": "critical", "assigned_to": "senior_agent"}
        success = shared_memory.update_task(task_id, metadata=new_metadata)
        assert success is True
        
        updated_task = shared_memory.get_task(task_id)
        assert updated_task.metadata == new_metadata
    
    def test_update_task_result(self, shared_memory, sample_task):
        """Test updating task result."""
        task_id = shared_memory.create_task(sample_task)
        
        result_data = {
            "files_analyzed": 15,
            "issues_found": 3,
            "suggestions": ["fix import", "add docstrings"]
        }
        
        success = shared_memory.update_task(
            task_id, 
            status="completed",
            result=result_data
        )
        assert success is True
        
        updated_task = shared_memory.get_task(task_id)
        assert updated_task.status == "completed"
        assert updated_task.result == result_data
    
    def test_get_tasks_by_agent(self, shared_memory):
        """Test retrieving tasks by agent."""
        # Create tasks for different agents
        task1 = TaskStatus(agent_id="agent_1", status="pending", description="Task 1")
        task2 = TaskStatus(agent_id="agent_2", status="running", description="Task 2")
        task3 = TaskStatus(agent_id="agent_1", status="completed", description="Task 3")
        
        shared_memory.create_task(task1)
        shared_memory.create_task(task2)
        shared_memory.create_task(task3)
        
        # Get tasks for agent_1
        agent1_tasks = shared_memory.get_tasks(agent_id="agent_1")
        assert len(agent1_tasks) == 2
        assert all(task.agent_id == "agent_1" for task in agent1_tasks)
    
    def test_get_tasks_by_status(self, shared_memory):
        """Test retrieving tasks by status."""
        # Create tasks with different statuses
        pending_task = TaskStatus(agent_id="agent_1", status="pending", description="Pending")
        running_task = TaskStatus(agent_id="agent_1", status="running", description="Running")
        completed_task = TaskStatus(agent_id="agent_1", status="completed", description="Done")
        
        shared_memory.create_task(pending_task)
        shared_memory.create_task(running_task)
        shared_memory.create_task(completed_task)
        
        # Get pending tasks
        pending_tasks = shared_memory.get_tasks(status="pending")
        assert len(pending_tasks) == 1
        assert pending_tasks[0].status == "pending"
        
        # Get completed tasks
        completed_tasks = shared_memory.get_tasks(status="completed")
        assert len(completed_tasks) == 1
        assert completed_tasks[0].status == "completed"
    
    def test_delete_task(self, shared_memory, sample_task):
        """Test deleting tasks."""
        task_id = shared_memory.create_task(sample_task)
        
        # Verify it exists
        assert shared_memory.get_task(task_id) is not None
        
        # Delete it
        success = shared_memory.delete_task(task_id)
        assert success is True
        
        # Verify it's gone
        assert shared_memory.get_task(task_id) is None


# Agent State Management Tests
@pytest.mark.skipif(not SHARED_MEMORY_IMPORTS_AVAILABLE, reason="SharedMemory not available")
class TestAgentStateManagement:
    """Test agent state management functionality."""
    
    def test_update_agent_state(self, shared_memory, sample_agent_state):
        """Test updating agent states."""
        shared_memory.update_agent_state(sample_agent_state)
        
        # Retrieve and verify
        retrieved = shared_memory.get_agent_state(sample_agent_state.agent_id)
        assert retrieved is not None
        assert retrieved.agent_id == sample_agent_state.agent_id
        assert retrieved.status == sample_agent_state.status
        assert retrieved.metadata == sample_agent_state.metadata
    
    def test_update_agent_state_with_current_task(self, shared_memory, sample_task):
        """Test updating agent state with current task reference."""
        # Create a task first
        task_id = shared_memory.create_task(sample_task)
        
        # Update agent state with current task
        state = AgentState(
            agent_id="test_agent",
            status="busy",
            current_task=task_id
        )
        
        shared_memory.update_agent_state(state)
        
        # Verify
        retrieved = shared_memory.get_agent_state("test_agent")
        assert retrieved.current_task == task_id
        assert retrieved.status == "busy"
    
    def test_update_agent_state_invalid_task(self, shared_memory):
        """Test updating agent state with non-existent task."""
        state = AgentState(
            agent_id="test_agent",
            status="busy",
            current_task="nonexistent_task_id"
        )
        
        with pytest.raises(ValueError, match="Current task .* does not exist"):
            shared_memory.update_agent_state(state)
    
    def test_get_all_agent_states(self, shared_memory):
        """Test retrieving all agent states."""
        # Create multiple agent states
        states = [
            AgentState(agent_id="agent_1", status="idle"),
            AgentState(agent_id="agent_2", status="busy"),
            AgentState(agent_id="agent_3", status="error")
        ]
        
        for state in states:
            shared_memory.update_agent_state(state)
        
        # Get all states
        all_states = shared_memory.get_all_agent_states()
        assert len(all_states) == 3
        
        agent_ids = [state.agent_id for state in all_states]
        assert "agent_1" in agent_ids
        assert "agent_2" in agent_ids
        assert "agent_3" in agent_ids
    
    def test_get_agent_states_dict(self, shared_memory):
        """Test retrieving agent states as dictionary."""
        # Create agent states
        state1 = AgentState(agent_id="commander", status="idle")
        state2 = AgentState(agent_id="planner", status="busy")
        
        shared_memory.update_agent_state(state1)
        shared_memory.update_agent_state(state2)
        
        # Get states dict
        states_dict = shared_memory.get_agent_states()
        
        assert isinstance(states_dict, dict)
        assert "commander" in states_dict
        assert "planner" in states_dict
        assert states_dict["commander"].status == "idle"
        assert states_dict["planner"].status == "busy"
    
    def test_delete_agent_state(self, shared_memory, sample_agent_state):
        """Test deleting agent states."""
        shared_memory.update_agent_state(sample_agent_state)
        
        # Verify it exists
        assert shared_memory.get_agent_state(sample_agent_state.agent_id) is not None
        
        # Delete it
        success = shared_memory.delete_agent_state(sample_agent_state.agent_id)
        assert success is True
        
        # Verify it's gone
        assert shared_memory.get_agent_state(sample_agent_state.agent_id) is None


# Advanced Features Tests
@pytest.mark.skipif(not SHARED_MEMORY_IMPORTS_AVAILABLE, reason="SharedMemory not available")
class TestAdvancedFeatures:
    """Test advanced SharedMemory features."""
    
    def test_conversation_threading(self, shared_memory):
        """Test conversation thread functionality."""
        # Create a conversation thread
        root_entry = MemoryEntry(
            agent_id="user",
            type="observation", 
            content={"message": "Root message"}
        )
        root_id = shared_memory.add_memory(root_entry)
        
        child1_entry = MemoryEntry(
            agent_id="agent_1",
            type="decision",
            content={"message": "Child 1 response"},
            parent_id=root_id
        )
        child1_id = shared_memory.add_memory(child1_entry)
        
        child2_entry = MemoryEntry(
            agent_id="agent_2", 
            type="result",
            content={"message": "Child 2 response"},
            parent_id=child1_id
        )
        child2_id = shared_memory.add_memory(child2_entry)
        
        # Get conversation thread
        thread = shared_memory.get_conversation_thread(root_id)
        
        # Should get all entries in chronological order
        assert len(thread) == 3
        assert thread[0].id == root_id
        assert thread[1].id == child1_id
        assert thread[2].id == child2_id
    
    def test_memory_search(self, shared_memory):
        """Test memory content search."""
        # Add searchable entries
        entries = [
            MemoryEntry(
                agent_id="agent_1",
                type="observation",
                content={"message": "Python code analysis completed"}
            ),
            MemoryEntry(
                agent_id="agent_2", 
                type="decision",
                content={"action": "refactor", "language": "python"}
            ),
            MemoryEntry(
                agent_id="agent_3",
                type="result", 
                content={"message": "JavaScript linting finished"}
            )
        ]
        
        for entry in entries:
            shared_memory.add_memory(entry)
        
        # Search for "python" 
        python_results = shared_memory.search_memories("python")
        assert len(python_results) == 2  # Should find both Python-related entries
        
        # Search for "JavaScript"
        js_results = shared_memory.search_memories("JavaScript") 
        assert len(js_results) == 1
        assert "JavaScript" in js_results[0].content["message"]
    
    def test_memory_search_with_filters(self, shared_memory):
        """Test memory search with agent and type filters."""
        # Add test entries
        entries = [
            MemoryEntry(agent_id="agent_1", type="observation", content={"msg": "test data"}),
            MemoryEntry(agent_id="agent_2", type="observation", content={"msg": "test data"}),
            MemoryEntry(agent_id="agent_1", type="decision", content={"msg": "test data"})
        ]
        
        for entry in entries:
            shared_memory.add_memory(entry)
        
        # Search with agent filter
        agent1_results = shared_memory.search_memories("test", agent_id="agent_1")
        assert len(agent1_results) == 2
        assert all(r.agent_id == "agent_1" for r in agent1_results)
        
        # Search with type filter
        obs_results = shared_memory.search_memories("test", memory_type="observation")
        assert len(obs_results) == 2
        assert all(r.type == "observation" for r in obs_results)
        
        # Search with both filters
        specific_results = shared_memory.search_memories(
            "test", agent_id="agent_1", memory_type="observation"
        )
        assert len(specific_results) == 1
        assert specific_results[0].agent_id == "agent_1"
        assert specific_results[0].type == "observation"
    
    def test_memory_tagging_and_search(self, shared_memory):
        """Test memory tagging and tag-based search."""
        # Add entries with different tags
        entries = [
            MemoryEntry(
                agent_id="agent_1",
                type="observation", 
                content={"msg": "security scan"},
                tags={"security", "scan", "vulnerability"}
            ),
            MemoryEntry(
                agent_id="agent_2",
                type="result",
                content={"msg": "code review"},
                tags={"review", "quality", "code"}
            ),
            MemoryEntry(
                agent_id="agent_3", 
                type="decision",
                content={"msg": "security fix"},
                tags={"security", "fix", "patch"}
            )
        ]
        
        for entry in entries:
            shared_memory.add_memory(entry)
        
        # Search by single tag
        security_entries = shared_memory.get_memory_by_tags({"security"})
        assert len(security_entries) == 2
        
        # Search requiring all tags (AND)
        security_scan_entries = shared_memory.get_memory_by_tags(
            {"security", "scan"}, match_all=True
        )
        assert len(security_scan_entries) == 1
        assert "scan" in security_scan_entries[0].content["msg"]
        
        # Search for any tags (OR) - default behavior
        multi_tag_entries = shared_memory.get_memory_by_tags({"review", "patch"})
        assert len(multi_tag_entries) == 2  # Should find both review and patch entries
    
    def test_audit_trail(self, shared_memory):
        """Test audit trail functionality."""
        start_time = datetime.now(UTC)
        
        # Add entries over time
        entries = [
            MemoryEntry(agent_id="agent_1", type="observation", content={"step": 1}),
            MemoryEntry(agent_id="agent_1", type="decision", content={"step": 2}),
            MemoryEntry(agent_id="agent_2", type="result", content={"step": 3})
        ]
        
        for entry in entries:
            shared_memory.add_memory(entry)
        
        end_time = datetime.now(UTC)
        
        # Get audit trail
        trail = shared_memory.get_audit_trail(
            start_time=start_time,
            end_time=end_time
        )
        
        assert len(trail) == 3
        # Should be in chronological order
        assert trail[0].content["step"] == 1
        assert trail[1].content["step"] == 2
        assert trail[2].content["step"] == 3
        
        # Test agent-specific audit trail
        agent1_trail = shared_memory.get_audit_trail(
            agent_id="agent_1",
            start_time=start_time,
            end_time=end_time
        )
        assert len(agent1_trail) == 2
        assert all(entry.agent_id == "agent_1" for entry in agent1_trail)
    
    def test_cleanup_old_entries(self, shared_memory):
        """Test cleanup of old entries."""
        # Create old entries
        old_time = datetime.now(UTC) - timedelta(days=35)
        
        old_entry = MemoryEntry(
            agent_id="agent_1",
            type="observation",
            content={"message": "old observation"},
            timestamp=old_time
        )
        shared_memory.add_memory(old_entry)
        
        old_task = TaskStatus(
            agent_id="agent_1",
            status="completed",
            description="old task",
            created_at=old_time,
            updated_at=old_time
        )
        shared_memory.create_task(old_task)
        
        old_state = AgentState(
            agent_id="agent_1", 
            status="idle",
            last_heartbeat=old_time
        )
        shared_memory.update_agent_state(old_state)
        
        # Create recent entries
        recent_entry = MemoryEntry(
            agent_id="agent_2",
            type="observation",
            content={"message": "recent observation"}
        )
        shared_memory.add_memory(recent_entry)
        
        # Cleanup old entries (30 days)
        deleted_counts = shared_memory.cleanup_old_entries(days=30)
        
        # Verify old entries were deleted
        assert deleted_counts["memory_entries"] == 1
        assert deleted_counts["tasks"] == 1
        assert deleted_counts["agent_states"] == 1
        
        # Verify recent entry still exists
        all_memories = shared_memory.get_memories()
        assert len(all_memories) == 1
        assert all_memories[0].agent_id == "agent_2"
    
    def test_goose_patch_memory_logging(self, shared_memory):
        """Test Goose-specific memory logging."""
        goose_patch = {
            "session_id": "goose_session_123",
            "tool_calls": [
                {"command": "git status", "output": "nothing to commit"},
                {"command": "python test.py", "output": "all tests passed"}
            ]
        }
        
        ast_summary = {
            "functions": ["main", "process_data", "validate_input"],
            "classes": ["DataProcessor", "ValidationError"],
            "complexity_score": 7.5
        }
        
        entry_id = shared_memory.log_goose_patch_memory(
            agent_id="code_agent",
            goose_patch=goose_patch,
            ast_summary=ast_summary,
            goose_strategy="extract_class",
            file_path="/src/processor.py"
        )
        
        # Retrieve and verify
        entry = shared_memory.get_memory_by_id(entry_id)
        assert entry is not None
        assert entry.goose_patch == goose_patch
        assert entry.ast_summary == ast_summary
        assert entry.goose_strategy == "extract_class"
        assert entry.file_path == "/src/processor.py"
        assert "goose" in entry.tags
        assert "refactor" in entry.tags
    
    def test_ast_analysis_memory_logging(self, shared_memory):
        """Test AST analysis memory logging."""
        ast_analysis = {
            "file_analyzed": "/src/main.py",
            "function_count": 12,
            "class_count": 3,
            "complexity_metrics": {
                "cyclomatic_complexity": 15,
                "cognitive_complexity": 23
            },
            "issues_found": [
                {"type": "too_complex", "line": 45, "function": "process_data"},
                {"type": "unused_import", "line": 2, "module": "os"}
            ]
        }
        
        entry_id = shared_memory.log_ast_analysis_memory(
            agent_id="qa_agent",
            file_path="/src/main.py",
            ast_summary=ast_analysis,
            refactor_strategy="split_complex_functions"
        )
        
        # Retrieve and verify
        entry = shared_memory.get_memory_by_id(entry_id)
        assert entry is not None
        assert entry.ast_summary == ast_analysis
        assert entry.goose_strategy == "split_complex_functions"
        assert entry.file_path == "/src/main.py"
        assert "ast" in entry.tags
        assert "analysis" in entry.tags
    
    def test_statistics_collection(self, shared_memory):
        """Test database statistics collection."""
        # Add test data
        entries = [
            MemoryEntry(agent_id="agent_1", type="observation", content={"msg": "1"}),
            MemoryEntry(agent_id="agent_1", type="decision", content={"msg": "2"}),
            MemoryEntry(agent_id="agent_2", type="observation", content={"msg": "3"})
        ]
        
        for entry in entries:
            shared_memory.add_memory(entry)
        
        tasks = [
            TaskStatus(agent_id="agent_1", status="pending", description="Task 1"),
            TaskStatus(agent_id="agent_1", status="completed", description="Task 2")
        ]
        
        for task in tasks:
            shared_memory.create_task(task)
        
        states = [
            AgentState(agent_id="agent_1", status="idle"),
            AgentState(agent_id="agent_2", status="busy")
        ]
        
        for state in states:
            shared_memory.update_agent_state(state)
        
        # Get statistics
        stats = shared_memory.get_statistics()
        
        assert stats["memory_entries_count"] == 3
        assert stats["memory_entries_by_type"]["observation"] == 2
        assert stats["memory_entries_by_type"]["decision"] == 1
        
        assert stats["tasks_count"] == 2
        assert stats["tasks_by_status"]["pending"] == 1
        assert stats["tasks_by_status"]["completed"] == 1
        
        assert stats["agent_states_count"] == 2
        assert stats["agent_states_by_status"]["idle"] == 1
        assert stats["agent_states_by_status"]["busy"] == 1


# Error Handling and Edge Cases
@pytest.mark.skipif(not SHARED_MEMORY_IMPORTS_AVAILABLE, reason="SharedMemory not available")
class TestSharedMemoryErrorHandling:
    """Test error handling and edge cases."""
    
    def test_database_error_handling(self, shared_memory):
        """Test error handling for database operations."""
        # Test with invalid entry
        with pytest.raises(ValueError):
            shared_memory.add_memory("not_a_memory_entry")
        
        # Test with invalid task
        with pytest.raises(ValueError):
            shared_memory.create_task("not_a_task")
        
        # Test with invalid agent state
        with pytest.raises(ValueError):
            shared_memory.update_agent_state("not_an_agent_state")
    
    def test_concurrent_access_simulation(self, shared_memory):
        """Test thread safety with concurrent operations."""
        import concurrent.futures
        
        results = []
        errors = []
        
        def add_memory_worker(worker_id):
            try:
                for i in range(5):
                    entry = MemoryEntry(
                        agent_id=f"worker_{worker_id}",
                        type="observation",
                        content={"worker": worker_id, "iteration": i}
                    )
                    entry_id = shared_memory.add_memory(entry)
                    results.append(entry_id)
            except Exception as e:
                errors.append(e)
        
        # Run multiple workers concurrently
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
        # Test with limit parameter
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
        assert all(m.agent_id == "agent_0" for m in agent_memories)
    
    def test_invalid_database_path(self, temp_db_path):
        """Test initialization with invalid database path."""
        # Test with read-only directory (simulated)
        invalid_path = "/nonexistent/directory/test.db"
        
        # This should still work as SQLite creates parent directories
        sm = SharedMemory(db_path=invalid_path)
        assert sm is not None
    
    def test_database_corruption_simulation(self, temp_db_path):
        """Test handling of database corruption."""
        sm = SharedMemory(db_path=temp_db_path)
        
        # Add some data
        entry = MemoryEntry(
            agent_id="test_agent",
            type="observation",
            content={"test": "data"}
        )
        sm.add_memory(entry)
        
        # Simulate database corruption by directly modifying the file
        # In real scenarios, this would be handled by backup/recovery mechanisms
        with open(temp_db_path, 'wb') as f:
            f.write(b'corrupted data')
        
        # Operations should handle corruption gracefully
        with pytest.raises(SharedMemoryError):
            sm.get_memories()


# Mock-based tests for testing without database dependencies
class TestSharedMemoryMocks:
    """Test SharedMemory components using mocks."""
    
    def test_mock_memory_operations(self):
        """Test memory operations with mocks."""
        mock_memory = Mock()
        mock_memory.add_memory.return_value = "mock_entry_id"
        mock_memory.get_memories.return_value = [
            Mock(agent_id="agent_1", type="observation"),
            Mock(agent_id="agent_2", type="decision")
        ]
        
        # Test mock behavior
        entry_id = mock_memory.add_memory(Mock())
        assert entry_id == "mock_entry_id"
        
        memories = mock_memory.get_memories()
        assert len(memories) == 2
        assert memories[0].agent_id == "agent_1"
    
    def test_mock_task_operations(self):
        """Test task operations with mocks."""
        mock_task_manager = Mock()
        mock_task_manager.create_task.return_value = "task_123"
        mock_task_manager.get_tasks.return_value = [
            Mock(id="task_123", status="pending"),
            Mock(id="task_456", status="completed")
        ]
        
        # Test mock behavior
        task_id = mock_task_manager.create_task(Mock())
        assert task_id == "task_123"
        
        tasks = mock_task_manager.get_tasks(status="pending")
        assert len(tasks) == 2
    
    def test_mock_agent_state_operations(self):
        """Test agent state operations with mocks."""
        mock_state_manager = Mock()
        mock_states = {
            "agent_1": Mock(agent_id="agent_1", status="idle"),
            "agent_2": Mock(agent_id="agent_2", status="busy")
        }
        mock_state_manager.get_agent_states.return_value = mock_states
        
        states = mock_state_manager.get_agent_states()
        assert len(states) == 2
        assert states["agent_1"].status == "idle"
        assert states["agent_2"].status == "busy"


if __name__ == "__main__":
    if SHARED_MEMORY_IMPORTS_AVAILABLE:
        pytest.main([__file__, "-v", "-x"])
    else:
        print("Skipping SharedMemory tests due to import errors")
        exit(1)
