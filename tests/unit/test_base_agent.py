"""Unit tests for BaseAgent class."""

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dev_guard.agents.base_agent import BaseAgent
from dev_guard.core.config import AgentConfig, Config
from dev_guard.memory.shared_memory import AgentState, MemoryEntry, SharedMemory, TaskStatus
from src.dev_guard.memory.vector_db import VectorDatabase


class ConcreteBaseAgent(BaseAgent):
    """Test implementation of BaseAgent for testing."""
    
    async def execute(self, state):
        """Test implementation of execute method."""
        return {"status": "completed", "result": "test_result"}


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = MagicMock(spec=Config)
    agent_config = AgentConfig(
        enabled=True,
        max_retries=3,
        retry_delay=1.0,
        timeout=30.0,
        custom_instructions="Test instructions",
        priority=5,
        max_concurrent_tasks=2,
        heartbeat_interval=10.0
    )
    config.get_agent_config.return_value = agent_config
    return config


@pytest.fixture
def mock_shared_memory():
    """Mock shared memory for testing."""
    memory = MagicMock(spec=SharedMemory)
    memory.add_memory.return_value = "test-memory-id"
    memory.create_task.return_value = "test-task-id"
    memory.update_task.return_value = True
    memory.update_agent_state.return_value = None
    memory.get_agent_state.return_value = None
    memory.get_memories.return_value = []
    memory.get_tasks.return_value = []
    return memory


@pytest.fixture
def mock_vector_db():
    """Mock vector database for testing."""
    vector_db = MagicMock(spec=VectorDatabase)
    vector_db.search.return_value = [
        {"content": "test content", "metadata": {"file": "test.py"}}
    ]
    vector_db.search_files.return_value = [
        {"id": "doc1", "metadata": {"file": "test.py", "repository": "/test/repo"}}
    ]
    return vector_db


@pytest.fixture
def base_agent(mock_config, mock_shared_memory, mock_vector_db):
    """Create a test BaseAgent instance."""
    return ConcreteBaseAgent(
        agent_id="test_agent",
        config=mock_config,
        shared_memory=mock_shared_memory,
        vector_db=mock_vector_db
    )


class TestBaseAgentInitialization:
    """Test BaseAgent initialization."""
    
    def test_initialization(self, base_agent, mock_shared_memory):
        """Test that BaseAgent initializes correctly."""
        assert base_agent.agent_id == "test_agent"
        assert base_agent.config is not None
        assert base_agent.shared_memory is mock_shared_memory
        assert base_agent.vector_db is not None
        assert base_agent.logger is not None
        
        # Check that initial state was set
        mock_shared_memory.update_agent_state.assert_called_once()
        call_args = mock_shared_memory.update_agent_state.call_args[0][0]
        assert isinstance(call_args, AgentState)
        assert call_args.agent_id == "test_agent"
        assert call_args.status == "idle"
        assert call_args.current_task is None
    
    def test_agent_config_access(self, base_agent, mock_config):
        """Test that agent configuration is properly accessed."""
        mock_config.get_agent_config.assert_called_once_with("test_agent")
        assert base_agent.agent_config.enabled is True
        assert base_agent.agent_config.max_retries == 3
        assert base_agent.agent_config.timeout == 30.0


class TestBaseAgentExecution:
    """Test BaseAgent execution methods."""
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_success(self, base_agent):
        """Test successful execution without retries."""
        state = {"test": "data"}
        result = await base_agent.execute_with_retry(state)
        
        assert result == {"status": "completed", "result": "test_result"}
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_timeout(self, base_agent, mock_shared_memory):
        """Test execution with timeout."""
        # Mock execute to timeout
        async def slow_execute(state):
            await asyncio.sleep(2)
            return state
        
        base_agent.execute = slow_execute
        base_agent.agent_config.timeout = 0.1  # Very short timeout
        base_agent.agent_config.max_retries = 1
        base_agent.agent_config.retry_delay = 0.1
        
        state = {"test": "data"}
        
        with pytest.raises(asyncio.TimeoutError):
            await base_agent.execute_with_retry(state)
        
        # Check that error was logged
        assert mock_shared_memory.add_memory.called
        error_entry = mock_shared_memory.add_memory.call_args[0][0]
        assert error_entry.type == "error"
        assert "timed out" in error_entry.content["error"]
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_exception(self, base_agent, mock_shared_memory):
        """Test execution with exception and retries."""
        # Mock execute to raise exception
        async def failing_execute(state):
            raise ValueError("Test error")
        
        base_agent.execute = failing_execute
        base_agent.agent_config.max_retries = 2
        base_agent.agent_config.retry_delay = 0.1
        
        state = {"test": "data"}
        
        with pytest.raises(ValueError):
            await base_agent.execute_with_retry(state)
        
        # Check that error was logged
        assert mock_shared_memory.add_memory.called
        error_entry = mock_shared_memory.add_memory.call_args[0][0]
        assert error_entry.type == "error"
        assert "Test error" in error_entry.content["error"]
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_success_after_failure(self, base_agent):
        """Test successful execution after initial failures."""
        call_count = 0
        
        async def intermittent_execute(state):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return {"status": "completed", "result": "success"}
        
        base_agent.execute = intermittent_execute
        base_agent.agent_config.max_retries = 3
        base_agent.agent_config.retry_delay = 0.1
        
        state = {"test": "data"}
        result = await base_agent.execute_with_retry(state)
        
        assert result == {"status": "completed", "result": "success"}
        assert call_count == 3


class TestBaseAgentMemoryLogging:
    """Test BaseAgent memory logging methods."""
    
    def test_log_observation(self, base_agent, mock_shared_memory):
        """Test logging observations."""
        observation = "Test observation"
        data = {"key": "value"}
        tags = ["test", "observation"]
        
        entry_id = base_agent.log_observation(observation, data, tags)
        
        assert entry_id == "test-memory-id"
        mock_shared_memory.add_memory.assert_called_once()
        
        entry = mock_shared_memory.add_memory.call_args[0][0]
        assert isinstance(entry, MemoryEntry)
        assert entry.agent_id == "test_agent"
        assert entry.type == "observation"
        assert entry.content["observation"] == observation
        assert entry.content["data"] == data
        assert "test" in entry.tags
        assert "observation" in entry.tags
        assert "test_agent" in entry.tags
    
    def test_log_decision(self, base_agent, mock_shared_memory):
        """Test logging decisions."""
        decision = "Test decision"
        reasoning = "Test reasoning"
        data = {"context": "test"}
        tags = ["decision"]
        
        entry_id = base_agent.log_decision(decision, reasoning, data, tags)
        
        assert entry_id == "test-memory-id"
        mock_shared_memory.add_memory.assert_called_once()
        
        entry = mock_shared_memory.add_memory.call_args[0][0]
        assert isinstance(entry, MemoryEntry)
        assert entry.agent_id == "test_agent"
        assert entry.type == "decision"
        assert entry.content["decision"] == decision
        assert entry.content["reasoning"] == reasoning
        assert entry.content["data"] == data
        assert "decision" in entry.tags
        assert "test_agent" in entry.tags
    
    def test_log_result(self, base_agent, mock_shared_memory):
        """Test logging results."""
        result = "Test result"
        data = {"output": "test"}
        tags = ["result"]
        parent_id = "12345678-1234-1234-1234-123456789012"
        
        entry_id = base_agent.log_result(result, data, tags, parent_id)
        
        assert entry_id == "test-memory-id"
        mock_shared_memory.add_memory.assert_called_once()
        
        entry = mock_shared_memory.add_memory.call_args[0][0]
        assert isinstance(entry, MemoryEntry)
        assert entry.agent_id == "test_agent"
        assert entry.type == "result"
        assert entry.content["result"] == result
        assert entry.content["data"] == data
        assert entry.parent_id == parent_id
        assert "result" in entry.tags
        assert "test_agent" in entry.tags
    
    def test_get_recent_memories(self, base_agent, mock_shared_memory):
        """Test getting recent memories."""
        mock_memories = [
            MemoryEntry(
                agent_id="test_agent",
                type="observation",
                content={"observation": "test"}
            )
        ]
        mock_shared_memory.get_memories.return_value = mock_memories
        
        memories = base_agent.get_recent_memories(memory_type="observation", limit=10)
        
        assert memories == mock_memories
        mock_shared_memory.get_memories.assert_called_once_with(
            agent_id="test_agent",
            memory_type="observation",
            limit=10
        )
    
    def test_get_recent_memories_include_other_agents(self, base_agent, mock_shared_memory):
        """Test getting memories from all agents."""
        mock_memories = [
            MemoryEntry(
                agent_id="other_agent",
                type="decision",
                content={"decision": "test"}
            )
        ]
        mock_shared_memory.get_memories.return_value = mock_memories
        
        memories = base_agent.get_recent_memories(
            memory_type="decision",
            limit=20,
            include_other_agents=True
        )
        
        assert memories == mock_memories
        mock_shared_memory.get_memories.assert_called_once_with(
            agent_id=None,
            memory_type="decision",
            limit=20
        )


class TestBaseAgentKnowledgeSearch:
    """Test BaseAgent knowledge search methods."""
    
    def test_search_knowledge(self, base_agent, mock_vector_db):
        """Test searching knowledge in vector database."""
        query = "test query"
        filters = {"file_type": "python"}
        
        results = base_agent.search_knowledge(query, n_results=5, filters=filters)
        
        assert len(results) == 1
        assert results[0]["content"] == "test content"
        mock_vector_db.search.assert_called_once_with(
            query, n_results=5, where=filters
        )
    
    def test_search_code(self, base_agent, mock_vector_db):
        """Test searching code in vector database."""
        query = "function definition"
        repo_path = "/test/repo"
        file_extensions = [".py", ".js"]
        
        results = base_agent.search_code(
            query, repo_path=repo_path, file_extensions=file_extensions, n_results=3
        )
        
        assert len(results) == 1
        mock_vector_db.search.assert_called_once_with(
            query,
            n_results=3,
            where={
                "content_type": "code",
                "repository": repo_path,
                "file_extension": {"$in": file_extensions}
            }
        )
    
    def test_get_repository_files(self, base_agent, mock_vector_db):
        """Test getting repository files."""
        repo_path = "/test/repo"
        file_extensions = [".py"]
        
        # Mock search_files to return files with repository metadata
        mock_vector_db.search_files.return_value = [
            {"id": "doc1", "metadata": {"file": "test.py", "repository": "/test/repo"}},
            {"id": "doc2", "metadata": {"file": "other.py", "repository": "/other/repo"}}
        ]
        
        files = base_agent.get_repository_files(repo_path, file_extensions)
        
        assert len(files) == 1
        assert files[0]["id"] == "doc1"
        assert files[0]["metadata"]["repository"] == repo_path
        mock_vector_db.search_files.assert_called_once_with(
            query="",
            file_extensions=file_extensions,
            n_results=1000
        )


class TestBaseAgentStateManagement:
    """Test BaseAgent state management methods."""
    
    def test_update_heartbeat(self, base_agent, mock_shared_memory):
        """Test updating agent heartbeat."""
        old_time = datetime.now(UTC) - timedelta(minutes=1)
        mock_state = AgentState(
            agent_id="test_agent",
            status="busy",
            last_heartbeat=old_time,
            metadata={"old": "data"}
        )
        mock_shared_memory.get_agent_state.return_value = mock_state
        
        metadata = {"new": "data"}
        base_agent.update_heartbeat(metadata)
        
        mock_shared_memory.get_agent_state.assert_called_once_with("test_agent")
        # Called twice: once during init, once in update_heartbeat
        assert mock_shared_memory.update_agent_state.call_count == 2
        
        updated_state = mock_shared_memory.update_agent_state.call_args[0][0]
        assert updated_state.agent_id == "test_agent"
        assert updated_state.metadata["old"] == "data"
        assert updated_state.metadata["new"] == "data"
        assert updated_state.last_heartbeat > old_time
    
    def test_update_heartbeat_no_existing_state(self, base_agent, mock_shared_memory):
        """Test updating heartbeat when no existing state."""
        mock_shared_memory.get_agent_state.return_value = None
        
        base_agent.update_heartbeat()
        
        # Should not call update_agent_state if no existing state
        mock_shared_memory.get_agent_state.assert_called_once_with("test_agent")
        # update_agent_state is called once during initialization, not again
        assert mock_shared_memory.update_agent_state.call_count == 1
    
    def test_set_status(self, base_agent, mock_shared_memory):
        """Test setting agent status."""
        status = "busy"
        current_task = "12345678-1234-1234-1234-123456789012"  # Valid UUID format
        
        base_agent.set_status(status, current_task)
        
        # Called once during init, once in set_status
        assert mock_shared_memory.update_agent_state.call_count == 2
        
        updated_state = mock_shared_memory.update_agent_state.call_args[0][0]
        assert isinstance(updated_state, AgentState)
        assert updated_state.agent_id == "test_agent"
        assert updated_state.status == status
        assert updated_state.current_task == current_task


class TestBaseAgentTaskManagement:
    """Test BaseAgent task management methods."""
    
    def test_create_task(self, base_agent, mock_shared_memory):
        """Test creating a task."""
        description = "Test task"
        task_type = "code_generation"
        target_agent = "code_agent"
        metadata = {"priority": "high"}
        
        task_id = base_agent.create_task(description, task_type, target_agent, metadata)
        
        assert task_id == "test-task-id"
        mock_shared_memory.create_task.assert_called_once()
        
        task = mock_shared_memory.create_task.call_args[0][0]
        assert isinstance(task, TaskStatus)
        assert task.agent_id == target_agent
        assert task.status == "pending"
        assert task.description == description
        assert task.metadata["type"] == task_type
        assert task.metadata["created_by"] == "test_agent"
        assert task.metadata["priority"] == "high"
        
        # Check that task creation was logged
        mock_shared_memory.add_memory.assert_called()
        log_entry = mock_shared_memory.add_memory.call_args[0][0]
        assert log_entry.type == "observation"
        assert "Created task" in log_entry.content["observation"]
    
    def test_create_task_default_target(self, base_agent, mock_shared_memory):
        """Test creating task with default target agent."""
        description = "Test task"
        task_type = "planning"
        
        task_id = base_agent.create_task(description, task_type)
        
        task = mock_shared_memory.create_task.call_args[0][0]
        assert task.agent_id == "planner"  # Default target
    
    def test_get_pending_tasks(self, base_agent, mock_shared_memory):
        """Test getting pending tasks."""
        mock_tasks = [
            TaskStatus(
                agent_id="test_agent",
                status="pending",
                description="Test task"
            )
        ]
        mock_shared_memory.get_tasks.return_value = mock_tasks
        
        tasks = base_agent.get_pending_tasks()
        
        assert tasks == mock_tasks
        mock_shared_memory.get_tasks.assert_called_once_with(
            agent_id="test_agent", status="pending"
        )
    
    def test_update_task_status(self, base_agent, mock_shared_memory):
        """Test updating task status."""
        task_id = "12345678-1234-1234-1234-123456789012"  # Valid UUID
        status = "completed"
        result = {"output": "success"}
        error = None
        
        # Reset call count to ignore initialization calls
        mock_shared_memory.add_memory.reset_mock()
        
        success = base_agent.update_task_status(task_id, status, result, error)
        
        assert success is True
        # Only non-None values are passed to update_task
        mock_shared_memory.update_task.assert_called_once_with(
            task_id, status=status, result=result
        )
        
        # Check that update was logged
        mock_shared_memory.add_memory.assert_called_once()
        log_entry = mock_shared_memory.add_memory.call_args[0][0]
        assert log_entry.type == "observation"
        assert "Updated task" in log_entry.content["observation"]
    
    def test_update_task_status_failure(self, base_agent, mock_shared_memory):
        """Test updating task status when update fails."""
        mock_shared_memory.update_task.return_value = False
        
        # Reset call count to ignore initialization calls
        mock_shared_memory.add_memory.reset_mock()
        
        task_id = "12345678-1234-1234-1234-123456789012"  # Valid UUID
        success = base_agent.update_task_status(task_id, "failed")
        
        assert success is False
        # Should not log if update failed
        mock_shared_memory.add_memory.assert_not_called()


class TestBaseAgentCommandExecution:
    """Test BaseAgent command execution methods."""
    
    @pytest.mark.asyncio
    async def test_execute_command_success(self, base_agent, mock_shared_memory):
        """Test successful command execution."""
        with patch('asyncio.create_subprocess_shell') as mock_subprocess:
            # Mock successful process
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate.return_value = (b"output", b"")
            mock_subprocess.return_value = mock_process
            
            result = await base_agent.execute_command("echo test")
            
            assert result["command"] == "echo test"
            assert result["return_code"] == 0
            assert result["stdout"] == "output"
            assert result["stderr"] == ""
            assert result["success"] is True
            
            # Check that command execution was logged
            mock_shared_memory.add_memory.assert_called()
            log_entry = mock_shared_memory.add_memory.call_args[0][0]
            assert log_entry.type == "observation"
            assert "Executed command" in log_entry.content["observation"]
    
    @pytest.mark.asyncio
    async def test_execute_command_failure(self, base_agent):
        """Test failed command execution."""
        with patch('asyncio.create_subprocess_shell') as mock_subprocess:
            # Mock failed process
            mock_process = AsyncMock()
            mock_process.returncode = 1
            mock_process.communicate.return_value = (b"", b"error message")
            mock_subprocess.return_value = mock_process
            
            result = await base_agent.execute_command("false")
            
            assert result["command"] == "false"
            assert result["return_code"] == 1
            assert result["stdout"] == ""
            assert result["stderr"] == "error message"
            assert result["success"] is False
    
    @pytest.mark.asyncio
    async def test_execute_command_timeout(self, base_agent):
        """Test command execution timeout."""
        with patch('asyncio.create_subprocess_shell') as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.side_effect = TimeoutError()
            mock_subprocess.return_value = mock_process
            
            with pytest.raises(asyncio.TimeoutError):
                await base_agent.execute_command("sleep 10", timeout=0.1)
    
    @pytest.mark.asyncio
    async def test_execute_command_with_cwd(self, base_agent):
        """Test command execution with working directory."""
        with patch('asyncio.create_subprocess_shell') as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate.return_value = (b"output", b"")
            mock_subprocess.return_value = mock_process
            
            cwd = Path("/tmp")
            await base_agent.execute_command("pwd", cwd=cwd)
            
            mock_subprocess.assert_called_once_with(
                "pwd",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd
            )
    
    @pytest.mark.asyncio
    async def test_execute_command_exception(self, base_agent):
        """Test command execution with exception."""
        with patch('asyncio.create_subprocess_shell') as mock_subprocess:
            mock_subprocess.side_effect = OSError("Command not found")
            
            with pytest.raises(OSError):
                await base_agent.execute_command("nonexistent_command")


class TestBaseAgentConfiguration:
    """Test BaseAgent configuration methods."""
    
    def test_get_custom_instructions(self, base_agent):
        """Test getting custom instructions."""
        instructions = base_agent.get_custom_instructions()
        assert instructions == "Test instructions"
    
    def test_is_enabled(self, base_agent):
        """Test checking if agent is enabled."""
        assert base_agent.is_enabled() is True
    
    def test_is_disabled(self, mock_config, mock_shared_memory, mock_vector_db):
        """Test checking if agent is disabled."""
        agent_config = AgentConfig(enabled=False)
        mock_config.get_agent_config.return_value = agent_config
        
        agent = ConcreteBaseAgent(
            agent_id="disabled_agent",
            config=mock_config,
            shared_memory=mock_shared_memory,
            vector_db=mock_vector_db
        )
        
        assert agent.is_enabled() is False


class TestBaseAgentErrorHandling:
    """Test BaseAgent error handling."""
    
    def test_log_error_with_exception(self, base_agent, mock_shared_memory):
        """Test logging errors with exception details."""
        error_msg = "Test error"
        state = {"test": "state"}
        exception = ValueError("Test exception")
        
        base_agent._log_error(error_msg, state, exception)
        
        mock_shared_memory.add_memory.assert_called()
        error_entry = mock_shared_memory.add_memory.call_args[0][0]
        assert error_entry.type == "error"
        assert error_entry.content["error"] == error_msg
        assert error_entry.content["exception"] == "Test exception"
        assert "error" in error_entry.tags
        assert "agent_failure" in error_entry.tags
        assert "test_agent" in error_entry.tags
    
    def test_log_error_without_exception(self, base_agent, mock_shared_memory):
        """Test logging errors without exception details."""
        error_msg = "Test error"
        state = {"test": "state"}
        
        base_agent._log_error(error_msg, state)
        
        mock_shared_memory.add_memory.assert_called()
        error_entry = mock_shared_memory.add_memory.call_args[0][0]
        assert error_entry.type == "error"
        assert error_entry.content["error"] == error_msg
        assert error_entry.content["exception"] is None
    
    def test_log_error_with_pydantic_state(self, base_agent, mock_shared_memory):
        """Test logging errors with Pydantic model state."""
        error_msg = "Test error"
        
        # Create a mock state with model_dump method
        state = MagicMock()
        state.model_dump.return_value = {"test": "pydantic_state"}
        
        base_agent._log_error(error_msg, state)
        
        mock_shared_memory.add_memory.assert_called()
        error_entry = mock_shared_memory.add_memory.call_args[0][0]
        assert error_entry.content["state"] == {"test": "pydantic_state"}
        state.model_dump.assert_called_once()


class TestBaseAgentIntegration:
    """Integration tests for BaseAgent functionality."""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self, base_agent, mock_shared_memory):
        """Test a complete agent workflow."""
        # 1. Agent starts and logs observation
        obs_id = base_agent.log_observation("Starting work", {"task": "test"})
        assert obs_id == "test-memory-id"
        
        # 2. Agent makes a decision
        dec_id = base_agent.log_decision(
            "Will execute test task", 
            "Task is simple and can be completed quickly"
        )
        assert dec_id == "test-memory-id"
        
        # 3. Agent creates a subtask
        task_id = base_agent.create_task(
            "Subtask for testing", 
            "test_task", 
            "test_agent"
        )
        assert task_id == "test-task-id"
        
        # 4. Agent executes main task
        state = {"input": "test"}
        result = await base_agent.execute_with_retry(state)
        assert result["status"] == "completed"
        
        # 5. Agent updates task status
        success = base_agent.update_task_status(
            task_id, 
            "completed", 
            {"output": "success"}
        )
        assert success is True
        
        # 6. Agent logs final result
        result_id = base_agent.log_result(
            "Task completed successfully", 
            {"output": "success"}
        )
        assert result_id == "test-memory-id"
        
        # Verify all memory operations were called
        assert mock_shared_memory.add_memory.call_count >= 4  # obs, dec, task creation log, result
        assert mock_shared_memory.create_task.call_count == 1
        assert mock_shared_memory.update_task.call_count == 1
    
    def test_memory_entry_validation(self, base_agent):
        """Test that memory entries are properly validated."""
        # Test with valid data
        entry_id = base_agent.log_observation("Valid observation", {"key": "value"})
        assert entry_id == "test-memory-id"
        
        # Test with empty observation (should still work as content is not empty)
        entry_id = base_agent.log_observation("", {"key": "value"})
        assert entry_id == "test-memory-id"
    
    def test_task_creation_validation(self, base_agent):
        """Test that task creation properly validates inputs."""
        # Test with valid data
        task_id = base_agent.create_task("Valid task", "test_type")
        assert task_id == "test-task-id"
        
        # Test with minimal data
        task_id = base_agent.create_task("Minimal task", "minimal")
        assert task_id == "test-task-id"


class TestBaseAgentEnhancedStateManagement:
    """Test enhanced state management methods."""
    
    def test_get_agent_state(self, base_agent, mock_shared_memory):
        """Test getting agent state."""
        mock_state = AgentState(
            agent_id="test_agent",
            status="busy",
            current_task="12345678-1234-1234-1234-123456789012",
            last_heartbeat=datetime.now(UTC)
        )
        mock_shared_memory.get_agent_state.return_value = mock_state
        
        state = base_agent.get_agent_state()
        
        assert state == mock_state
        mock_shared_memory.get_agent_state.assert_called_once_with("test_agent")
    
    def test_get_all_agent_states(self, base_agent, mock_shared_memory):
        """Test getting all agent states."""
        mock_states = [
            AgentState(agent_id="agent1", status="idle"),
            AgentState(agent_id="agent2", status="busy")
        ]
        mock_shared_memory.get_all_agent_states.return_value = mock_states
        
        states = base_agent.get_all_agent_states()
        
        assert states == mock_states
        mock_shared_memory.get_all_agent_states.assert_called_once()
    
    def test_get_active_agents(self, base_agent, mock_shared_memory):
        """Test getting active agents."""
        mock_states = [
            AgentState(agent_id="agent1", status="idle"),
            AgentState(agent_id="agent2", status="busy"),
            AgentState(agent_id="agent3", status="error"),
            AgentState(agent_id="agent4", status="stopped")
        ]
        mock_shared_memory.get_all_agent_states.return_value = mock_states
        
        active_agents = base_agent.get_active_agents()
        
        assert len(active_agents) == 2
        assert active_agents[0].agent_id == "agent2"
        assert active_agents[1].agent_id == "agent3"
    
    def test_get_available_agents(self, base_agent, mock_shared_memory):
        """Test getting available agents."""
        mock_states = [
            AgentState(agent_id="agent1", status="idle"),
            AgentState(agent_id="agent2", status="busy"),
            AgentState(agent_id="agent3", status="idle")
        ]
        mock_shared_memory.get_all_agent_states.return_value = mock_states
        
        available_agents = base_agent.get_available_agents()
        
        assert len(available_agents) == 2
        assert available_agents[0].agent_id == "agent1"
        assert available_agents[1].agent_id == "agent3"
    
    def test_update_agent_metadata(self, base_agent, mock_shared_memory):
        """Test updating agent metadata."""
        mock_state = AgentState(
            agent_id="test_agent",
            status="busy",
            metadata={"old_key": "old_value"}
        )
        mock_shared_memory.get_agent_state.return_value = mock_state
        
        metadata = {"new_key": "new_value", "another_key": "another_value"}
        base_agent.update_agent_metadata(metadata)
        
        mock_shared_memory.update_agent_state.assert_called()
        updated_state = mock_shared_memory.update_agent_state.call_args[0][0]
        assert updated_state.metadata["old_key"] == "old_value"
        assert updated_state.metadata["new_key"] == "new_value"
        assert updated_state.metadata["another_key"] == "another_value"
    
    def test_get_agent_capabilities(self, base_agent, mock_shared_memory):
        """Test getting agent capabilities."""
        mock_state = AgentState(
            agent_id="test_agent",
            status="busy",
            current_task="12345678-1234-1234-1234-123456789012",
            last_heartbeat=datetime.now(UTC),
            metadata={"custom": "metadata"}
        )
        mock_shared_memory.get_agent_state.return_value = mock_state
        
        capabilities = base_agent.get_agent_capabilities()
        
        assert capabilities["agent_id"] == "test_agent"
        assert capabilities["enabled"] is True
        assert capabilities["priority"] == 5
        assert capabilities["current_status"] == "busy"
        assert capabilities["current_task"] == "12345678-1234-1234-1234-123456789012"
        assert capabilities["metadata"] == {"custom": "metadata"}
    
    def test_check_agent_health_healthy(self, base_agent):
        """Test checking agent health when healthy."""
        base_agent.agent_config.heartbeat_interval = 10.0
        
        health = base_agent.check_agent_health()
        
        assert health["agent_id"] == "test_agent"
        assert health["status"] == "healthy"
        assert health["issues"] == []
    
    def test_check_agent_health_unhealthy(self, base_agent, mock_shared_memory):
        """Test checking agent health when unhealthy."""
        old_time = datetime.now(UTC) - timedelta(minutes=5)
        mock_state = AgentState(
            agent_id="test_agent",
            status="idle",
            last_heartbeat=old_time
        )
        mock_shared_memory.get_agent_state.return_value = mock_state
        base_agent.agent_config.heartbeat_interval = 1.0
        
        health = base_agent.check_agent_health()
        
        assert health["status"] == "unhealthy"
        assert len(health["issues"]) == 1
        assert "Heartbeat too old" in health["issues"][0]
    
    def test_check_agent_health_error_state(self, base_agent, mock_shared_memory):
        """Test checking agent health when in error state."""
        mock_state = AgentState(
            agent_id="test_agent",
            status="error",
            last_heartbeat=datetime.now(UTC)
        )
        mock_shared_memory.get_agent_state.return_value = mock_state
        
        health = base_agent.check_agent_health()
        
        assert health["status"] == "error"
        assert len(health["issues"]) == 1
        assert "Agent is in error state" in health["issues"][0]


class TestBaseAgentEnhancedTaskManagement:
    """Test enhanced task management methods."""
    
    def test_get_all_tasks(self, base_agent, mock_shared_memory):
        """Test getting all tasks for an agent."""
        mock_tasks = [
            TaskStatus(agent_id="test_agent", status="pending", description="Task 1"),
            TaskStatus(agent_id="test_agent", status="completed", description="Task 2")
        ]
        mock_shared_memory.get_tasks.return_value = mock_tasks
        
        tasks = base_agent.get_all_tasks()
        
        assert tasks == mock_tasks
        mock_shared_memory.get_tasks.assert_called_once_with(
            agent_id="test_agent", status=None, limit=50
        )
    
    def test_get_all_tasks_filtered(self, base_agent, mock_shared_memory):
        """Test getting tasks filtered by status."""
        mock_tasks = [
            TaskStatus(agent_id="test_agent", status="completed", description="Task 1")
        ]
        mock_shared_memory.get_tasks.return_value = mock_tasks
        
        tasks = base_agent.get_all_tasks(status="completed")
        
        assert tasks == mock_tasks
        mock_shared_memory.get_tasks.assert_called_once_with(
            agent_id="test_agent", status="completed", limit=50
        )
    
    def test_get_task_by_id(self, base_agent, mock_shared_memory):
        """Test getting a specific task by ID."""
        mock_task = TaskStatus(
            agent_id="test_agent",
            status="pending",
            description="Test task"
        )
        mock_shared_memory.get_task.return_value = mock_task
        
        task = base_agent.get_task_by_id("task123")
        
        assert task == mock_task
        mock_shared_memory.get_task.assert_called_once_with("task123")
    
    def test_assign_task_to_self_success(self, base_agent, mock_shared_memory):
        """Test successfully assigning a task to self."""
        mock_task = TaskStatus(
            agent_id="other_agent",
            status="pending",
            description="Test task"
        )
        mock_shared_memory.get_task.return_value = mock_task

        success = base_agent.assign_task_to_self("12345678-1234-1234-1234-123456789012")
        
        assert success is True
        mock_shared_memory.update_task.assert_called_once()
        call_args = mock_shared_memory.update_task.call_args
        assert call_args[0][0] == "12345678-1234-1234-1234-123456789012"
        assert call_args[1]["status"] == "running"
        assert call_args[1]["metadata"]["assigned_to"] == "test_agent"
    
    def test_assign_task_to_self_not_found(self, base_agent, mock_shared_memory):
        """Test assigning a task that doesn't exist."""
        mock_shared_memory.get_task.return_value = None
        
        success = base_agent.assign_task_to_self("nonexistent_task")
        
        assert success is False
        mock_shared_memory.update_task.assert_not_called()
    
    def test_complete_current_task_success(self, base_agent, mock_shared_memory):
        """Test successfully completing the current task."""
        mock_state = AgentState(
            agent_id="test_agent",
            status="busy",
            current_task="12345678-1234-1234-1234-123456789012"
        )
        mock_shared_memory.get_agent_state.return_value = mock_state
        
        success = base_agent.complete_current_task({"result": "success"})
        
        assert success is True
        mock_shared_memory.update_task.assert_called_once_with(
            "12345678-1234-1234-1234-123456789012", status="completed", result={"result": "success"}
        )
    
    def test_complete_current_task_no_current_task(self, base_agent, mock_shared_memory):
        """Test completing when no current task."""
        mock_state = AgentState(agent_id="test_agent", status="idle", current_task=None)
        mock_shared_memory.get_agent_state.return_value = mock_state
        
        success = base_agent.complete_current_task()
        
        assert success is False
        mock_shared_memory.update_task.assert_not_called()
    
    def test_fail_current_task_success(self, base_agent, mock_shared_memory):
        """Test successfully failing the current task."""
        mock_state = AgentState(
            agent_id="test_agent",
            status="busy",
            current_task="12345678-1234-1234-1234-123456789012"
        )
        mock_shared_memory.get_agent_state.return_value = mock_state
        
        success = base_agent.fail_current_task("Test error", {"partial": "result"})
        
        assert success is True
        mock_shared_memory.update_task.assert_called_once_with(
            "12345678-1234-1234-1234-123456789012", status="failed", result={"partial": "result"}, error="Test error"
        )
    
    def test_cancel_task_success(self, base_agent, mock_shared_memory):
        """Test successfully cancelling a task."""
        success = base_agent.cancel_task("12345678-1234-1234-1234-123456789012", "User requested cancellation")

        assert success is True
        call_args = mock_shared_memory.update_task.call_args
        assert call_args[0][0] == "12345678-1234-1234-1234-123456789012"
        assert call_args[1]["status"] == "cancelled"
        assert call_args[1]["metadata"]["cancelled_by"] == "test_agent"
        assert call_args[1]["metadata"]["cancelled_reason"] == "User requested cancellation"
        assert "cancelled_at" in call_args[1]["metadata"]
    
    def test_cancel_task_without_reason(self, base_agent, mock_shared_memory):
        """Test cancelling a task without reason."""
        success = base_agent.cancel_task("12345678-1234-1234-1234-123456789012")

        assert success is True
        call_args = mock_shared_memory.update_task.call_args
        assert call_args[0][0] == "12345678-1234-1234-1234-123456789012"
        assert call_args[1]["status"] == "cancelled"
        assert "metadata" not in call_args[1]
    
    def test_get_task_dependencies(self, base_agent, mock_shared_memory):
        """Test getting task dependencies."""
        mock_task = TaskStatus(
            agent_id="test_agent",
            status="pending",
            description="Test task",
            dependencies=["12345678-1234-1234-1234-123456789012", "87654321-4321-4321-4321-210987654321"]
        )
        mock_shared_memory.get_task.return_value = mock_task

        mock_dep1 = TaskStatus(agent_id="test_agent", status="completed", description="Dep 1")
        mock_dep2 = TaskStatus(agent_id="test_agent", status="pending", description="Dep 2")
        mock_shared_memory.get_task.side_effect = [mock_task, mock_dep1, mock_dep2]

        dependencies = base_agent.get_task_dependencies("12345678-1234-1234-1234-123456789012")

        assert len(dependencies) == 2
        assert dependencies[0].description == "Dep 1"
        assert dependencies[1].description == "Dep 2"
    
    def test_get_task_dependencies_none(self, base_agent, mock_shared_memory):
        """Test getting task dependencies when none exist."""
        mock_task = TaskStatus(
            agent_id="test_agent",
            status="pending",
            description="Test task"
        )
        mock_shared_memory.get_task.return_value = mock_task
        
        dependencies = base_agent.get_task_dependencies("task123")
        
        assert dependencies == []
    
    def test_check_task_dependencies_completed(self, base_agent, mock_shared_memory):
        """Test checking if task dependencies are completed."""
        mock_task = TaskStatus(
            agent_id="test_agent",
            status="pending",
            description="Test task",
            dependencies=["dep1", "dep2"]
        )
        mock_shared_memory.get_task.return_value = mock_task

        mock_dep1 = TaskStatus(agent_id="test_agent", status="completed", description="Dep 1")
        mock_dep2 = TaskStatus(agent_id="test_agent", status="cancelled", description="Dep 2")
        mock_shared_memory.get_task.side_effect = [mock_dep1, mock_dep2]
        
        result = base_agent.check_task_dependencies_completed("task123")
        
        assert result is True
    
    def test_check_task_dependencies_not_completed(self, base_agent, mock_shared_memory):
        """Test checking when task dependencies are not completed."""
        mock_task = TaskStatus(
            agent_id="test_agent",
            status="pending",
            description="Test task",
            dependencies=["12345678-1234-1234-1234-123456789012", "87654321-4321-4321-4321-210987654321"]
        )
        mock_shared_memory.get_task.return_value = mock_task

        mock_dep1 = TaskStatus(agent_id="test_agent", status="completed", description="Dep 1")
        mock_dep2 = TaskStatus(agent_id="test_agent", status="pending", description="Dep 2")
        mock_shared_memory.get_task.side_effect = [mock_task, mock_dep1, mock_dep2]

        result = base_agent.check_task_dependencies_completed("12345678-1234-1234-1234-123456789012")

        assert result is False
    
    def test_create_dependent_task_success(self, base_agent, mock_shared_memory):
        """Test creating a dependent task successfully."""
        mock_dep1 = TaskStatus(agent_id="test_agent", status="completed", description="Dep 1")
        mock_dep2 = TaskStatus(agent_id="test_agent", status="completed", description="Dep 2")
        mock_shared_memory.get_task.side_effect = [mock_dep1, mock_dep2]
        
        task_id = base_agent.create_dependent_task(
            "Dependent task",
            "test_type",
            ["dep1", "dep2"],
            "target_agent",
            {"priority": "high"}
        )
        
        assert task_id == "test-task-id"
        mock_shared_memory.create_task.assert_called_once()
        
        # Check the created task
        created_task = mock_shared_memory.create_task.call_args[0][0]
        assert created_task.description == "Dependent task"
        assert created_task.metadata["type"] == "test_type"
        assert created_task.metadata["depends_on"] == ["dep1", "dep2"]
        assert created_task.dependencies == ["dep1", "dep2"]
    
    def test_create_dependent_task_invalid_dependency(self, base_agent, mock_shared_memory):
        """Test creating a dependent task with invalid dependency."""
        mock_dep1 = TaskStatus(agent_id="test_agent", status="completed", description="Dep 1")
        mock_shared_memory.get_task.side_effect = [mock_dep1, None]  # Second dependency doesn't exist
        
        with pytest.raises(ValueError, match="Dependency task nonexistent does not exist"):
            base_agent.create_dependent_task(
                "Dependent task",
                "test_type",
                ["dep1", "nonexistent"]
            )
    
    def test_get_tasks_by_status(self, base_agent, mock_shared_memory):
        """Test getting tasks by status."""
        mock_tasks = [
            TaskStatus(agent_id="test_agent", status="completed", description="Task 1"),
            TaskStatus(agent_id="test_agent", status="completed", description="Task 2")
        ]
        mock_shared_memory.get_tasks.return_value = mock_tasks
        
        tasks = base_agent.get_tasks_by_status("completed")
        
        assert tasks == mock_tasks
        mock_shared_memory.get_tasks.assert_called_once_with(
            agent_id="test_agent", status="completed", limit=50
        )
    
    def test_get_task_history(self, base_agent, mock_shared_memory):
        """Test getting task history."""
        now = datetime.now(UTC)
        mock_tasks = [
            TaskStatus(
                agent_id="test_agent",
                status="completed",
                description="Task 1",
                created_at=now - timedelta(hours=2)
            ),
            TaskStatus(
                agent_id="test_agent",
                status="failed",
                description="Task 2",
                created_at=now - timedelta(hours=1)
            ),
            TaskStatus(
                agent_id="test_agent",
                status="pending",
                description="Task 3",
                created_at=now - timedelta(minutes=30)
            )
        ]
        mock_shared_memory.get_tasks.return_value = mock_tasks
        
        history = base_agent.get_task_history()
        
        assert len(history) == 2  # Only completed and failed tasks
        assert history[0].description == "Task 2"  # Most recent first
        assert history[1].description == "Task 1"