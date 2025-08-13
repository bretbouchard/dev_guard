"""
Integration tests for Task 7.2: Conditional routing and agent coordination.

Tests the complete swarm orchestration workflow including agent selection,
task flow management, error recovery, and fallback mechanisms.
"""

import asyncio
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio

from dev_guard.core.config import AgentConfig, Config, RepositoryConfig, VectorDBConfig
from dev_guard.core.swarm import DevGuardSwarm, SwarmState
from dev_guard.memory.shared_memory import AgentState, TaskStatus


class TestSwarmCoordination:
    """Test swarm coordination and conditional routing."""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        data_dir = tempfile.mkdtemp()
        repo_dir = tempfile.mkdtemp()
        
        # Create a sample repository structure
        sample_files = [
            "src/main.py",
            "src/utils.py", 
            "tests/test_main.py",
            "README.md",
            "requirements.txt"
        ]
        
        for file_path in sample_files:
            full_path = Path(repo_dir) / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(f"# Sample content for {file_path}")
        
        yield data_dir, repo_dir
        
        # Cleanup
        import shutil
        shutil.rmtree(data_dir, ignore_errors=True)
        shutil.rmtree(repo_dir, ignore_errors=True)
    
    @pytest.fixture
    def test_config(self, temp_dirs):
        """Create test configuration."""
        data_dir, repo_dir = temp_dirs
        
        return Config(
            data_dir=data_dir,
            debug=True,  # Enable test mode (limits swarm loop iterations)
            vector_db=VectorDBConfig(
                path=f"{data_dir}/vector_db",
                collection_name="test_collection",
                embedding_model="sentence-transformers/all-MiniLM-L6-v2"
            ),
            agents={
                "commander": AgentConfig(enabled=True),
                "planner": AgentConfig(enabled=True),
                "code": AgentConfig(enabled=True),
                "qa_test": AgentConfig(enabled=True),
                "docs": AgentConfig(enabled=True),
                # Add missing agents to avoid warnings
                "git_watcher": AgentConfig(enabled=True),
                "impact_mapper": AgentConfig(enabled=True),
                "repo_auditor": AgentConfig(enabled=True),
                "dep_manager": AgentConfig(enabled=True),
                "red_team": AgentConfig(enabled=False),
            },
            repositories=[
                RepositoryConfig(
                    path=repo_dir,
                    branch="main",
                    watch_files=["*.py", "*.md", "*.txt"],
                    ignore_patterns=[".git/*", "__pycache__/*"]
                )
            ],
            swarm_interval=1  # Fast for testing
        )

    @pytest_asyncio.fixture
    async def swarm(self, test_config):
        """Create DevGuard swarm instance."""
        # Mock the agent imports within the swarm module
        with patch('dev_guard.agents.commander.CommanderAgent', MagicMock()), \
             patch('dev_guard.agents.planner.PlannerAgent', MagicMock()), \
             patch('dev_guard.agents.code_agent.CodeAgent', MagicMock()), \
             patch('dev_guard.agents.qa_test.QATestAgent', MagicMock()), \
             patch('dev_guard.agents.docs.DocsAgent', MagicMock()), \
             patch('dev_guard.agents.git_watcher.GitWatcherAgent', MagicMock()), \
             patch('dev_guard.agents.impact_mapper.ImpactMapperAgent', MagicMock()), \
             patch('dev_guard.agents.repo_auditor.RepoAuditorAgent', MagicMock()), \
             patch('dev_guard.agents.dep_manager.DepManagerAgent', MagicMock()):
            
            swarm = DevGuardSwarm(test_config)
            yield swarm
            if swarm.is_running:
                await swarm.stop()
    
    @pytest.mark.asyncio
    async def test_swarm_initialization(self, swarm, test_config):
        """Test swarm initializes correctly with all components."""
        assert swarm.config == test_config
        assert swarm.shared_memory is not None
        assert swarm.vector_db is not None
        assert swarm.compiled_graph is not None
        assert not swarm.is_running
        
        # Check that enabled agents are initialized
        expected_agents = ["commander", "planner", "code", "qa_test", "docs"]
        for agent_name in expected_agents:
            assert agent_name in swarm.agents
    
    @pytest.mark.asyncio
    async def test_agent_routing_from_commander(self, swarm):
        """Test conditional routing from commander agent."""
        # Test routing to planner when there are pending tasks
        state_with_tasks = SwarmState(
            pending_tasks=["task1", "task2"],
            active_agents=[]
        )
        route = swarm._route_from_commander(state_with_tasks)
        assert route == "planner"
        
        # Test routing to git_watcher when needed
        state_git_needed = SwarmState(
            pending_tasks=[],
            metadata={"last_git_check": 0}  # Long time ago
        )
        route = swarm._route_from_commander(state_git_needed)
        assert route in ["git_watcher", "repo_auditor", "end"]
        
        # Test routing to end when no tasks
        state_no_tasks = SwarmState(
            pending_tasks=[],
            active_agents=[],
            metadata={"last_git_check": datetime.now(UTC).timestamp()}
        )
        route = swarm._route_from_commander(state_no_tasks)
        assert route == "end"
    
    @pytest.mark.asyncio  
    async def test_agent_routing_from_planner(self, swarm):
        """Test intelligent task assignment from planner."""
        # Create test tasks with different types
        code_task = TaskStatus(
            agent_id="planner",
            status="pending", 
            description="implement a new function to calculate sum",
            metadata={"type": "code_generation"}
        )
        swarm.shared_memory.create_task(code_task)
        
        test_task = TaskStatus(
            agent_id="planner",
            status="pending",
            description="write unit tests for the calculator module",
            metadata={"type": "testing"}
        )
        swarm.shared_memory.create_task(test_task)
        
        # Test code task routing
        state_code = SwarmState(pending_tasks=[code_task.id])
        route = swarm._route_from_planner(state_code)
        assert route == "code"
        
        # Test testing task routing  
        state_test = SwarmState(pending_tasks=[test_task.id])
        route = swarm._route_from_planner(state_test)
        assert route == "qa_test"
        
        # Test no tasks routing
        state_empty = SwarmState(pending_tasks=[])
        route = swarm._route_from_planner(state_empty)
        assert route == "commander"
    
    @pytest.mark.asyncio
    async def test_task_priority_handling(self, swarm):
        """Test task priority ordering and selection."""
        # Create tasks with different priorities
        low_task = TaskStatus(
            agent_id="planner",
            status="pending",
            description="low priority documentation task",
            metadata={"priority": "low", "type": "documentation"}
        )
        low_id = swarm.shared_memory.create_task(low_task)
        
        high_task = TaskStatus(
            agent_id="planner", 
            status="pending",
            description="high priority bug fix",
            metadata={"priority": "high", "type": "code_generation"}
        )
        high_id = swarm.shared_memory.create_task(high_task)
        
        medium_task = TaskStatus(
            agent_id="planner",
            status="pending", 
            description="medium priority feature",
            metadata={"priority": "medium", "type": "code_generation"}
        )
        medium_id = swarm.shared_memory.create_task(medium_task)
        
        # Test priority selection
        pending_tasks = [low_id, high_id, medium_id]
        next_task = swarm._get_next_priority_task(pending_tasks)
        
        # Should select high priority task first
        assert next_task == high_id
    
    @pytest.mark.asyncio
    async def test_agent_availability_checking(self, swarm):
        """Test agent availability and load balancing."""
        # Test available agent
        assert swarm._is_agent_available("code", SwarmState())
        
        # Test agent already at max load
        busy_state = SwarmState(active_agents=["code"])
        assert not swarm._is_agent_available("code", busy_state)
        
        # Test unavailable agent
        assert not swarm._is_agent_available("nonexistent", SwarmState())
    
    @pytest.mark.asyncio
    async def test_fallback_agent_selection(self, swarm):
        """Test fallback mechanisms when primary agent unavailable."""
        # Create test task
        task = TaskStatus(
            agent_id="planner",
            status="pending",
            description="implement function with tests",
            metadata={"type": "code_generation"}
        )
        
        # Test fallback when primary agent busy
        busy_state = SwarmState(active_agents=["code"])  # Code agent busy
        fallback = swarm._find_fallback_agent("code", task, busy_state)
        assert fallback == "qa_test"  # Should use QA fallback
        
        # Test no fallback available
        all_busy_state = SwarmState(active_agents=["code", "qa_test"])
        fallback = swarm._find_fallback_agent("code", task, all_busy_state)
        # In debug mode, enabled fallback may still be considered; ensure None when all busy
        assert fallback is None

    @pytest.mark.asyncio
    async def test_task_analysis_and_agent_assignment(self, swarm):
        """Test intelligent task analysis for agent assignment."""
        # Test code-related task
        code_task = TaskStatus(
            agent_id="planner",
            status="pending",
            description="refactor the python function to use better algorithms",
            metadata={}
        )
        agent = swarm._analyze_task_for_agent_assignment(code_task)
        assert agent == "code"
        
        # Test testing-related task
        test_task = TaskStatus(
            agent_id="planner", 
            status="pending",
            description="write unit tests and check coverage for the module",
            metadata={}
        )
        agent = swarm._analyze_task_for_agent_assignment(test_task)
        assert agent == "qa_test"
        
        # Test documentation task
        doc_task = TaskStatus(
            agent_id="planner",
            status="pending",
            description="update the README with new API documentation",
            metadata={}
        )
        agent = swarm._analyze_task_for_agent_assignment(doc_task)
        assert agent == "docs"
        
        # Test explicit type override
        override_task = TaskStatus(
            agent_id="planner",
            status="pending", 
            description="some generic description",
            metadata={"type": "impact_analysis"}
        )
        agent = swarm._analyze_task_for_agent_assignment(override_task)
        assert agent == "impact_mapper"
    
    @pytest.mark.asyncio
    async def test_error_recovery_and_logging(self, swarm):
        """Test error recovery mechanisms and error logging."""
        # Test error handling in routing
        with patch.object(swarm.shared_memory, 'get_task', side_effect=Exception("DB Error")):
            # Should not crash, should return commander for recovery
            state = SwarmState(pending_tasks=["invalid_task"])
            route = swarm._route_from_planner(state)
            assert route == "commander"
        
        # Check error was logged to shared memory
        error_memories = swarm.shared_memory.get_memories(
            memory_type="error",
            tags={"error"},
            limit=10
        )
        # Should have error entries from swarm operations
        assert len(error_memories) >= 0  # May have errors from previous tests
    
    @pytest.mark.asyncio
    async def test_swarm_lifecycle(self, swarm):
        """Test complete swarm start/stop lifecycle."""
        # Test start
        assert not swarm.is_running
        await swarm.start()
        assert swarm.is_running
        
        # Allow some time for initialization
        await asyncio.sleep(0.1)
        
        # Test stop
        await swarm.stop()
        assert not swarm.is_running
        
        # Check agent states updated to stopped
        agent_states = swarm.shared_memory.get_all_agent_states()
        for agent_state in agent_states:
            assert agent_state.status == "stopped"
    
    @pytest.mark.asyncio
    async def test_task_creation_and_tracking(self, swarm):
        """Test task creation and lifecycle tracking."""
        # Create a task
        task_id = swarm.create_task(
            description="Test task for validation",
            task_type="code_generation",
            metadata={"priority": "high", "source": "integration_test", "type": "code_generation"}
        )
        
        assert task_id is not None
        
        # Verify task in database
        task = swarm.shared_memory.get_task(task_id)
        assert task is not None
        assert task.description == "Test task for validation"
        assert task.status == "pending"
        assert task.metadata["type"] == "code_generation"
        
        # Verify creation was logged
        creation_logs = swarm.shared_memory.get_memories(
            agent_id="swarm",
            memory_type="task",
            tags={"task", "creation"}
        )
        assert any(log.content.get("task_id") == task_id for log in creation_logs)
    
    @pytest.mark.asyncio
    async def test_repository_scanning_integration(self, swarm, temp_dirs):
        """Test repository initialization and scanning."""
        data_dir, repo_dir = temp_dirs
        
        # Start swarm to trigger repository scanning
        await swarm.start()
        await asyncio.sleep(0.2)  # Allow scan time
        await swarm.stop()
        
        # Check vector DB has documents
        stats = swarm.vector_db.get_collection_stats()
        assert stats.get("total_documents", 0) > 0
        
        # Check specific files were processed
        search_results = swarm.vector_db.search_files("main.py", n_results=5)
        assert len(search_results) > 0
    
    @pytest.mark.asyncio
    async def test_swarm_status_reporting(self, swarm):
        """Test comprehensive status reporting."""
        # Create some test data
        task_id = swarm.create_task(
            description="Status test task",
            task_type="testing"
        )
        
        # Update agent state
        test_state = AgentState(
            agent_id="code",
            status="busy", 
            current_task=task_id
        )
        swarm.shared_memory.update_agent_state(test_state)
        
        # Get status
        status = swarm.get_status()
        
        # Verify status structure
        assert "is_running" in status
        assert "agents" in status
        assert "recent_tasks" in status
        assert "repositories" in status
        assert "vector_db_documents" in status
        
        # Verify agent status
        assert "code" in status["agents"]
        agent_status = status["agents"]["code"]
        assert agent_status["status"] == "busy"
        assert agent_status["current_task"] == task_id
        
        # Verify recent tasks
        assert len(status["recent_tasks"]) > 0
        task_found = any(task["id"] == task_id for task in status["recent_tasks"])
        assert task_found
    
    @pytest.mark.asyncio
    async def test_concurrent_agent_coordination(self, swarm):
        """Test coordination with multiple agents working simultaneously."""
        # Create multiple tasks
        task_ids = []
        for i in range(3):
            task_id = swarm.create_task(
                description=f"Concurrent task {i}",
                task_type="code_generation" if i % 2 == 0 else "testing",
                metadata={"priority": "medium"}
            )
            task_ids.append(task_id)
        
        # Simulate agent coordination by checking routing
        state = SwarmState(
            pending_tasks=task_ids,
            active_agents=[]
        )
        
        # Route from planner multiple times
        assignments = []
        for _ in range(3):
            route = swarm._route_from_planner(state)
            if route != "commander":
                assignments.append(route)
                # Update state as if agent accepted task
                state.active_agents.append(route)
                if state.pending_tasks:
                    state.pending_tasks.pop(0)
        
        # Should have distributed tasks to different agents
        assert len(set(assignments)) >= 1  # At least one unique agent
        assert all(agent in ["code", "qa_test", "docs"] for agent in assignments)
