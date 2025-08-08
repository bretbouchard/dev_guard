"""Unit tests for CommanderAgent implementation."""

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from dev_guard.agents.commander import CommanderAgent
from dev_guard.memory.shared_memory import AgentState, TaskStatus


@pytest_asyncio.fixture
async def commander_agent(temp_dir, test_config):
    """Create a CommanderAgent instance for testing."""
    with patch('dev_guard.memory.shared_memory.SharedMemory') as mock_memory, \
         patch('dev_guard.memory.vector_db.VectorDatabase') as mock_vector:
        
        # Create proper Config mock
        from dev_guard.core.config import AgentConfig, Config
        mock_config = MagicMock(spec=Config)
        agent_config = AgentConfig(
            enabled=True,
            max_retries=3,
            retry_delay=1.0,
            timeout=30.0,
            custom_instructions="Test commander instructions",
            priority=10,
            max_concurrent_tasks=1,
            heartbeat_interval=10.0
        )
        mock_config.get_agent_config.return_value = agent_config
        mock_config.repositories = []  # Empty repositories list for testing
        
        mock_memory_instance = MagicMock()
        mock_vector_instance = MagicMock()
        mock_memory.return_value = mock_memory_instance
        mock_vector.return_value = mock_vector_instance
        
        agent = CommanderAgent(
            agent_id="commander_test",
            config=mock_config,
            shared_memory=mock_memory_instance,
            vector_db=mock_vector_instance
        )
        
        # Set up common mock returns
        mock_vector_instance.count_documents.return_value = 150
        
        # Mock BaseAgent methods that are called
        agent.create_task = MagicMock(return_value="test_task_id")
        agent.set_status = MagicMock()
        agent.update_heartbeat = MagicMock()
        agent.log_observation = MagicMock()
        agent.log_decision = MagicMock()
        
        return agent


class TestCommanderAgent:
    """Test suite for CommanderAgent functionality."""
    
    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_commander_initialization(self, commander_agent):
        """Test commander agent initializes correctly."""
        assert commander_agent.agent_id == "commander_test"
        assert commander_agent.shared_memory is not None
        assert commander_agent.vector_db is not None
    
    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_agent_health_check_all_healthy(self, commander_agent):
        """Test health check when all agents are healthy."""
        # Mock healthy agent states
        current_time = datetime.now(UTC)
        test_task_id = str(uuid.uuid4())
        healthy_states = [
            AgentState(
                agent_id="planner",
                status="idle",
                last_heartbeat=current_time - timedelta(minutes=1),
                current_task=None,
                metadata={}
            ),
            AgentState(
                agent_id="code_agent",
                status="busy",
                last_heartbeat=current_time - timedelta(minutes=2),
                current_task=test_task_id,
                metadata={}
            ),
            AgentState(
                agent_id="commander_test",  # Self - should be ignored
                status="busy",
                last_heartbeat=current_time,
                current_task=None,
                metadata={}
            )
        ]
        
        commander_agent.shared_memory.get_all_agent_states.return_value = healthy_states
        
        health_report = await commander_agent._check_agent_health()
        
        assert "planner" in health_report["healthy"]
        assert "code_agent" in health_report["healthy"]
        assert "commander_test" not in health_report["healthy"]  # Self ignored
        assert len(health_report["unhealthy"]) == 0
        assert len(health_report["unresponsive"]) == 0
    
    @pytest.mark.asyncio
    async def test_agent_health_check_with_unhealthy_agents(self, commander_agent):
        """Test health check with unhealthy agents."""
        current_time = datetime.now(UTC)
        mixed_states = [
            AgentState(
                agent_id="planner",
                status="error",
                last_heartbeat=current_time - timedelta(minutes=1),
                current_task=None,
                metadata={}
            ),
            AgentState(
                agent_id="code_agent",
                status="idle",
                last_heartbeat=current_time - timedelta(minutes=10),  # Unresponsive
                current_task=None,
                metadata={}
            ),
            AgentState(
                agent_id="qa_agent",
                status="busy",
                last_heartbeat=current_time - timedelta(minutes=1),
                current_task=str(uuid.uuid4()),
                metadata={}
            )
        ]
        
        commander_agent.shared_memory.get_all_agent_states.return_value = mixed_states
        
        with patch.object(commander_agent, '_handle_unhealthy_agents', new_callable=AsyncMock) as mock_handle:
            health_report = await commander_agent._check_agent_health()
            
            assert "planner" in health_report["unhealthy"]
            assert "code_agent" in health_report["unresponsive"]
            assert "qa_agent" in health_report["healthy"]
            mock_handle.assert_called_once_with(health_report)
    
    @pytest.mark.asyncio
    async def test_handle_unhealthy_agents(self, commander_agent):
        """Test handling of unhealthy agents."""
        health_report = {
            "healthy": ["qa_agent"],
            "unhealthy": ["planner"],
            "unresponsive": ["code_agent"]
        }
        
        await commander_agent._handle_unhealthy_agents(health_report)
        
        # Verify logging calls were made (we can check the call count)
        assert commander_agent.log_decision.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_process_notifications_no_critical_errors(self, commander_agent):
        """Test notification processing with no critical errors."""
        with patch.object(commander_agent, 'get_recent_memories', return_value=[]) as mock_get_memories:
            await commander_agent._process_notifications()
            
            mock_get_memories.assert_called_once_with(
                memory_type="error",
                limit=10,
                include_other_agents=True
            )
    
    @pytest.mark.asyncio
    async def test_process_notifications_with_critical_errors(self, commander_agent):
        """Test notification processing with critical errors."""
        from dev_guard.memory.shared_memory import MemoryEntry
        
        critical_error = MemoryEntry(
            agent_id="planner",
            type="error",
            content={"message": "Critical system failure"},
            tags={"critical", "system_failure"},
            timestamp=datetime.now(UTC)
        )
        
        with patch.object(commander_agent, 'get_recent_memories', return_value=[critical_error]) as mock_get_memories:
            await commander_agent._process_notifications()
            
            # Should log observation about critical errors
            commander_agent.log_observation.assert_called_with(
                "Found 1 critical errors requiring attention",
                data={"error_count": 1},
                tags=["critical_alerts"]
            )
    
    @pytest.mark.asyncio
    async def test_evaluate_system_status(self, commander_agent):
        """Test system status evaluation."""
        # Mock task data
        current_time = datetime.now(UTC)
        mock_tasks = [
            TaskStatus(
                agent_id="planner", 
                status="completed", 
                description="Test task 1", 
                created_at=current_time
            ),
            TaskStatus(
                agent_id="code", 
                status="pending", 
                description="Test task 2", 
                created_at=current_time
            ),
            TaskStatus(
                agent_id="qa", 
                status="failed", 
                description="Test task 3", 
                created_at=current_time
            ),
            TaskStatus(
                agent_id="docs", 
                status="running", 
                description="Test task 4", 
                created_at=current_time
            ),
        ]
        
        commander_agent.shared_memory.get_tasks.return_value = mock_tasks
        commander_agent.vector_db.count_documents.return_value = 200
        
        system_status = await commander_agent._evaluate_system_status()
        
        assert system_status["task_stats"]["total"] == 4
        assert system_status["task_stats"]["completed"] == 1
        assert system_status["task_stats"]["pending"] == 1
        assert system_status["task_stats"]["failed"] == 1
        assert system_status["task_stats"]["running"] == 1
        assert system_status["success_rate"] == 0.5  # 1 completed out of 2 completed+failed
        assert system_status["vector_documents"] == 200
    
    @pytest.mark.asyncio
    async def test_decide_next_action_with_failed_tasks(self, commander_agent):
        """Test decision making when there are failed tasks."""
        system_status = {
            "task_stats": {"failed": 3, "pending": 2},
            "success_rate": 0.9,
            "vector_documents": 200
        }
        
        action = await commander_agent._decide_next_action(None, system_status)
        
        assert action == "investigate_failures"
    
    @pytest.mark.asyncio
    async def test_decide_next_action_with_many_pending_tasks(self, commander_agent):
        """Test decision making with many pending tasks."""
        system_status = {
            "task_stats": {"failed": 0, "pending": 8},
            "success_rate": 0.9,
            "vector_documents": 200
        }
        
        action = await commander_agent._decide_next_action(None, system_status)
        
        assert action == "delegate_to_planner"
    
    @pytest.mark.asyncio
    async def test_decide_next_action_with_low_vector_docs(self, commander_agent):
        """Test decision making with low vector document count."""
        system_status = {
            "task_stats": {"failed": 0, "pending": 2},
            "success_rate": 0.9,
            "vector_documents": 50
        }
        
        action = await commander_agent._decide_next_action(None, system_status)
        
        assert action == "audit_repositories"
    
    @pytest.mark.asyncio
    async def test_decide_next_action_with_low_success_rate(self, commander_agent):
        """Test decision making with low success rate."""
        system_status = {
            "task_stats": {"failed": 0, "pending": 2},
            "success_rate": 0.6,
            "vector_documents": 200
        }
        
        action = await commander_agent._decide_next_action(None, system_status)
        
        assert action == "analyze_performance"
    
    @pytest.mark.asyncio
    async def test_decide_next_action_healthy_system(self, commander_agent):
        """Test decision making when system is healthy."""
        system_status = {
            "task_stats": {"failed": 0, "pending": 2},
            "success_rate": 0.9,
            "vector_documents": 200
        }
        
        action = await commander_agent._decide_next_action(None, system_status)
        
        assert action == "monitor_repositories"
    
    @pytest.mark.parametrize("user_request,expected_type", [
        ("implement a new function", "code_generation"),
        ("write some code for auth", "code_generation"),
        ("create a class for user management", "code_generation"),
        ("update the README file", "documentation"),
        ("add docs for this module", "documentation"),
        ("analyze the impact of this change", "analysis"),
        ("check dependencies for this repo", "analysis"),
        ("run tests for the project", "testing"),
        ("perform quality checks", "testing"),
        ("help me with something", "general"),
    ])
    @pytest.mark.asyncio
    async def test_categorize_request(self, commander_agent, user_request, expected_type):
        """Test request categorization logic."""
        result = await commander_agent._categorize_request(user_request)
        assert result == expected_type
    
    @pytest.mark.asyncio
    async def test_handle_user_request_code_generation(self, commander_agent):
        """Test handling user request for code generation."""
        request = "implement a new authentication function"
        
        commander_agent.create_task.return_value = "task_12345"
        
        response = await commander_agent.handle_user_request(request)
        
        assert response["status"] == "accepted"
        assert response["task_id"] == "task_12345"
        assert response["request_type"] == "code_generation"
        assert "Task ID: task_12345" in response["message"]
        
        commander_agent.create_task.assert_called_once_with(
            description=f"User requested: {request}",
            task_type="code_generation",
            target_agent="code",
            metadata={"user_request": True, "context": None}
        )
    
    @pytest.mark.asyncio
    async def test_handle_user_request_with_context(self, commander_agent):
        """Test handling user request with additional context."""
        request = "update documentation"
        context = {"file_path": "/path/to/file.py", "priority": "high"}
        
        commander_agent.create_task.return_value = "task_67890"
        
        response = await commander_agent.handle_user_request(request, context)
        
        assert response["status"] == "accepted"
        assert response["request_type"] == "documentation"
        
        commander_agent.create_task.assert_called_once_with(
            description=f"User requested: {request}",
            task_type="documentation",
            target_agent="docs",
            metadata={"user_request": True, "context": context}
        )
    
    @pytest.mark.asyncio
    async def test_get_system_overview(self, commander_agent):
        """Test comprehensive system overview generation."""
        # Mock agent states
        current_time = datetime.now(UTC)
        test_task_uuid = str(uuid.uuid4())
        agent_states = [
            AgentState(
                agent_id="planner",
                status="busy",
                current_task=test_task_uuid,
                last_heartbeat=current_time,
                metadata={"queue_size": 5}
            ),
            AgentState(
                agent_id="code_agent",
                status="idle",
                current_task=None,
                last_heartbeat=current_time - timedelta(minutes=1),
                metadata={}
            )
        ]
        
        # Mock tasks
        mock_tasks = [
            TaskStatus(agent_id="planner", status="completed", description="Test task 1", created_at=current_time),
            TaskStatus(agent_id="code", status="pending", description="Test task 2", created_at=current_time),
            TaskStatus(agent_id="qa", status="running", description="Test task 3", created_at=current_time),
        ]
        
        commander_agent.shared_memory.get_all_agent_states.return_value = agent_states
        commander_agent.shared_memory.get_tasks.return_value = mock_tasks
        commander_agent.vector_db.count_documents.return_value = 250
        
        overview = commander_agent.get_system_overview()
        
        assert "timestamp" in overview
        assert "agents" in overview
        assert "task_summary" in overview
        assert "repositories" in overview
        assert "vector_documents" in overview
        
        assert "planner" in overview["agents"]
        assert "code_agent" in overview["agents"]
        assert overview["agents"]["planner"]["status"] == "busy"
        assert overview["agents"]["planner"]["current_task"] == test_task_uuid
        
        assert overview["task_summary"]["total_recent"] == 3
        assert overview["task_summary"]["by_status"]["completed"] == 1
        assert overview["task_summary"]["by_status"]["pending"] == 1
        assert overview["task_summary"]["by_status"]["running"] == 1
        
        assert overview["vector_documents"] == 250
    
    @pytest.mark.asyncio
    async def test_execute_full_cycle(self, commander_agent):
        """Test full execute cycle."""
        # Mock dependencies
        commander_agent.shared_memory.get_all_agent_states.return_value = []
        with patch.object(commander_agent, 'get_recent_memories', 
                         return_value=[]):
            commander_agent.shared_memory.get_tasks.return_value = []
            commander_agent.vector_db.count_documents.return_value = 150
            
            # Create a mock state
            mock_state = MagicMock()
            mock_state.model_dump.return_value = {"test": "data"}
            
            result = await commander_agent.execute(mock_state)
            
            # Should return the same state
            assert result == mock_state
            
            # Should have logged observations
            assert commander_agent.log_observation.call_count > 0
            
            # Should have set status to idle at the end
            commander_agent.set_status.assert_called_with("idle")
    
    @pytest.mark.asyncio
    async def test_execute_with_error(self, commander_agent):
        """Test execute cycle handles errors gracefully."""
        # Mock an error in health check
        commander_agent.shared_memory.get_all_agent_states.side_effect = Exception("Database error")
        
        mock_state = MagicMock()
        
        with pytest.raises(Exception, match="Database error"):
            await commander_agent.execute(mock_state)
        
        # Should set status to error
        commander_agent.set_status.assert_called_with("error")
