"""Unit tests for PlannerAgent implementation."""

import json
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from src.dev_guard.agents.planner import PlannerAgent
from src.dev_guard.memory.shared_memory import AgentState, TaskStatus


@pytest_asyncio.fixture
async def planner_agent(temp_dir, test_config):
    """Create a PlannerAgent instance for testing."""
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
            custom_instructions="Test planner instructions",
            priority=8,
            max_concurrent_tasks=3,
            heartbeat_interval=10.0
        )
        mock_config.get_agent_config.return_value = agent_config
        mock_config.repositories = []
        
        mock_memory_instance = MagicMock()
        mock_vector_instance = MagicMock()
        mock_memory.return_value = mock_memory_instance
        mock_vector.return_value = mock_vector_instance
        
        # Mock LLM provider
        mock_llm_provider = MagicMock()
        mock_llm_response = MagicMock()
        mock_llm_response.content = json.dumps({
            "approach": "systematic",
            "complexity": "high",
            "subtasks": ["Analyze requirements", "Design solution", "Implement"],
            "recommended_agents": ["code_agent"],
            "dependencies": []
        })
        mock_llm_provider.generate = AsyncMock(return_value=mock_llm_response)
        
        agent = PlannerAgent(
            agent_id="planner_test",
            config=mock_config,
            shared_memory=mock_memory_instance,
            vector_db=mock_vector_instance
        )
        
        # Set the LLM provider after initialization
        agent.llm_provider = mock_llm_provider
        
        # Set up common mock returns
        mock_vector_instance.search = AsyncMock(return_value=[
            {
                "content": "def example_function():",
                "metadata": {"file_path": "/test/example.py", "file_type": "python"}
            }
        ])
        
        return agent


class TestPlannerAgent:
    """Test suite for PlannerAgent functionality."""
    
    @pytest.mark.asyncio
    async def test_planner_initialization(self, planner_agent):
        """Test planner agent initializes correctly."""
        assert planner_agent.agent_id == "planner_test"
        assert planner_agent.shared_memory is not None
        assert planner_agent.vector_db is not None
        assert planner_agent.llm_provider is not None
        assert planner_agent.max_task_depth == 5
        assert planner_agent.parallel_task_limit == 3
    
    @pytest.mark.asyncio
    async def test_analyze_and_plan_with_llm(self, planner_agent):
        """Test task analysis and planning with LLM."""
        task = {
            "type": "analyze_and_plan",
            "description": "Implement user authentication system",
            "context": {"priority": "high", "deadline": "1 week"},
            "task_id": str(uuid.uuid4())
        }
        
        result = await planner_agent._analyze_and_plan(task)
        
        assert result["success"] is True
        assert "plan" in result
        assert "subtasks" in result
        assert len(result["subtasks"]) == 3
        assert result["next_action"] == "distribute_tasks"
        
        # Verify LLM was called
        planner_agent.llm_provider.generate.assert_called_once()
        
        # Verify vector search was performed
        planner_agent.vector_db.search.assert_called_once_with(
            query="Implement user authentication system",
            limit=5,
            filter_metadata={"type": "code"}
        )
    
    @pytest.mark.asyncio
    async def test_analyze_and_plan_fallback_to_heuristic(self, planner_agent):
        """Test fallback to heuristic planning when LLM fails."""
        task = {
            "type": "analyze_and_plan",
            "description": "Fix authentication bug",
            "context": {},
            "task_id": str(uuid.uuid4())
        }
        
        # Make LLM return invalid JSON
        mock_response = MagicMock()
        mock_response.content = "Invalid JSON response"
        planner_agent.llm_provider.generate = AsyncMock(return_value=mock_response)
        
        result = await planner_agent._analyze_and_plan(task)
        
        assert result["success"] is True
        assert "plan" in result
        assert result["plan"]["approach"] == "heuristic"
        assert "subtasks" in result
    
    @pytest.mark.asyncio
    async def test_heuristic_planning_code_task(self, planner_agent):
        """Test heuristic planning for code-related tasks."""
        result = await planner_agent._heuristic_planning(
            "implement a new function for user login",
            {"priority": "medium"}
        )
        
        assert result["approach"] == "heuristic"
        assert "code_agent" in result["recommended_agents"]
        assert len(result["subtasks"]) > 0
        assert any("code" in subtask.lower() for subtask in result["subtasks"])
    
    @pytest.mark.asyncio
    async def test_heuristic_planning_documentation_task(self, planner_agent):
        """Test heuristic planning for documentation tasks."""
        result = await planner_agent._heuristic_planning(
            "update the README documentation",
            {"priority": "low"}
        )
        
        assert result["approach"] == "heuristic"
        assert "code_agent" in result["recommended_agents"]  # Will expand when docs agent exists
        assert len(result["subtasks"]) > 0
        assert any("doc" in subtask.lower() for subtask in result["subtasks"])
    
    @pytest.mark.asyncio
    async def test_route_task_basic(self, planner_agent):
        """Test basic task routing functionality."""
        task = {
            "description": "implement authentication middleware",
            "task_type": "code_generation",
            "priority": "high"
        }
        
        # Mock agent states
        planner_agent.shared_memory.get_agent_states.return_value = {
            "code_agent": AgentState(
                agent_id="code_agent",
                status="idle",
                current_task=None,
                last_heartbeat=datetime.now(UTC),
                metadata={}
            )
        }
        
        result = await planner_agent._route_task(task)
        
        assert result["success"] is True
        assert result["target_agent"] == "code_agent"
        assert "routing_reason" in result
        assert result["priority"] == "high"
    
    @pytest.mark.asyncio
    async def test_route_task_with_busy_agent(self, planner_agent):
        """Test task routing when primary agent is busy."""
        task = {
            "description": "fix bug in user module",
            "task_type": "debug",
        }
        
        # Mock busy agent state
        planner_agent.shared_memory.get_agent_states.return_value = {
            "code_agent": AgentState(
                agent_id="code_agent",
                status="busy",
                current_task=str(uuid.uuid4()),
                last_heartbeat=datetime.now(UTC),
                metadata={}
            )
        }
        
        result = await planner_agent._route_task(task)
        
        assert result["success"] is True
        assert result["target_agent"] == "code_agent"  # Currently falls back to same agent
    
    @pytest.mark.asyncio
    async def test_monitor_progress_with_active_tasks(self, planner_agent):
        """Test progress monitoring with active tasks."""
        current_time = datetime.now(UTC)
        mock_tasks = [
            TaskStatus(
                agent_id="code_agent",
                status="running",
                created_at=current_time - timedelta(minutes=2),
                description="Recent task"
            ),
            TaskStatus(
                agent_id="qa_agent",
                status="running",
                created_at=current_time - timedelta(minutes=10),  # Long running
                description="Long running task that might be stuck"
            )
        ]
        
        planner_agent.shared_memory.get_tasks.return_value = mock_tasks
        
        result = await planner_agent._monitor_progress({})
        
        assert result["success"] is True
        assert result["progress"]["active_tasks"] == 2
        assert len(result["progress"]["task_details"]) == 2
        assert len(result["progress"]["bottlenecks"]) == 1  # One long-running task
        assert len(result["progress"]["recommendations"]) > 0
    
    @pytest.mark.asyncio
    async def test_monitor_progress_no_active_tasks(self, planner_agent):
        """Test progress monitoring with no active tasks."""
        planner_agent.shared_memory.get_tasks.return_value = []
        
        result = await planner_agent._monitor_progress({})
        
        assert result["success"] is True
        assert result["progress"]["active_tasks"] == 0
        assert len(result["progress"]["task_details"]) == 0
        assert len(result["progress"]["bottlenecks"]) == 0
    
    @pytest.mark.asyncio
    async def test_coordinate_agents_sequential(self, planner_agent):
        """Test sequential agent coordination."""
        task = {
            "workflow_type": "sequential",
            "agents": ["code_agent", "qa_agent", "docs_agent"]
        }
        
        result = await planner_agent._coordinate_agents(task)
        
        assert result["success"] is True
        assert result["coordination_plan"]["workflow_type"] == "sequential"
        assert len(result["coordination_plan"]["agents"]) == 3
        assert len(result["coordination_plan"]["coordination_steps"]) == 3
    
    @pytest.mark.asyncio
    async def test_coordinate_agents_parallel(self, planner_agent):
        """Test parallel agent coordination."""
        task = {
            "workflow_type": "parallel",
            "agents": ["code_agent", "qa_agent"]
        }
        
        result = await planner_agent._coordinate_agents(task)
        
        assert result["success"] is True
        assert result["coordination_plan"]["workflow_type"] == "parallel"
        assert "Parallel execution" in result["coordination_plan"]["coordination_steps"][0]
    
    @pytest.mark.asyncio
    async def test_generic_planning(self, planner_agent):
        """Test generic planning for unknown task types."""
        task = {
            "description": "Some unknown task type",
            "context": {"priority": "low"}
        }
        
        result = await planner_agent._generic_planning(task)
        
        assert result["success"] is True
        assert result["plan"]["approach"] == "analyze_then_execute"
        assert len(result["plan"]["steps"]) > 0
        assert result["next_action"] == "route_to_appropriate_agent"
    
    @pytest.mark.parametrize("description,expected_agent", [
        ("implement a new function", "code_agent"),
        ("fix the authentication bug", "code_agent"),
        ("coordinate the deployment process", "commander"),
        ("manage the testing workflow", "commander"),
        ("debug the payment system", "code_agent"),
    ])
    @pytest.mark.asyncio
    async def test_determine_target_agent(self, planner_agent, description, expected_agent):
        """Test agent determination based on task description."""
        result = planner_agent._determine_target_agent(description, "")
        assert result == expected_agent
    
    def test_create_subtasks(self, planner_agent):
        """Test subtask creation from plan."""
        plan = {
            "subtasks": [
                "Analyze requirements",
                "Design solution",
                "Implement code"
            ],
            "recommended_agents": ["code_agent"],
            "dependencies": []
        }
        
        parent_task_id = str(uuid.uuid4())
        subtasks = planner_agent._create_subtasks(plan, parent_task_id)
        
        assert len(subtasks) == 3
        for i, subtask in enumerate(subtasks):
            assert subtask["parent_task_id"] == parent_task_id
            assert f"subtask_{i}" in subtask["task_id"]
            assert subtask["recommended_agent"] == "code_agent"
            assert subtask["type"] == "execute"
    
    @pytest.mark.asyncio
    async def test_execute_task_analyze_and_plan(self, planner_agent):
        """Test execute method with analyze_and_plan task."""
        task = {
            "type": "analyze_and_plan",
            "description": "Build user management system",
            "task_id": str(uuid.uuid4())
        }
        
        result = await planner_agent.execute(task)
        
        assert result["success"] is True
        assert "plan" in result
        assert "subtasks" in result
    
    @pytest.mark.asyncio
    async def test_execute_task_route_task(self, planner_agent):
        """Test execute method with route_task."""
        task = {
            "type": "route_task",
            "description": "implement user login",
            "task_type": "code_generation"
        }
        
        planner_agent.shared_memory.get_agent_states.return_value = {}
        
        result = await planner_agent.execute(task)
        
        assert result["success"] is True
        assert result["target_agent"] == "code_agent"
    
    @pytest.mark.asyncio
    async def test_execute_task_error_handling(self, planner_agent):
        """Test execute method handles errors gracefully."""
        task = {
            "type": "analyze_and_plan",
            "description": "test task"
        }
        
        # Force an error by making vector_db.search fail
        planner_agent.vector_db.search.side_effect = Exception("Database error")
        
        result = await planner_agent.execute(task)
        
        assert result["success"] is False
        assert "error" in result
        assert result["subtasks"] == []
    
    def test_get_capabilities(self, planner_agent):
        """Test capabilities reporting."""
        capabilities = planner_agent.get_capabilities()
        
        expected_capabilities = [
            "task_analysis",
            "task_breakdown",
            "agent_routing",
            "progress_monitoring",
            "workflow_coordination",
            "dependency_management"
        ]
        
        for capability in expected_capabilities:
            assert capability in capabilities
    
    def test_get_status(self, planner_agent):
        """Test status reporting."""
        status = planner_agent.get_status()
        
        assert status["agent_id"] == "planner_test"
        assert status["type"] == "planner"
        assert "capabilities" in status
        assert "config" in status
        assert status["config"]["max_task_depth"] == 5
        assert status["config"]["parallel_task_limit"] == 3
    
    def test_format_relevant_docs(self, planner_agent):
        """Test document formatting for LLM context."""
        docs = [
            {
                "content": "def login_user(username, password):\n    # Implementation here\n    pass",
                "metadata": {"file_path": "/auth/login.py", "file_type": "python"}
            },
            {
                "content": "class UserManager:\n    def __init__(self):\n        pass",
                "metadata": {"file_path": "/models/user.py", "file_type": "python"}
            }
        ]
        
        formatted = planner_agent._format_relevant_docs(docs)
        
        assert "/auth/login.py" in formatted
        assert "/models/user.py" in formatted
        assert "def login_user" in formatted
        assert "class UserManager" in formatted
    
    def test_format_relevant_docs_empty(self, planner_agent):
        """Test document formatting with empty list."""
        formatted = planner_agent._format_relevant_docs([])
        assert "No relevant code context found" in formatted
    
    @pytest.mark.asyncio
    async def test_llm_analyze_task_json_parsing(self, planner_agent):
        """Test LLM task analysis with proper JSON response."""
        expected_plan = {
            "approach": "comprehensive",
            "complexity": "high",
            "subtasks": ["Phase 1", "Phase 2", "Phase 3"],
            "recommended_agents": ["code_agent", "qa_agent"],
            "dependencies": ["database_setup"]
        }
        
        mock_response = MagicMock()
        mock_response.content = json.dumps(expected_plan)
        planner_agent.llm_provider.generate = AsyncMock(return_value=mock_response)
        
        result = await planner_agent._llm_analyze_task(
            "Complex system implementation",
            {"priority": "high"},
            []
        )
        
        assert result == expected_plan
    
    @pytest.mark.asyncio
    async def test_llm_analyze_task_fallback_on_error(self, planner_agent):
        """Test LLM task analysis falls back to heuristic on error."""
        planner_agent.llm_provider.generate = AsyncMock(side_effect=Exception("LLM error"))
        
        result = await planner_agent._llm_analyze_task(
            "implement user authentication",
            {},
            []
        )
        
        # Should fallback to heuristic planning
        assert result["approach"] == "heuristic"
        assert "recommended_agents" in result
