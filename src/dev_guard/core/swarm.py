"""Core LangGraph swarm orchestration for DevGuard."""

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

from pydantic import BaseModel, Field

from ..agents.base_agent import BaseAgent
from ..core.config import Config
from ..memory.shared_memory import AgentState, MemoryEntry, SharedMemory, TaskStatus
from ..memory.vector_db import VectorDatabase

logger = logging.getLogger(__name__)


class SwarmState(BaseModel):
    """State shared across all agents in the swarm."""
    current_task: str | None = None
    active_agents: list[str] = Field(default_factory=list)
    pending_tasks: list[str] = Field(default_factory=list)
    completed_tasks: list[str] = Field(default_factory=list)
    failed_tasks: list[str] = Field(default_factory=list)
    repositories: dict[str, dict[str, Any]] = Field(default_factory=dict)
    last_update: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class DevGuardSwarm:
    """Main swarm orchestrator using LangGraph for agent coordination."""
    
    def __init__(self, config: Config):
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "LangGraph is required but not installed. "
                "Install with: pip install langgraph"
            )
        
        self.config = config
        self.shared_memory = SharedMemory(
            db_path=Path(config.data_dir) / "shared_memory.db"
        )
        self.vector_db = VectorDatabase(config.vector_db)
        
        # Initialize agents
        self.agents: dict[str, BaseAgent] = {}
        self._initialize_agents()
        
        # Initialize LangGraph
        self.graph = None
        self.checkpointer = None
        self._build_graph()
        
        # Swarm state
        self.is_running = False
        self._shutdown_event = asyncio.Event()
        
        logger.info("DevGuard swarm initialized")
    
    def _initialize_agents(self) -> None:
        """Initialize all agents based on configuration."""
        from ..agents.code_agent import CodeAgent
        from ..agents.commander import CommanderAgent
        from ..agents.dep_manager import DepManagerAgent
        from ..agents.docs import DocsAgent
        from ..agents.git_watcher import GitWatcherAgent
        from ..agents.impact_mapper import ImpactMapperAgent
        from ..agents.planner import PlannerAgent
        from ..agents.qa_test import QATestAgent
        from ..agents.repo_auditor import RepoAuditorAgent
        
        agent_classes = {
            "commander": CommanderAgent,
            "planner": PlannerAgent,
            "code": CodeAgent,
            "qa_test": QATestAgent,
            "docs": DocsAgent,
            "git_watcher": GitWatcherAgent,
            "impact_mapper": ImpactMapperAgent,
            "repo_auditor": RepoAuditorAgent,
            "dep_manager": DepManagerAgent,
        }
        
        for agent_name, agent_class in agent_classes.items():
            if agent_name in self.config.agents and self.config.agents[agent_name].enabled:
                try:
                    agent = agent_class(
                        agent_id=agent_name,
                        config=self.config,
                        shared_memory=self.shared_memory,
                        vector_db=self.vector_db
                    )
                    self.agents[agent_name] = agent
                    logger.info(f"Initialized {agent_name} agent")
                except Exception as e:
                    logger.error(f"Failed to initialize {agent_name} agent: {e}")
    
    def _build_graph(self) -> None:
        """Build the LangGraph workflow."""
        # Create state graph
        self.graph = StateGraph(SwarmState)
        
        # Add agent nodes
        for agent_name, agent in self.agents.items():
            self.graph.add_node(agent_name, self._create_agent_node(agent))
        
        # Add edges and conditional routing
        self._add_graph_edges()
        
        # Set up checkpointer for state persistence (using MemorySaver for now)
        # TODO: Implement persistent checkpointing when langgraph-checkpoint is available
        self.checkpointer = MemorySaver()
        
        # Compile the graph
        self.compiled_graph = self.graph.compile(checkpointer=self.checkpointer)
        
        logger.info("LangGraph workflow compiled successfully")
    
    def _create_agent_node(self, agent: BaseAgent) -> Callable:
        """Create a node function for an agent."""
        async def agent_node(state: SwarmState) -> SwarmState:
            try:
                # Update agent state
                agent_state = AgentState(
                    agent_id=agent.agent_id,
                    status="busy",
                    current_task=state.current_task,
                    last_heartbeat=datetime.now(UTC)
                )
                self.shared_memory.update_agent_state(agent_state)
                
                # Execute agent
                result = await agent.execute(state)
                
                # Update agent state to idle
                agent_state.status = "idle"
                agent_state.current_task = None
                self.shared_memory.update_agent_state(agent_state)
                
                return result
            except Exception as e:
                logger.error(f"Agent {agent.agent_id} failed: {e}")
                
                # Update agent state to error
                agent_state = AgentState(
                    agent_id=agent.agent_id,
                    status="error",
                    current_task=state.current_task,
                    metadata={"error": str(e)}
                )
                self.shared_memory.update_agent_state(agent_state)
                
                # Log error to shared memory
                error_entry = MemoryEntry(
                    agent_id="swarm",
                    type="error",
                    content={"error": str(e), "context": "swarm_loop"},
                    parent_id=None
                )
                self.shared_memory.add_memory(error_entry)
                
                return state
        
        return agent_node
    
    def _add_graph_edges(self) -> None:
        """Add edges and conditional routing to the graph."""
        # Start with commander agent
        self.graph.add_edge(START, "commander")
        
        # Commander decides next actions
        self.graph.add_conditional_edges(
            "commander",
            self._route_from_commander,
            {
                "planner": "planner",
                "git_watcher": "git_watcher",
                "repo_auditor": "repo_auditor",
                "end": END
            }
        )
        
        # Planner distributes tasks
        self.graph.add_conditional_edges(
            "planner",
            self._route_from_planner,
            {
                "code": "code",
                "docs": "docs",
                "qa_test": "qa_test",
                "impact_mapper": "impact_mapper",
                "dep_manager": "dep_manager",
                "commander": "commander"
            }
        )
        
        # All agents can return to commander or end
        for agent_name in ["code", "docs", "qa_test", "git_watcher", 
                          "impact_mapper", "repo_auditor", "dep_manager"]:
            if agent_name in self.agents:
                self.graph.add_conditional_edges(
                    agent_name,
                    self._route_to_commander_or_end,
                    {
                        "commander": "commander",
                        "end": END
                    }
                )
    
    def _route_from_commander(self, state: SwarmState) -> str:
        """Route from commander based on current needs and agent availability."""
        try:
            # Check if there are pending tasks
            if state.pending_tasks:
                return "planner"
            
            # Check current task status
            if state.current_task:
                task = self.shared_memory.get_task(state.current_task)
                if task and task.status == "running":
                    # Task is still running, wait
                    return "end"
            
            # Check agent health and availability
            agent_states = self.shared_memory.get_agent_states()
            available_agents = []
            
            for agent_id, agent_state in agent_states.items():
                if (agent_state.status in ["idle", "available"] and 
                    agent_id not in ["commander", "planner"]):
                    available_agents.append(agent_id)
            
            # Prioritized agent activation based on system needs
            
            # 1. Check if repositories need monitoring (high priority)
            if ("git_watcher" in self.agents and 
                "git_watcher" in available_agents and
                not any("git_watcher" in agent for agent in state.active_agents)):
                
                last_git_check = state.metadata.get("last_git_check", 0)
                if datetime.now(UTC).timestamp() - last_git_check > 300:  # 5 minutes
                    return "git_watcher"
            
            # 2. Check if repositories need auditing (medium priority)
            if ("repo_auditor" in self.agents and 
                "repo_auditor" in available_agents):
                
                last_audit = state.metadata.get("last_audit", 0)
                if datetime.now(UTC).timestamp() - last_audit > 3600:  # 1 hour
                    return "repo_auditor"
            
            # 3. Check for dependency updates (low priority)
            if ("dep_manager" in self.agents and 
                "dep_manager" in available_agents):
                
                last_dep_check = state.metadata.get("last_dependency_check", 0)
                if datetime.now(UTC).timestamp() - last_dep_check > 86400:  # 24 hours
                    return "dep_manager"
            
            # If no specific tasks, end cycle
            return "end"
            
        except Exception as e:
            logger.error(f"Error in commander routing: {e}")
            return "end"
    
    def _route_from_planner(self, state: SwarmState) -> str:
        """Route from planner to appropriate agent with intelligent assignment."""
        try:
            if not state.pending_tasks:
                return "commander"
            
            # Get the next highest priority task
            task_id = self._get_next_priority_task(state.pending_tasks)
            if not task_id:
                return "commander"
            
            task = self.shared_memory.get_task(task_id)
            if not task:
                # Remove invalid task from pending list
                if task_id in state.pending_tasks:
                    state.pending_tasks.remove(task_id)
                return "commander"
            
            # Enhanced routing based on task analysis
            target_agent = self._analyze_task_for_agent_assignment(task)
            
            # Check agent availability and load balancing
            if self._is_agent_available(target_agent, state):
                # Update state to track active assignment
                state.current_task = task_id
                if target_agent not in state.active_agents:
                    state.active_agents.append(target_agent)
                
                # Remove from pending, will be managed by agent
                if task_id in state.pending_tasks:
                    state.pending_tasks.remove(task_id)
                
                logger.info(f"Planner assigning task {task_id} to {target_agent}")
                return target_agent
            else:
                # Agent busy, try fallback or defer
                fallback_agent = self._find_fallback_agent(target_agent, task, state)
                if fallback_agent:
                    logger.info(f"Primary agent {target_agent} busy, using fallback {fallback_agent}")
                    return fallback_agent
                
                # No agents available, return to commander for later retry
                logger.warning(f"No available agents for task {task_id}, deferring")
                return "commander"
        
        except Exception as e:
            logger.error(f"Error in planner routing: {e}")
            return "commander"
    
    def _get_next_priority_task(self, pending_tasks: list[str]) -> str | None:
        """Get the highest priority task from pending list."""
        if not pending_tasks:
            return None
        
        # Get task details and sort by priority
        task_priorities: list[tuple[str, int, datetime]] = []
        for task_id in pending_tasks:
            task = self.shared_memory.get_task(task_id)
            if task:
                priority_map = {"high": 3, "medium": 2, "low": 1}
                priority_value = priority_map.get(task.metadata.get("priority", "medium"), 2)
                task_priorities.append((task_id, priority_value, task.created_at))
        
        if not task_priorities:
            return None
        
        # Sort by priority (desc), then by creation time (asc)
        task_priorities.sort(key=lambda x: (-x[1], x[2]))
        return task_priorities[0][0]
    
    def _analyze_task_for_agent_assignment(self, task: TaskStatus) -> str:
        """Analyze task content and determine the best agent assignment."""
        task_description = task.description.lower()
        task_type = task.metadata.get("type", "").lower()
        
        # Keyword-based analysis with priority weighting
        agent_scores = {
            "code": 0,
            "qa_test": 0, 
            "docs": 0,
            "git_watcher": 0,
            "impact_mapper": 0,
            "repo_auditor": 0,
            "dep_manager": 0
        }
        
        # Code-related keywords (highest weight)
        code_keywords = ["implement", "refactor", "function", "class", "method", "code", 
                        "python", "javascript", "typescript", "debug", "fix", "bug"]
        for keyword in code_keywords:
            if keyword in task_description:
                agent_scores["code"] += 3
        
        # Testing keywords
        test_keywords = ["test", "pytest", "unittest", "coverage", "quality", "lint", 
                        "format", "check", "validate"]
        for keyword in test_keywords:
            if keyword in task_description:
                agent_scores["qa_test"] += 3
        
        # Documentation keywords
        doc_keywords = ["document", "readme", "docs", "docstring", "comment", "api doc"]
        for keyword in doc_keywords:
            if keyword in task_description:
                agent_scores["docs"] += 3
        
        # Git/repository keywords
        git_keywords = ["commit", "branch", "merge", "git", "repository", "repo", "push", "pull"]
        for keyword in git_keywords:
            if keyword in task_description:
                agent_scores["git_watcher"] += 2
        
        # Impact analysis keywords
        impact_keywords = ["impact", "dependency", "breaking", "compatibility", "api change"]
        for keyword in impact_keywords:
            if keyword in task_description:
                agent_scores["impact_mapper"] += 2
        
        # Repository audit keywords
        audit_keywords = ["scan", "audit", "missing", "file", "structure", "organize"]
        for keyword in audit_keywords:
            if keyword in task_description:
                agent_scores["repo_auditor"] += 2
        
        # Dependency keywords
        dep_keywords = ["dependency", "package", "requirements", "version", "update", "upgrade"]
        for keyword in dep_keywords:
            if keyword in task_description:
                agent_scores["dep_manager"] += 2
        
        # Task type overrides (if explicitly specified)
        task_type_map = {
            "code_generation": "code",
            "code_refactor": "code",
            "testing": "qa_test",
            "documentation": "docs",
            "git_monitoring": "git_watcher",
            "impact_analysis": "impact_mapper",
            "repository_audit": "repo_auditor",
            "dependency_management": "dep_manager"
        }
        
        if task_type in task_type_map:
            agent_scores[task_type_map[task_type]] += 5  # Strong override
        
        # Return agent with highest score, default to code if tie
        best_agent = max(agent_scores.keys(), key=lambda x: agent_scores[x])
        if agent_scores[best_agent] == 0:
            return "code"  # Default fallback
        
        return best_agent
    
    def _is_agent_available(self, agent_name: str, state: SwarmState) -> bool:
        """Check if an agent is available for new tasks."""
        if agent_name not in self.agents:
            return False
        
        # Check if agent is already handling maximum tasks
        current_load = state.active_agents.count(agent_name)
        max_concurrent = 1  # Default to 1 concurrent task per agent
        
        if current_load >= max_concurrent:
            return False
        
        # Check agent state in shared memory
        try:
            agent_states = self.shared_memory.get_agent_states()
            agent_state = agent_states.get(agent_name)
            
            if not agent_state:
                return True  # Assume available if no state
            
            return agent_state.status in ["idle", "available"]
        except Exception as e:
            logger.warning(f"Could not check agent {agent_name} availability: {e}")
            return False
    
    def _find_fallback_agent(
        self, primary_agent: str, task: TaskStatus, state: SwarmState
    ) -> str | None:
        """Find a fallback agent when primary agent is unavailable."""
        # Define fallback mappings
        fallback_map = {
            "code": ["qa_test"],  # QA can sometimes handle simple code tasks
            "qa_test": ["code"],  # Code agent can run tests
            "docs": ["code"],     # Code agent can generate basic docs
            "git_watcher": ["repo_auditor"],  # Similar repository operations
            "repo_auditor": ["git_watcher"],  # Similar repository operations
        }
        
        fallbacks = fallback_map.get(primary_agent, [])
        
        for fallback_agent in fallbacks:
            if self._is_agent_available(fallback_agent, state):
                return fallback_agent
        
        return None
    
    def _route_to_commander_or_end(self, state: SwarmState) -> str:
        """Route back to commander or end based on state."""
        if state.pending_tasks or state.current_task:
            return "commander"
        return "end"
    
    async def start(self) -> None:
        """Start the swarm operation."""
        if self.is_running:
            logger.warning("Swarm is already running")
            return
        
        self.is_running = True
        self._shutdown_event.clear()
        
        logger.info("Starting DevGuard swarm")
        
        # Initialize all repositories in vector database
        await self._initialize_repositories()
        
        # Start the main swarm loop
        asyncio.create_task(self._swarm_loop())
        
        logger.info("DevGuard swarm started successfully")
    
    async def stop(self) -> None:
        """Stop the swarm operation."""
        if not self.is_running:
            logger.warning("Swarm is not running")
            return
        
        logger.info("Stopping DevGuard swarm")
        
        self.is_running = False
        self._shutdown_event.set()
        
        # Update all agent states to stopped
        for agent_name in self.agents:
            agent_state = AgentState(
                agent_id=agent_name,
                status="stopped",
                current_task=None
            )
            self.shared_memory.update_agent_state(agent_state)
        
        logger.info("DevGuard swarm stopped")
    
    async def _swarm_loop(self) -> None:
        """Main swarm execution loop."""
        while self.is_running:
            try:
                # Create initial state
                initial_state = SwarmState(
                    repositories={repo.path: {"branch": repo.branch} 
                                for repo in self.config.repositories},
                    last_update=datetime.now(UTC)
                )
                
                # Execute the graph
                thread_id = f"swarm_{datetime.now(UTC).timestamp()}"
                config = {"configurable": {"thread_id": thread_id}}
                
                async for state in self.compiled_graph.astream(
                    initial_state.model_dump(),
                    config=config
                ):
                    if not self.is_running:
                        break
                    
                    logger.debug(f"Swarm state update: {list(state.keys())}")
                
                # Wait for next cycle or shutdown
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.config.swarm_interval
                    )
                    break  # Shutdown requested
                except TimeoutError:
                    continue  # Normal cycle timeout
                    
            except Exception as e:
                logger.error(f"Error in swarm loop: {e}")
                
                # Log error to shared memory
                error_entry = MemoryEntry(
                    agent_id="swarm",
                    type="error",
                    content={"error": str(e), "context": "swarm_loop"},
                    tags={"error", "swarm_failure"}
                )
                self.shared_memory.add_memory(error_entry)
                
                # Wait before retrying
                await asyncio.sleep(min(30, self.config.swarm_interval))
    
    async def _initialize_repositories(self) -> None:
        """Initialize repositories in the vector database."""
        for repo_config in self.config.repositories:
            try:
                repo_path = Path(repo_config.path)
                if not repo_path.exists():
                    logger.warning(f"Repository path does not exist: {repo_path}")
                    continue
                
                # Add repository files to vector database
                await self._scan_repository(repo_path, repo_config)
                
                logger.info(f"Initialized repository: {repo_path}")
            except Exception as e:
                logger.error(f"Failed to initialize repository {repo_config.path}: {e}")
    
    async def _scan_repository(self, repo_path: Path, repo_config) -> None:
        """Scan repository and add files to vector database."""
        # This is a basic implementation - will be enhanced by RepoAuditorAgent
        for pattern in repo_config.watch_files:
            for file_path in repo_path.rglob(pattern):
                # Check if file should be ignored
                should_ignore = any(
                    file_path.match(ignore_pattern) 
                    for ignore_pattern in repo_config.ignore_patterns
                )
                
                if should_ignore or not file_path.is_file():
                    continue
                
                try:
                    # Read file content
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    
                    # Add to vector database
                    self.vector_db.add_file_content(
                        file_path,
                        content,
                        metadata={
                            "repository": str(repo_path),
                            "branch": repo_config.branch,
                            "last_modified": file_path.stat().st_mtime
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to process file {file_path}: {e}")
    
    def create_task(
        self,
        description: str,
        task_type: str,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Create a new task for the swarm."""
        task = TaskStatus(
            agent_id=agent_id or "planner",
            status="pending",
            description=description,
            metadata=metadata or {"type": task_type}
        )
        
        task_id = self.shared_memory.create_task(task)
        
        # Log task creation
        entry = MemoryEntry(
            agent_id="swarm",
            type="task",
            content={
                "action": "create_task",
                "task_id": task_id,
                "description": description,
                "task_type": task_type
            },
            tags={"task", "creation"},
            parent_id=None
        )
        self.shared_memory.add_memory(entry)
        
        logger.info(f"Created task {task_id}: {description}")
        return task_id
    
    def get_status(self) -> dict[str, Any]:
        """Get current swarm status."""
        agent_states = self.shared_memory.get_all_agent_states()
        recent_tasks = self.shared_memory.get_tasks(limit=10)
        
        # Get vector DB stats
        vector_stats = self.vector_db.get_collection_stats()
        
        return {
            "is_running": self.is_running,
            "agents": {
                state.agent_id: {
                    "status": state.status,
                    "current_task": state.current_task,
                    "last_heartbeat": state.last_heartbeat.isoformat()
                }
                for state in agent_states
            },
            "recent_tasks": [
                {
                    "id": task.id,
                    "status": task.status,
                    "description": task.description,
                    "agent_id": task.agent_id,
                    "created_at": task.created_at.isoformat()
                }
                for task in recent_tasks
            ],
            "repositories": [repo.path for repo in self.config.repositories],
            "vector_db_documents": vector_stats.get('total_documents', 0)
        }

    def pause_agent(self, agent_id: str) -> bool:
        """Pause a specific agent."""
        if agent_id not in self.agents:
            logger.warning(f"Agent {agent_id} not found")
            return False
        
        agent_state = AgentState(
            agent_id=agent_id,
            status="paused",
            current_task=None
        )
        self.shared_memory.update_agent_state(agent_state)
        
        # Log the pause action
        entry = MemoryEntry(
            agent_id="swarm",
            type="control",
            content={
                "action": "pause_agent",
                "agent_id": agent_id,
                "timestamp": datetime.now(UTC).isoformat()
            },
            tags={"control", "pause"},
            parent_id=None
        )
        self.shared_memory.add_memory(entry)
        
        logger.info(f"Agent {agent_id} paused")
        return True

    def resume_agent(self, agent_id: str) -> bool:
        """Resume a paused agent."""
        if agent_id not in self.agents:
            logger.warning(f"Agent {agent_id} not found")
            return False
        
        agent_state = AgentState(
            agent_id=agent_id,
            status="active",
            current_task=None
        )
        self.shared_memory.update_agent_state(agent_state)
        
        # Log the resume action
        entry = MemoryEntry(
            agent_id="swarm",
            type="control",
            content={
                "action": "resume_agent",
                "agent_id": agent_id,
                "timestamp": datetime.now(UTC).isoformat()
            },
            tags={"control", "resume"},
            parent_id=None
        )
        self.shared_memory.add_memory(entry)
        
        logger.info(f"Agent {agent_id} resumed")
        return True

    def inject_task(
        self,
        description: str,
        task_type: str,
        agent_id: str | None = None,
        priority: str = "normal",
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Inject a high-priority task into the system."""
        task_metadata = metadata or {}
        task_metadata.update({
            "type": task_type,
            "priority": priority,
            "injected": True,
            "injected_at": datetime.now(UTC).isoformat()
        })
        
        task = TaskStatus(
            agent_id=agent_id or "planner",
            status="pending",
            description=description,
            metadata=task_metadata
        )
        
        task_id = self.shared_memory.create_task(task)
        
        # Log task injection
        entry = MemoryEntry(
            agent_id="swarm",
            type="control",
            content={
                "action": "inject_task",
                "task_id": task_id,
                "description": description,
                "task_type": task_type,
                "priority": priority,
                "target_agent": agent_id,
                "timestamp": datetime.now(UTC).isoformat()
            },
            tags={"control", "inject", priority},
            parent_id=None
        )
        self.shared_memory.add_memory(entry)
        
        logger.info(f"Injected {priority} priority task {task_id}: {description}")
        return task_id

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task."""
        task = self.shared_memory.get_task(task_id)
        if not task:
            logger.warning(f"Task {task_id} not found")
            return False
        
        if task.status in ["completed", "failed", "cancelled"]:
            logger.warning(f"Task {task_id} is already {task.status}")
            return False
        
        # Update task status
        task.status = "cancelled"
        task.metadata = task.metadata or {}
        task.metadata["cancelled_at"] = datetime.now(UTC).isoformat()
        
        self.shared_memory.update_task(task)
        
        # Log the cancellation
        entry = MemoryEntry(
            agent_id="swarm",
            type="control",
            content={
                "action": "cancel_task",
                "task_id": task_id,
                "description": task.description,
                "timestamp": datetime.now(UTC).isoformat()
            },
            tags={"control", "cancel"},
            parent_id=None
        )
        self.shared_memory.add_memory(entry)
        
        logger.info(f"Task {task_id} cancelled")
        return True

    def get_task_details(self, task_id: str) -> dict[str, Any] | None:
        """Get detailed information about a specific task."""
        task = self.shared_memory.get_task(task_id)
        if not task:
            return None
        
        return {
            "id": task.id,
            "status": task.status,
            "description": task.description,
            "agent_id": task.agent_id,
            "created_at": task.created_at.isoformat(),
            "updated_at": task.updated_at.isoformat(),
            "metadata": task.metadata
        }

    def get_agent_details(self, agent_id: str) -> dict[str, Any] | None:
        """Get detailed information about a specific agent."""
        if agent_id not in self.agents:
            return None
        
        agent_state = self.shared_memory.get_agent_state(agent_id)
        agent = self.agents[agent_id]
        
        return {
            "id": agent_id,
            "status": agent_state.status if agent_state else "unknown",
            "current_task": agent_state.current_task if agent_state else None,
            "last_heartbeat": agent_state.last_heartbeat.isoformat() if agent_state else "never",
            "capabilities": getattr(agent, 'get_capabilities', lambda: [])(),
            "enabled": self.config.agents.get(agent_id, {}).get("enabled", False)
        }

    def list_agents(self) -> list[dict[str, Any]]:
        """List all agents with their current status."""
        return [
            self.get_agent_details(agent_id) 
            for agent_id in self.agents.keys()
        ]

    def list_tasks(
        self, 
        status: str | None = None, 
        agent_id: str | None = None,
        limit: int = 20
    ) -> list[dict[str, Any]]:
        """List tasks with optional filtering."""
        tasks = self.shared_memory.get_tasks(limit=limit)
        
        filtered_tasks = []
        for task in tasks:
            if status and task.status != status:
                continue
            if agent_id and task.agent_id != agent_id:
                continue
            
            filtered_tasks.append({
                "id": task.id,
                "status": task.status,
                "description": task.description,
                "agent_id": task.agent_id,
                "created_at": task.created_at.isoformat(),
                "updated_at": task.updated_at.isoformat(),
                "metadata": task.metadata
            })
        
        return filtered_tasks
