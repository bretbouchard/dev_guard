"""Base agent class for DevGuard agents."""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from ..core.config import Config
from ..memory.shared_memory import (
    SharedMemory, MemoryEntry, TaskStatus, AgentState
)
from ..memory.vector_db import VectorDatabase

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all DevGuard agents."""
    
    def __init__(
        self,
        agent_id: str,
        config: Config,
        shared_memory: SharedMemory,
        vector_db: VectorDatabase
    ):
        self.agent_id = agent_id
        self.config = config
        self.agent_config = config.get_agent_config(agent_id)
        self.shared_memory = shared_memory
        self.vector_db = vector_db
        self.logger = logging.getLogger(f"{__name__}.{agent_id}")
        
        # Initialize agent state
        self._initialize_state()
    
    def _initialize_state(self) -> None:
        """Initialize agent state in shared memory."""
        initial_state = AgentState(
            agent_id=self.agent_id,
            status="idle",
            current_task=None,
            last_heartbeat=datetime.now(timezone.utc),
            metadata={"initialized": True}
        )
        self.shared_memory.update_agent_state(initial_state)
        self.logger.info(f"Agent {self.agent_id} initialized")
    
    @abstractmethod
    async def execute(self, state: Any) -> Any:
        """Execute the agent's main logic."""
        pass
    
    async def execute_with_retry(self, state: Any) -> Any:
        """Execute with retry logic based on agent configuration."""
        max_retries = self.agent_config.max_retries
        retry_delay = self.agent_config.retry_delay
        
        for attempt in range(max_retries + 1):
            try:
                # Set timeout for execution
                return await asyncio.wait_for(
                    self.execute(state),
                    timeout=self.agent_config.timeout
                )
            except asyncio.TimeoutError:
                error_msg = (
                    f"Agent {self.agent_id} timed out after "
                    f"{self.agent_config.timeout}s"
                )
                self.logger.error(error_msg)
                
                if attempt < max_retries:
                    self.logger.info(
                        f"Retrying in {retry_delay}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(retry_delay)
                    continue
                
                self._log_error(error_msg, state)
                raise
            except Exception as e:
                error_msg = f"Agent {self.agent_id} failed: {e}"
                self.logger.error(error_msg)
                
                if attempt < max_retries:
                    self.logger.info(
                        f"Retrying in {retry_delay}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(retry_delay)
                    continue
                
                self._log_error(error_msg, state, exception=e)
                raise
        
        return state
    
    def _log_error(
        self,
        error_msg: str,
        state: Any,
        exception: Optional[Exception] = None
    ) -> None:
        """Log error to shared memory."""
        error_entry = MemoryEntry(
            agent_id=self.agent_id,
            type="error",
            content={
                "error": error_msg,
                "exception": str(exception) if exception else None,
                "state": (
                    state.model_dump() if hasattr(state, 'model_dump')
                    else str(state)
                )
            },
            tags={"error", "agent_failure", self.agent_id}
        )
        self.shared_memory.add_memory(error_entry)
    
    def log_observation(
        self,
        observation: str,
        data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Log an observation to shared memory."""
        entry = MemoryEntry(
            agent_id=self.agent_id,
            type="observation",
            content={
                "observation": observation,
                "data": data or {}
            },
            tags=set(tags or []) | {"observation", self.agent_id}
        )
        
        entry_id = self.shared_memory.add_memory(entry)
        self.logger.debug(f"Logged observation: {observation}")
        return entry_id
    
    def log_decision(
        self,
        decision: str,
        reasoning: str,
        data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Log a decision to shared memory."""
        entry = MemoryEntry(
            agent_id=self.agent_id,
            type="decision",
            content={
                "decision": decision,
                "reasoning": reasoning,
                "data": data or {}
            },
            tags=set(tags or []) | {"decision", self.agent_id}
        )
        
        entry_id = self.shared_memory.add_memory(entry)
        self.logger.info(f"Logged decision: {decision}")
        return entry_id
    
    def log_result(
        self,
        result: str,
        data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        parent_id: Optional[str] = None
    ) -> str:
        """Log a result to shared memory."""
        entry = MemoryEntry(
            agent_id=self.agent_id,
            type="result",
            content={
                "result": result,
                "data": data or {}
            },
            tags=set(tags or []) | {"result", self.agent_id},
            parent_id=parent_id
        )
        
        entry_id = self.shared_memory.add_memory(entry)
        self.logger.info(f"Logged result: {result}")
        return entry_id
    
    def get_recent_memories(
        self,
        memory_type: Optional[str] = None,
        limit: int = 50,
        include_other_agents: bool = False
    ) -> List[MemoryEntry]:
        """Get recent memories, optionally filtered by type."""
        agent_id = None if include_other_agents else self.agent_id
        return self.shared_memory.get_memories(
            agent_id=agent_id,
            memory_type=memory_type,
            limit=limit
        )
    
    def search_knowledge(
        self,
        query: str,
        n_results: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search the vector database for relevant knowledge."""
        return self.vector_db.search(query, n_results=n_results, where=filters)
    
    def update_heartbeat(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update agent heartbeat in shared memory."""
        current_state = self.shared_memory.get_agent_state(self.agent_id)
        if current_state:
            current_state.last_heartbeat = datetime.now(timezone.utc)
            if metadata:
                current_state.metadata.update(metadata)
            self.shared_memory.update_agent_state(current_state)
    
    def set_status(self, status: str, current_task: Optional[str] = None) -> None:
        """Update agent status in shared memory."""
        agent_state = AgentState(
            agent_id=self.agent_id,
            status=status,
            current_task=current_task,
            last_heartbeat=datetime.now(timezone.utc)
        )
        self.shared_memory.update_agent_state(agent_state)
        self.logger.debug(f"Status updated to: {status}")
    
    def get_agent_state(self) -> Optional[AgentState]:
        """Get the current state of this agent."""
        return self.shared_memory.get_agent_state(self.agent_id)
    
    def get_all_agent_states(self) -> List[AgentState]:
        """Get the current state of all agents."""
        return self.shared_memory.get_all_agent_states()
    
    def get_active_agents(self) -> List[AgentState]:
        """Get all agents that are currently active (not idle or stopped)."""
        all_states = self.shared_memory.get_all_agent_states()
        return [state for state in all_states if state.status in ["busy", "error"]]
    
    def get_available_agents(self) -> List[AgentState]:
        """Get all agents that are available for task assignment."""
        all_states = self.shared_memory.get_all_agent_states()
        return [state for state in all_states if state.status == "idle"]
    
    def update_agent_metadata(self, metadata: Dict[str, Any]) -> None:
        """Update agent metadata in shared memory."""
        current_state = self.shared_memory.get_agent_state(self.agent_id)
        if current_state:
            current_state.metadata.update(metadata)
            current_state.last_heartbeat = datetime.now(timezone.utc)
            self.shared_memory.update_agent_state(current_state)
            self.logger.debug(f"Updated metadata for agent {self.agent_id}")
    
    def get_agent_capabilities(self) -> Dict[str, Any]:
        """Get agent capabilities and configuration from metadata."""
        current_state = self.shared_memory.get_agent_state(self.agent_id)
        capabilities = {
            "agent_id": self.agent_id,
            "enabled": self.is_enabled(),
            "priority": self.agent_config.priority,
            "max_concurrent_tasks": self.agent_config.max_concurrent_tasks,
            "heartbeat_interval": self.agent_config.heartbeat_interval,
            "custom_instructions": self.get_custom_instructions()
        }
        
        if current_state:
            capabilities.update({
                "current_status": current_state.status,
                "current_task": current_state.current_task,
                "last_heartbeat": current_state.last_heartbeat,
                "metadata": current_state.metadata
            })
        
        return capabilities
    
    def check_agent_health(self) -> Dict[str, Any]:
        """Check the health of this agent and return health status."""
        current_state = self.shared_memory.get_agent_state(self.agent_id)
        health_status = {
            "agent_id": self.agent_id,
            "status": "healthy",
            "last_heartbeat": None,
            "heartbeat_age_seconds": None,
            "current_task": None,
            "issues": []
        }
        
        if current_state:
            health_status["last_heartbeat"] = current_state.last_heartbeat
            health_status["current_task"] = current_state.current_task
            
            # Calculate heartbeat age
            now = datetime.now(timezone.utc)
            heartbeat_age = (now - current_state.last_heartbeat).total_seconds()
            health_status["heartbeat_age_seconds"] = heartbeat_age
            
            # Check for potential issues
            heartbeat_threshold = self.agent_config.heartbeat_interval * 3
            if heartbeat_age > heartbeat_threshold:
                health_status["status"] = "unhealthy"
                health_status["issues"].append(f"Heartbeat too old: {heartbeat_age:.1f}s")
            
            if current_state.status == "error":
                health_status["status"] = "error"
                health_status["issues"].append("Agent is in error state")
        
        return health_status
    
    def create_task(
        self,
        description: str,
        task_type: str,
        target_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a task for another agent or the swarm."""
        task = TaskStatus(
            agent_id=target_agent or "planner",
            status="pending",
            description=description,
            metadata={
                "type": task_type,
                "created_by": self.agent_id,
                **(metadata or {})
            }
        )
        
        task_id = self.shared_memory.create_task(task)
        
        # Log task creation
        self.log_observation(
            f"Created task for {target_agent or 'planner'}: {description}",
            data={"task_id": task_id, "task_type": task_type},
            tags=["task_creation"]
        )
        
        return task_id
    
    def get_pending_tasks(self) -> List[TaskStatus]:
        """Get pending tasks assigned to this agent."""
        return self.shared_memory.get_tasks(agent_id=self.agent_id, status="pending")
    
    def get_all_tasks(self, status: Optional[str] = None, limit: int = 50) -> List[TaskStatus]:
        """Get all tasks for this agent, optionally filtered by status."""
        return self.shared_memory.get_tasks(agent_id=self.agent_id, status=status, limit=limit)
    
    def get_task_by_id(self, task_id: str) -> Optional[TaskStatus]:
        """Get a specific task by ID."""
        return self.shared_memory.get_task(task_id)
    
    def assign_task_to_self(self, task_id: str) -> bool:
        """Assign a task to this agent and update status to running."""
        task = self.shared_memory.get_task(task_id)
        if not task:
            self.logger.warning(f"Task {task_id} not found for assignment")
            return False
        
        # Update task status to running and assign to this agent
        updates = {
            "status": "running",
            "metadata": {
                **task.metadata,
                "assigned_to": self.agent_id,
                "assigned_at": datetime.now(timezone.utc).isoformat()
            }
        }
        
        success = self.shared_memory.update_task(task_id, **updates)
        if success:
            # Update agent state to reflect current task
            self.set_status("busy", current_task=task_id)
            self.log_observation(
                f"Assigned task {task_id} to self",
                data={"task_description": task.description, "task_id": task_id},
                tags=["task_assignment"]
            )
        
        return success
    
    def complete_current_task(self, result: Optional[Dict[str, Any]] = None) -> bool:
        """Complete the current task assigned to this agent."""
        current_state = self.shared_memory.get_agent_state(self.agent_id)
        if not current_state or not current_state.current_task:
            self.logger.warning("No current task to complete")
            return False
        
        task_id = current_state.current_task
        success = self.update_task_status(task_id, "completed", result=result)
        
        if success:
            # Clear current task from agent state
            self.set_status("idle")
            self.log_observation(
                f"Completed task {task_id}",
                data={"task_id": task_id, "result": result},
                tags=["task_completion"]
            )
        
        return success
    
    def fail_current_task(self, error: str, result: Optional[Dict[str, Any]] = None) -> bool:
        """Mark the current task as failed."""
        current_state = self.shared_memory.get_agent_state(self.agent_id)
        if not current_state or not current_state.current_task:
            self.logger.warning("No current task to fail")
            return False
        
        task_id = current_state.current_task
        success = self.update_task_status(task_id, "failed", result=result, error=error)
        
        if success:
            # Clear current task from agent state and set error status
            self.set_status("error")
            self.log_observation(
                f"Failed task {task_id}: {error}",
                data={"task_id": task_id, "error": error, "result": result},
                tags=["task_failure"]
            )
        
        return success
    
    def update_task_status(
        self,
        task_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> bool:
        """Update a task's status and result."""
        updates = {"status": status}
        if result is not None:
            updates["result"] = result
        if error is not None:
            updates["error"] = error
        
        success = self.shared_memory.update_task(task_id, **updates)
        
        if success:
            self.log_observation(
                f"Updated task {task_id} status to {status}",
                data={"task_id": task_id, "result": result, "error": error},
                tags=["task_update"]
            )
        
        return success
    
    def cancel_task(self, task_id: str, reason: Optional[str] = None) -> bool:
        """Cancel a task and update its status."""
        updates = {"status": "cancelled"}
        if reason:
            updates["metadata"] = {
                "cancelled_by": self.agent_id,
                "cancelled_reason": reason,
                "cancelled_at": datetime.now(timezone.utc).isoformat()
            }
        
        success = self.shared_memory.update_task(task_id, **updates)
        
        if success:
            self.log_observation(
                f"Cancelled task {task_id}",
                data={"task_id": task_id, "reason": reason},
                tags=["task_cancellation"]
            )
            
            # If this was the current task, update agent state
            current_state = self.shared_memory.get_agent_state(self.agent_id)
            if current_state and current_state.current_task == task_id:
                self.set_status("idle")
        
        return success
    
    def get_task_dependencies(self, task_id: str) -> List[TaskStatus]:
        """Get all dependencies for a task."""
        task = self.shared_memory.get_task(task_id)
        if not task or not task.dependencies:
            return []
        
        dependencies = []
        for dep_id in task.dependencies:
            dep_task = self.shared_memory.get_task(dep_id)
            if dep_task:
                dependencies.append(dep_task)
        
        return dependencies
    
    def check_task_dependencies_completed(self, task_id: str) -> bool:
        """Check if all dependencies for a task are completed."""
        dependencies = self.get_task_dependencies(task_id)
        for dep_task in dependencies:
            if dep_task.status not in ["completed", "cancelled"]:
                return False
        return True
    
    def create_dependent_task(
        self,
        description: str,
        task_type: str,
        depends_on: List[str],
        target_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a task that depends on other tasks."""
        # Validate that dependency tasks exist
        for dep_id in depends_on:
            dep_task = self.shared_memory.get_task(dep_id)
            if not dep_task:
                raise ValueError(f"Dependency task {dep_id} does not exist")
        
        task = TaskStatus(
            agent_id=target_agent or "planner",
            status="pending",
            description=description,
            metadata={
                "type": task_type,
                "created_by": self.agent_id,
                "depends_on": depends_on,
                **(metadata or {})
            },
            dependencies=depends_on
        )
        
        task_id = self.shared_memory.create_task(task)
        
        # Log task creation
        self.log_observation(
            f"Created dependent task for {target_agent or 'planner'}: {description}",
            data={"task_id": task_id, "task_type": task_type, "dependencies": depends_on},
            tags=["task_creation", "dependent_task"]
        )
        
        return task_id
    
    def get_tasks_by_status(self, status: str, limit: int = 50) -> List[TaskStatus]:
        """Get tasks for this agent filtered by status."""
        return self.shared_memory.get_tasks(agent_id=self.agent_id, status=status, limit=limit)
    
    def get_task_history(self, limit: int = 100) -> List[TaskStatus]:
        """Get the task history for this agent, ordered by creation time."""
        all_tasks = self.shared_memory.get_tasks(agent_id=self.agent_id, limit=limit)
        # Filter out pending tasks and sort by creation time
        completed_tasks = [task for task in all_tasks if task.status in ["completed", "failed", "cancelled"]]
        return sorted(completed_tasks, key=lambda x: x.created_at, reverse=True)
    
    def get_repository_files(
        self,
        repo_path: str,
        file_extensions: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get files from a repository in the vector database."""
        # Use search_files with empty query to get all files, then filter by repository
        all_files = self.vector_db.search_files(
            query="", 
            file_extensions=file_extensions, 
            n_results=1000  # Large number to get all files
        )
        
        # Filter by repository path
        return [
            file for file in all_files 
            if file.get("metadata", {}).get("repository") == repo_path
        ]
    
    def search_code(
        self,
        query: str,
        repo_path: Optional[str] = None,
        file_extensions: Optional[List[str]] = None,
        n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for code in the vector database."""
        filters = {"content_type": "code"}
        if repo_path:
            filters["repository"] = repo_path
        if file_extensions:
            filters["file_extension"] = {"$in": file_extensions}
        
        return self.vector_db.search(query, n_results=n_results, where=filters)
    
    async def execute_command(
        self,
        command: str,
        cwd: Optional[Union[str, Path]] = None,
        timeout: float = 300.0
    ) -> Dict[str, Any]:
        """Execute a shell command and return the result."""
        self.logger.debug(f"Executing command: {command}")
        
        try:
            if cwd:
                cwd = Path(cwd)
            
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
            
            result: Dict[str, Any] = {
                "command": command,
                "return_code": process.returncode,
                "stdout": stdout.decode('utf-8', errors='ignore').strip(),
                "stderr": stderr.decode('utf-8', errors='ignore').strip(),
                "success": process.returncode == 0
            }
            
            self.log_observation(
                f"Executed command: {command}",
                data=result,
                tags=["command_execution"]
            )
            
            return result
            
        except asyncio.TimeoutError:
            error_msg = f"Command timed out: {command}"
            self.logger.error(error_msg)
            raise
        except Exception as e:
            error_msg = f"Command failed: {command} - {e}"
            self.logger.error(error_msg)
            raise
    
    def get_custom_instructions(self) -> Optional[str]:
        """Get custom instructions for this agent from configuration."""
        return self.agent_config.custom_instructions
    
    def is_enabled(self) -> bool:
        """Check if this agent is enabled in configuration."""
        return self.agent_config.enabled
