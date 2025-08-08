"""Planner agent for DevGuard - orchestrates tasks across the swarm."""

import json
import logging
from datetime import UTC, datetime
from typing import Any

from ..memory.shared_memory import AgentState, MemoryEntry
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    """
    Planner agent responsible for:
    - Analyzing complex tasks and breaking them down
    - Distributing work to appropriate agents
    - Monitoring task progress and dependencies
    - Coordinating multi-agent workflows
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Remove LLM provider from kwargs since we'll get it from config
        self.llm_provider = kwargs.get('llm_provider')
        
        # Planning-specific configuration  
        try:
            self.max_task_depth = getattr(self.agent_config, 'max_task_depth', 5)
            self.parallel_task_limit = getattr(self.agent_config, 'parallel_task_limit', 3)
        except AttributeError:
            self.max_task_depth = 5
            self.parallel_task_limit = 3
        
    async def execute(self, state: Any) -> Any:
        """Execute the planner agent's main logic."""
        # Extract task from state
        if isinstance(state, dict):
            task = state
        else:
            # Assume state has task information
            task = {"type": "analyze_and_plan", "description": str(state)}
        
        """
        Execute a planning task.
        
        Args:
            task: Task dictionary with type, description, and parameters
            
        Returns:
            Execution result with subtasks and routing decisions
        """
        try:
            self.logger.info(f"Planner executing task: {task.get('type', 'unknown')}")
            
            # Update agent state
            self._update_state("busy", task.get("task_id"))
            
            task_type = task.get("type", "")
            
            if task_type == "analyze_and_plan":
                result = await self._analyze_and_plan(task)
            elif task_type == "route_task":
                result = await self._route_task(task)
            elif task_type == "monitor_progress":
                result = await self._monitor_progress(task)
            elif task_type == "coordinate_agents":
                result = await self._coordinate_agents(task)
            else:
                result = await self._generic_planning(task)
            
            self._update_state("idle")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in planner task execution: {e}")
            self._update_state("error", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "subtasks": []
            }
    
    async def _analyze_and_plan(self, task: dict[str, Any]) -> dict[str, Any]:
        """Analyze a complex task and create an execution plan."""
        try:
            description = task.get("description", "")
            context = task.get("context", {})
            
            # Get relevant context from vector database
            relevant_docs = await self.vector_db.search(
                query=description,
                limit=5,
                filter_metadata={"type": "code"}
            )
            
            # Use LLM to analyze and create plan if available
            if self.llm_provider:
                plan = await self._llm_analyze_task(description, context, relevant_docs)
            else:
                plan = await self._heuristic_planning(description, context)
            
            # Create subtasks based on the plan
            subtasks = self._create_subtasks(plan, task.get("task_id"))
            
            # Store plan in shared memory
            plan_entry = MemoryEntry(
                agent_id=self.agent_id,
                type="result",  # Use "result" as it's one of the allowed values
                content={
                    "plan_type": "task_analysis",
                    "plan_data": plan,
                    "task_id": task.get("task_id"),
                    "subtask_count": len(subtasks)
                },
                tags={"plan", "task_analysis"}
            )
            self.shared_memory.add_memory(plan_entry)
            
            return {
                "success": True,
                "plan": plan,
                "subtasks": subtasks,
                "next_action": "distribute_tasks"
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing and planning: {e}")
            return {
                "success": False,
                "error": str(e),
                "subtasks": []
            }
    
    async def _route_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Route a task to the most appropriate agent."""
        try:
            task_description = task.get("description", "")
            task_type = task.get("task_type", "")
            
            # Determine best agent for the task
            target_agent = self._determine_target_agent(task_description, task_type)
            
            # Check agent availability
            agent_states = self.shared_memory.get_agent_states()
            if target_agent in agent_states:
                agent_state = agent_states[target_agent]
                if agent_state.status == "busy":
                    # Find alternative or queue the task
                    target_agent = self._find_alternative_agent(task_description, task_type)
            
            return {
                "success": True,
                "target_agent": target_agent,
                "routing_reason": f"Best match for {task_type} task",
                "priority": task.get("priority", "medium")
            }
            
        except Exception as e:
            self.logger.error(f"Error routing task: {e}")
            return {
                "success": False,
                "error": str(e),
                "target_agent": "code_agent"  # Default fallback
            }
    
    async def _monitor_progress(self, task: dict[str, Any]) -> dict[str, Any]:
        """Monitor progress of ongoing tasks."""
        try:
            # Get all active tasks
            tasks = self.shared_memory.get_tasks(status="running")
            
            progress_report = {
                "active_tasks": len(tasks),
                "task_details": [],
                "bottlenecks": [],
                "recommendations": []
            }
            
            for task_entry in tasks:
                task_detail = {
                    "task_id": task_entry.id,
                    "agent_id": task_entry.agent_id,
                    "started_at": task_entry.created_at.isoformat(),
                    "description": task_entry.description[:100] + "..." if len(task_entry.description) > 100 else task_entry.description
                }
                progress_report["task_details"].append(task_detail)
                
                # Check for long-running tasks
                duration = datetime.now(UTC) - task_entry.created_at
                if duration.total_seconds() > 300:  # 5 minutes
                    progress_report["bottlenecks"].append({
                        "task_id": task_entry.id,
                        "duration_minutes": duration.total_seconds() / 60,
                        "agent_id": task_entry.agent_id
                    })
            
            # Add recommendations based on analysis
            if len(progress_report["bottlenecks"]) > 0:
                progress_report["recommendations"].append("Consider redistributing long-running tasks")
            
            return {
                "success": True,
                "progress": progress_report,
                "next_action": "continue_monitoring"
            }
            
        except Exception as e:
            self.logger.error(f"Error monitoring progress: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _coordinate_agents(self, task: dict[str, Any]) -> dict[str, Any]:
        """Coordinate multiple agents for complex workflows."""
        try:
            workflow_type = task.get("workflow_type", "sequential")
            agents_needed = task.get("agents", [])
            
            coordination_plan = {
                "workflow_type": workflow_type,
                "agents": agents_needed,
                "coordination_steps": []
            }
            
            if workflow_type == "sequential":
                coordination_plan["coordination_steps"] = [
                    f"Execute {step}" for step in agents_needed
                ]
            elif workflow_type == "parallel":
                coordination_plan["coordination_steps"] = [
                    f"Parallel execution: {', '.join(agents_needed)}"
                ]
            else:
                coordination_plan["coordination_steps"] = [
                    "Custom workflow coordination"
                ]
            
            return {
                "success": True,
                "coordination_plan": coordination_plan,
                "next_action": "execute_workflow"
            }
            
        except Exception as e:
            self.logger.error(f"Error coordinating agents: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _generic_planning(self, task: dict[str, Any]) -> dict[str, Any]:
        """Generic planning for unknown task types."""
        try:
            description = task.get("description", "")
            
            # Simple heuristic planning
            plan = {
                "approach": "analyze_then_execute",
                "steps": [
                    "Gather context and requirements",
                    "Analyze task complexity",
                    "Execute or delegate appropriately"
                ],
                "estimated_complexity": "medium"
            }
            
            return {
                "success": True,
                "plan": plan,
                "next_action": "route_to_appropriate_agent"
            }
            
        except Exception as e:
            self.logger.error(f"Error in generic planning: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _llm_analyze_task(
        self, 
        description: str, 
        context: dict[str, Any], 
        relevant_docs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Use LLM to analyze task and create detailed plan."""
        try:
            # Construct prompt for task analysis
            prompt = f"""
            As a DevGuard planning agent, analyze this task and create a detailed execution plan.
            
            Task Description: {description}
            Context: {json.dumps(context, indent=2)}
            
            Relevant Code Context:
            {self._format_relevant_docs(relevant_docs)}
            
            Please provide a structured plan with:
            1. Task breakdown into subtasks
            2. Recommended agent assignments
            3. Dependencies between subtasks
            4. Estimated complexity and time
            5. Risk assessment
            
            Respond in JSON format.
            """
            
            response = await self.llm_provider.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1  # Low temperature for structured planning
            )
            
            # Parse LLM response
            try:
                plan = json.loads(response.content)
            except json.JSONDecodeError:
                # Fallback to heuristic if LLM response isn't valid JSON
                plan = await self._heuristic_planning(description, context)
            
            return plan
            
        except Exception as e:
            self.logger.error(f"Error in LLM task analysis: {e}")
            return await self._heuristic_planning(description, context)
    
    async def _heuristic_planning(
        self, 
        description: str, 
        context: dict[str, Any]
    ) -> dict[str, Any]:
        """Fallback heuristic planning when LLM is not available."""
        
        # Simple keyword-based analysis
        keywords = description.lower().split()
        
        plan = {
            "approach": "heuristic",
            "complexity": "medium",
            "subtasks": [],
            "recommended_agents": [],
            "dependencies": []
        }
        
        # Code-related task
        if any(keyword in keywords for keyword in ["code", "function", "class", "refactor", "debug"]):
            plan["recommended_agents"] = ["code_agent"]
            plan["subtasks"] = [
                "Analyze existing code",
                "Implement changes",
                "Test modifications"
            ]
        
        # Documentation task
        elif any(keyword in keywords for keyword in ["document", "readme", "docs"]):
            plan["recommended_agents"] = ["code_agent"]  # Will expand when we have a docs agent
            plan["subtasks"] = [
                "Analyze codebase structure",
                "Generate documentation",
                "Update existing docs"
            ]
        
        # Generic task
        else:
            plan["recommended_agents"] = ["code_agent"]
            plan["subtasks"] = [
                "Analyze requirements",
                "Execute task",
                "Verify results"
            ]
        
        return plan
    
    def _create_subtasks(self, plan: dict[str, Any], parent_task_id: str | None) -> list[dict[str, Any]]:
        """Create subtask definitions from a plan."""
        subtasks = []
        
        subtask_definitions = plan.get("subtasks", [])
        recommended_agents = plan.get("recommended_agents", ["code_agent"])
        
        for i, subtask_desc in enumerate(subtask_definitions):
            subtask = {
                "task_id": f"{parent_task_id}_subtask_{i}" if parent_task_id else f"subtask_{i}",
                "type": "execute",
                "description": subtask_desc,
                "parent_task_id": parent_task_id,
                "priority": "medium",
                "recommended_agent": recommended_agents[0] if recommended_agents else "code_agent",
                "dependencies": plan.get("dependencies", [])
            }
            subtasks.append(subtask)
        
        return subtasks
    
    def _determine_target_agent(self, description: str, task_type: str) -> str:
        """Determine the best agent for a given task."""
        description_lower = description.lower()
        
        # Commander patterns (high-level coordination) - check first
        if any(pattern in description_lower for pattern in [
            "coordinate", "manage", "orchestrate", "supervise", "control"
        ]):
            return "commander"
        
        # Code-related patterns
        elif any(pattern in description_lower for pattern in [
            "code", "function", "class", "method", "refactor", "debug",
            "implement", "fix", "bug", "test", "python", "javascript"
        ]):
            return "code_agent"
        
        # Default to code agent for now
        return "code_agent"
    
    def _find_alternative_agent(self, description: str, task_type: str) -> str:
        """Find alternative agent when primary choice is busy."""
        # For now, fallback to code_agent
        # In the future, we might have multiple code agents or specialized agents
        return "code_agent"
    
    def _format_relevant_docs(self, docs: list[dict[str, Any]]) -> str:
        """Format relevant documents for LLM context."""
        if not docs:
            return "No relevant code context found."
        
        formatted = []
        for i, doc in enumerate(docs[:3]):  # Limit to top 3 results
            metadata = doc.get("metadata", {})
            content = doc.get("content", "")[:500]  # Truncate for brevity
            
            formatted.append(f"""
            Document {i+1}:
            File: {metadata.get('file_path', 'unknown')}
            Type: {metadata.get('file_type', 'unknown')}
            Content Preview: {content}...
            """)
        
        return "\n".join(formatted)
    
    def _update_state(self, status: str, task_id: str | None = None, error: str | None = None) -> None:
        """Update agent state in shared memory."""
        state = AgentState(
            agent_id=self.agent_id,
            status=status,
            current_task=task_id,
            last_heartbeat=datetime.now(UTC),
            metadata={
                "error": error if error else None,
                "capabilities": ["planning", "task_routing", "coordination"]
            }
        )
        self.shared_memory.update_agent_state(state)
    
    def get_capabilities(self) -> list[str]:
        """Return list of agent capabilities."""
        return [
            "task_analysis",
            "task_breakdown", 
            "agent_routing",
            "progress_monitoring",
            "workflow_coordination",
            "dependency_management"
        ]
    
    def get_status(self) -> dict[str, Any]:
        """Get current agent status."""
        return {
            "agent_id": self.agent_id,
            "type": "planner",
            "capabilities": self.get_capabilities(),
            "config": {
                "max_task_depth": self.max_task_depth,
                "parallel_task_limit": self.parallel_task_limit
            }
        }
