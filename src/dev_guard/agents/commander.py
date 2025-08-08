"""Commander Agent for system oversight and coordination."""

import logging
from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent
from ..core.config import Config
from ..notifications import NotificationManager, NotificationLevel

logger = logging.getLogger(__name__)

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class CommanderAgent(BaseAgent):
    """Commander Agent oversees system, communicates with user, monitors swarm state."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.info("Commander Agent initialized")
        
        # Initialize notification manager if config available
        self.notification_manager = None
        if hasattr(self, 'config') and hasattr(self.config, 'notifications'):
            try:
                self.notification_manager = NotificationManager(self.config.notifications)
                self.logger.info("Notification manager initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize notification manager: {e}")
    
    async def execute(self, state: Any) -> Any:
        """Execute commander logic - system oversight and coordination."""
        try:
            self.set_status("busy", "system_oversight")
            self.update_heartbeat()
            
            # Log system status
            self.log_observation(
                "Commander performing system oversight",
                data={"state": state.model_dump() if hasattr(state, 'model_dump') else str(state)},
                tags=["system_overview"]
            )
            
            # Check agent health
            await self._check_agent_health()
            
            # Process pending notifications
            await self._process_notifications()
            
            # Evaluate system status
            system_status = await self._evaluate_system_status()
            
            # Make decisions based on system state
            next_action = await self._decide_next_action(state, system_status)
            
            self.set_status("idle")
            return state
            
        except Exception as e:
            self.logger.error(f"Commander execution failed: {e}")
            self.set_status("error")
            raise
    
    async def _check_agent_health(self) -> Dict[str, Any]:
        """Check the health of all agents in the swarm."""
        agent_states = self.shared_memory.get_all_agent_states()
        current_time = datetime.now(timezone.utc)
        
        health_report = {
            "healthy": [],
            "unhealthy": [],
            "unresponsive": []
        }
        
        for agent_state in agent_states:
            if agent_state.agent_id == self.agent_id:
                continue  # Skip self
            
            # Check if agent is responsive (heartbeat within last 5 minutes)
            time_since_heartbeat = current_time - agent_state.last_heartbeat
            if time_since_heartbeat.total_seconds() > 300:  # 5 minutes
                health_report["unresponsive"].append(agent_state.agent_id)
            elif agent_state.status == "error":
                health_report["unhealthy"].append(agent_state.agent_id)
            else:
                health_report["healthy"].append(agent_state.agent_id)
        
        self.log_observation(
            "Agent health check completed",
            data=health_report,
            tags=["health_check", "monitoring"]
        )
        
        # Take action for unhealthy agents
        if health_report["unhealthy"] or health_report["unresponsive"]:
            await self._handle_unhealthy_agents(health_report)
        
        return health_report
    
    async def _handle_unhealthy_agents(self, health_report: Dict[str, Any]) -> None:
        """Handle unhealthy or unresponsive agents."""
        for agent_id in health_report["unhealthy"]:
            self.log_decision(
                f"Agent {agent_id} is unhealthy, attempting recovery",
                f"Agent {agent_id} reported error status",
                data={"agent_id": agent_id, "action": "recovery_attempt"},
                tags=["agent_recovery"]
            )
            
            # Could implement agent restart logic here
            
        for agent_id in health_report["unresponsive"]:
            self.log_decision(
                f"Agent {agent_id} is unresponsive, marking for investigation",
                f"Agent {agent_id} hasn't sent heartbeat in >5 minutes",
                data={"agent_id": agent_id, "action": "investigate"},
                tags=["agent_investigation"]
            )
    
    async def _process_notifications(self) -> None:
        """Process any pending notifications or alerts."""
        if not self.notification_manager:
            return
        
        # Get recent error entries
        recent_errors = self.get_recent_memories(
            memory_type="error",
            limit=10,
            include_other_agents=True
        )
        
        critical_errors = [
            error for error in recent_errors
            if "critical" in error.tags or "swarm_failure" in error.tags
        ]
        
        if critical_errors:
            self.log_observation(
                f"Found {len(critical_errors)} critical errors requiring attention",
                data={"error_count": len(critical_errors)},
                tags=["critical_alerts"]
            )
            
            # Send critical error notifications
            await self._send_critical_error_notifications(critical_errors)
        
        # Check for system health issues
        health_report = await self.check_agent_health()
        if health_report["unresponsive"] or health_report["unhealthy"]:
            await self._send_health_alert_notifications(health_report)
    
    async def _send_critical_error_notifications(self, errors: List[Any]) -> None:
        """Send notifications for critical errors."""
        try:
            for error in errors[:3]:  # Limit to prevent spam
                context = {
                    "agent_name": getattr(error, 'agent_id', 'Unknown'),
                    "error_message": getattr(error, 'content', 'Unknown error'),
                    "task_id": getattr(error, 'metadata', {}).get('task_id', 'Unknown'),
                    "timestamp": getattr(error, 'timestamp', 'Unknown').strftime('%Y-%m-%d %H:%M:%S') if hasattr(getattr(error, 'timestamp', None), 'strftime') else str(getattr(error, 'timestamp', 'Unknown'))
                }
                
                await self.notification_manager.send_templated_notification(
                    template_name="critical_error",
                    context=context,
                    source="commander_agent",
                    level=NotificationLevel.CRITICAL
                )
        except Exception as e:
            self.logger.error(f"Failed to send critical error notifications: {e}")
    
    async def _send_health_alert_notifications(self, health_report: Dict[str, Any]) -> None:
        """Send notifications for agent health issues."""
        try:
            for agent_id in health_report["unresponsive"][:2]:  # Limit to prevent spam
                context = {
                    "agent_name": agent_id,
                    "status": "unresponsive",
                    "last_heartbeat": "Unknown",
                    "error_count": "0",
                    "action_taken": "Investigation initiated"
                }
                
                await self.notification_manager.send_templated_notification(
                    template_name="agent_health",
                    context=context,
                    source="commander_agent",
                    level=NotificationLevel.WARNING
                )
        except Exception as e:
            self.logger.error(f"Failed to send health alert notifications: {e}")
    
    async def _evaluate_system_status(self) -> Dict[str, Any]:
        """Evaluate overall system status."""
        # Get recent task statistics
        recent_tasks = self.shared_memory.get_tasks(limit=50)
        
        task_stats = {
            "total": len(recent_tasks),
            "pending": len([t for t in recent_tasks if t.status == "pending"]),
            "running": len([t for t in recent_tasks if t.status == "running"]),
            "completed": len([t for t in recent_tasks if t.status == "completed"]),
            "failed": len([t for t in recent_tasks if t.status == "failed"])
        }
        
        # Calculate success rate
        completed_or_failed = task_stats["completed"] + task_stats["failed"]
        success_rate = (
            task_stats["completed"] / completed_or_failed 
            if completed_or_failed > 0 else 1.0
        )
        
        # Get vector DB statistics
        vector_doc_count = self.vector_db.count_documents()
        
        system_status = {
            "task_stats": task_stats,
            "success_rate": success_rate,
            "vector_documents": vector_doc_count,
            "repositories_monitored": len(self.config.repositories)
        }
        
        self.log_observation(
            "System status evaluation completed",
            data=system_status,
            tags=["system_status"]
        )
        
        return system_status
    
    async def _decide_next_action(self, state: Any, system_status: Dict[str, Any]) -> str:
        """Decide what action to take next based on system state."""
        task_stats = system_status["task_stats"]
        
        # Priority 1: If there are failed tasks, investigate
        if task_stats["failed"] > 0:
            self.log_decision(
                "Investigating failed tasks",
                f"Found {task_stats['failed']} failed tasks that need investigation",
                data={"failed_count": task_stats["failed"]},
                tags=["task_investigation"]
            )
            return "investigate_failures"
        
        # Priority 2: If there are many pending tasks, delegate to planner
        if task_stats["pending"] > 5:
            self.log_decision(
                "Delegating to planner for task distribution",
                f"Found {task_stats['pending']} pending tasks that need assignment",
                data={"pending_count": task_stats["pending"]},
                tags=["task_delegation"]
            )
            return "delegate_to_planner"
        
        # Priority 3: If vector DB is empty or has few documents, audit repos
        if system_status["vector_documents"] < 100:
            self.log_decision(
                "Initiating repository audit",
                f"Vector DB has only {system_status['vector_documents']} documents",
                data={"doc_count": system_status["vector_documents"]},
                tags=["repo_audit"]
            )
            return "audit_repositories"
        
        # Priority 4: If success rate is low, analyze issues
        if system_status["success_rate"] < 0.8:
            self.log_decision(
                "Analyzing system performance issues",
                f"Success rate is {system_status['success_rate']:.2%}, below 80% threshold",
                data={"success_rate": system_status["success_rate"]},
                tags=["performance_analysis"]
            )
            return "analyze_performance"
        
        # Default: Monitor repositories for changes
        self.log_decision(
            "Initiating repository monitoring",
            "System is healthy, monitoring repositories for changes",
            tags=["routine_monitoring"]
        )
        return "monitor_repositories"
    
    async def handle_user_request(self, request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle a direct user request."""
        self.log_observation(
            f"Received user request: {request}",
            data={"request": request, "context": context or {}},
            tags=["user_request"]
        )
        
        # Parse and categorize the request
        request_type = await self._categorize_request(request)
        
        # Create appropriate task based on request type
        task_id = None
        if request_type == "code_generation":
            task_id = self.create_task(
                description=f"User requested: {request}",
                task_type="code_generation",
                target_agent="code",
                metadata={"user_request": True, "context": context}
            )
        elif request_type == "documentation":
            task_id = self.create_task(
                description=f"User requested: {request}",
                task_type="documentation",
                target_agent="docs",
                metadata={"user_request": True, "context": context}
            )
        elif request_type == "analysis":
            task_id = self.create_task(
                description=f"User requested: {request}",
                task_type="impact_analysis",
                target_agent="impact_mapper",
                metadata={"user_request": True, "context": context}
            )
        else:
            # General request - delegate to planner
            task_id = self.create_task(
                description=f"User requested: {request}",
                task_type="general",
                target_agent="planner",
                metadata={"user_request": True, "context": context}
            )
        
        response = {
            "status": "accepted",
            "task_id": task_id,
            "request_type": request_type,
            "message": f"Request accepted and assigned to appropriate agent. Task ID: {task_id}"
        }
        
        self.log_result(
            f"User request processed and assigned to task {task_id}",
            data=response,
            tags=["user_response"]
        )
        
        return response
    
    async def _categorize_request(self, request: str) -> str:
        """Categorize a user request to determine appropriate handler."""
        request_lower = request.lower()
        
        # Simple keyword-based categorization
        if any(word in request_lower for word in ["code", "implement", "function", "class", "method"]):
            return "code_generation"
        elif any(word in request_lower for word in ["document", "docs", "readme", "comment"]):
            return "documentation"
        elif any(word in request_lower for word in ["analyze", "impact", "dependenc", "relationship"]):
            return "analysis"
        elif any(word in request_lower for word in ["test", "testing", "qa", "quality"]):
            return "testing"
        else:
            return "general"
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get a comprehensive system overview for user/monitoring."""
        agent_states = self.shared_memory.get_all_agent_states()
        recent_tasks = self.shared_memory.get_tasks(limit=20)
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agents": {
                state.agent_id: {
                    "status": state.status,
                    "current_task": state.current_task,
                    "last_heartbeat": state.last_heartbeat.isoformat(),
                    "metadata": state.metadata
                }
                for state in agent_states
            },
            "task_summary": {
                "total_recent": len(recent_tasks),
                "by_status": {
                    status: len([t for t in recent_tasks if t.status == status])
                    for status in ["pending", "running", "completed", "failed"]
                }
            },
            "repositories": len(self.config.repositories),
            "vector_documents": self.vector_db.count_documents()
        }
