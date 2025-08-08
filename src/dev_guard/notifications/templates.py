"""Notification templates and template management."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .base import NotificationLevel, NotificationMessage


class NotificationTemplate(BaseModel):
    """Template for formatting notification messages."""
    
    name: str = Field(..., description="Template name")
    title_template: str = Field(..., description="Title template")
    content_template: str = Field(..., description="Content template")
    level: Optional[NotificationLevel] = Field(None, description="Default level")
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def render(
        self, 
        context: Dict[str, Any], 
        **kwargs
    ) -> NotificationMessage:
        """Render template with provided context.
        
        Args:
            context: Template context variables
            **kwargs: Additional message parameters
            
        Returns:
            Rendered notification message
        """
        # Basic string formatting - could be enhanced with Jinja2
        title = self.title_template.format(**context)
        content = self.content_template.format(**context)
        
        return NotificationMessage(
            title=title,
            content=content,
            level=kwargs.get('level', self.level or NotificationLevel.INFO),
            source=kwargs.get('source', 'template_system'),
            tags=kwargs.get('tags', []) + self.tags,
            metadata={**self.metadata, **kwargs.get('metadata', {})},
            template_name=self.name
        )


class TemplateManager:
    """Manager for notification templates."""
    
    def __init__(self):
        """Initialize template manager."""
        self.templates: Dict[str, NotificationTemplate] = {}
        self._load_default_templates()
    
    def register_template(self, template: NotificationTemplate) -> None:
        """Register a notification template.
        
        Args:
            template: Template to register
        """
        self.templates[template.name] = template
    
    def get_template(self, name: str) -> Optional[NotificationTemplate]:
        """Get template by name.
        
        Args:
            name: Template name
            
        Returns:
            Template if found, None otherwise
        """
        return self.templates.get(name)
    
    def list_templates(self) -> List[str]:
        """List all available template names.
        
        Returns:
            List of template names
        """
        return list(self.templates.keys())
    
    def render_template(
        self,
        template_name: str,
        context: Dict[str, Any],
        **kwargs
    ) -> Optional[NotificationMessage]:
        """Render template with context.
        
        Args:
            template_name: Name of template to render
            context: Template context variables
            **kwargs: Additional message parameters
            
        Returns:
            Rendered message or None if template not found
        """
        template = self.get_template(template_name)
        if not template:
            return None
        
        return template.render(context, **kwargs)
    
    def _load_default_templates(self) -> None:
        """Load default notification templates."""
        
        # System startup template
        self.register_template(NotificationTemplate(
            name="system_startup",
            title_template="🚀 DevGuard System Started",
            content_template=(
                "DevGuard autonomous swarm has started successfully.\n"
                "• Version: {version}\n"
                "• Agents: {agent_count}\n"
                "• Repositories: {repo_count}\n"
                "• Started at: {timestamp}"
            ),
            level=NotificationLevel.INFO,
            tags=["system", "startup"]
        ))
        
        # System shutdown template
        self.register_template(NotificationTemplate(
            name="system_shutdown",
            title_template="🛑 DevGuard System Stopped",
            content_template=(
                "DevGuard system has been shut down.\n"
                "• Uptime: {uptime}\n"
                "• Tasks completed: {completed_tasks}\n"
                "• Stopped at: {timestamp}"
            ),
            level=NotificationLevel.INFO,
            tags=["system", "shutdown"]
        ))
        
        # Critical error template
        self.register_template(NotificationTemplate(
            name="critical_error",
            title_template="🚨 Critical Error in DevGuard",
            content_template=(
                "A critical error has occurred:\n"
                "• Agent: {agent_name}\n"
                "• Error: {error_message}\n"
                "• Task: {task_id}\n"
                "• Time: {timestamp}\n"
                "\nImmediate attention required!"
            ),
            level=NotificationLevel.CRITICAL,
            tags=["error", "critical"]
        ))
        
        # Task completion template
        self.register_template(NotificationTemplate(
            name="task_completed",
            title_template="✅ Task Completed: {task_type}",
            content_template=(
                "Task completed successfully:\n"
                "• Task ID: {task_id}\n"
                "• Agent: {agent_name}\n"
                "• Duration: {duration}\n"
                "• Result: {result_summary}\n"
                "• Files changed: {files_changed}"
            ),
            level=NotificationLevel.INFO,
            tags=["task", "completion"]
        ))
        
        # Security alert template
        self.register_template(NotificationTemplate(
            name="security_alert",
            title_template="🔒 Security Alert: {vulnerability_type}",
            content_template=(
                "Security vulnerability detected:\n"
                "• Type: {vulnerability_type}\n"
                "• Severity: {severity}\n"
                "• Repository: {repository}\n"
                "• File: {file_path}\n"
                "• Recommendation: {recommendation}\n"
                "• CVE: {cve_id}"
            ),
            level=NotificationLevel.WARNING,
            tags=["security", "vulnerability"]
        ))
        
        # Dependency update template
        self.register_template(NotificationTemplate(
            name="dependency_update",
            title_template="📦 Dependency Update Available",
            content_template=(
                "New dependency updates available:\n"
                "• Package: {package_name}\n"
                "• Current: {current_version}\n"
                "• Latest: {latest_version}\n"
                "• Type: {update_type}\n"
                "• Repository: {repository}\n"
                "• Breaking changes: {breaking_changes}"
            ),
            level=NotificationLevel.INFO,
            tags=["dependency", "update"]
        ))
        
        # Git repository changes template
        self.register_template(NotificationTemplate(
            name="git_changes",
            title_template="📝 Repository Changes Detected",
            content_template=(
                "Changes detected in repository:\n"
                "• Repository: {repository}\n"
                "• Branch: {branch}\n"
                "• Commits: {commit_count}\n"
                "• Files changed: {files_changed}\n"
                "• Impact analysis: {impact_status}\n"
                "• Last commit: {last_commit}"
            ),
            level=NotificationLevel.INFO,
            tags=["git", "changes"]
        ))
        
        # Agent health issue template
        self.register_template(NotificationTemplate(
            name="agent_health",
            title_template="⚠️ Agent Health Issue: {agent_name}",
            content_template=(
                "Agent health issue detected:\n"
                "• Agent: {agent_name}\n"
                "• Status: {status}\n"
                "• Last heartbeat: {last_heartbeat}\n"
                "• Error count: {error_count}\n"
                "• Action taken: {action_taken}"
            ),
            level=NotificationLevel.WARNING,
            tags=["agent", "health"]
        ))
        
        # User intervention required template
        self.register_template(NotificationTemplate(
            name="user_intervention",
            title_template="👤 User Intervention Required",
            content_template=(
                "Manual intervention needed:\n"
                "• Reason: {reason}\n"
                "• Context: {context}\n"
                "• Suggested action: {suggested_action}\n"
                "• Priority: {priority}\n"
                "• Deadline: {deadline}\n"
                "\nPlease review and take appropriate action."
            ),
            level=NotificationLevel.WARNING,
            tags=["intervention", "manual"]
        ))
        
        # Performance alert template
        self.register_template(NotificationTemplate(
            name="performance_alert",
            title_template="📊 Performance Alert: {metric_name}",
            content_template=(
                "Performance threshold exceeded:\n"
                "• Metric: {metric_name}\n"
                "• Current value: {current_value}\n"
                "• Threshold: {threshold}\n"
                "• Duration: {duration}\n"
                "• Affected components: {components}\n"
                "• Recommended action: {recommendation}"
            ),
            level=NotificationLevel.WARNING,
            tags=["performance", "alert"]
        ))
