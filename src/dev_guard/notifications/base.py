"""Base notification classes and interfaces."""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class NotificationLevel(Enum):
    """Notification severity levels."""
    
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class NotificationMessage(BaseModel):
    """Notification message with metadata."""
    
    title: str = Field(..., description="Message title")
    content: str = Field(..., description="Message content")
    level: NotificationLevel = Field(..., description="Notification level")
    timestamp: datetime = Field(default_factory=datetime.now)
    source: str = Field(..., description="Source agent or system")
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    template_name: str | None = Field(None, description="Template used")


class NotificationResult(BaseModel):
    """Result of notification sending attempt."""
    
    success: bool
    provider: str
    message_id: str | None = None
    error: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)


class NotificationProvider(ABC):
    """Abstract base class for notification providers."""
    
    def __init__(self, name: str, enabled: bool = True):
        """Initialize notification provider.
        
        Args:
            name: Provider name
            enabled: Whether provider is enabled
        """
        self.name = name
        self.enabled = enabled
    
    @abstractmethod
    async def send_message(
        self,
        message: NotificationMessage
    ) -> NotificationResult:
        """Send notification message.
        
        Args:
            message: Message to send
            
        Returns:
            Result of sending attempt
        """
        pass
    
    @abstractmethod
    def supports_level(self, level: NotificationLevel) -> bool:
        """Check if provider supports notification level.
        
        Args:
            level: Notification level to check
            
        Returns:
            True if level is supported
        """
        pass
    
    async def test_connection(self) -> bool:
        """Test provider connection.
        
        Returns:
            True if connection successful
        """
        try:
            test_message = NotificationMessage(
                title="DevGuard Test",
                content="Connection test message",
                level=NotificationLevel.INFO,
                source="notification_system"
            )
            result = await self.send_message(test_message)
            return result.success
        except Exception:
            return False


class NotificationFilter(BaseModel):
    """Filter for controlling which notifications are sent."""
    
    levels: list[NotificationLevel] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    exclude_tags: list[str] = Field(default_factory=list)
    
    def matches(self, message: NotificationMessage) -> bool:
        """Check if message matches filter criteria.
        
        Args:
            message: Message to check
            
        Returns:
            True if message matches filter
        """
        # Check levels
        if self.levels and message.level not in self.levels:
            return False
        
        # Check sources
        if self.sources and message.source not in self.sources:
            return False
        
        # Check include tags
        if self.tags and not any(tag in message.tags for tag in self.tags):
            return False
        
        # Check exclude tags
        if self.exclude_tags and any(tag in message.tags for tag 
                                   in self.exclude_tags):
            return False
        
        return True
