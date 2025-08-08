"""Slack notification provider."""

import logging
from typing import Dict, List, Optional

import aiohttp
from pydantic import BaseModel

from .base import (
    NotificationLevel,
    NotificationMessage,
    NotificationProvider,
    NotificationResult
)

logger = logging.getLogger(__name__)


class SlackMessage(BaseModel):
    """Slack webhook message format."""
    
    text: Optional[str] = None
    blocks: Optional[List[Dict]] = None
    attachments: Optional[List[Dict]] = None


class SlackProvider(NotificationProvider):
    """Slack webhook notification provider."""
    
    def __init__(
        self,
        webhook_url: str,
        channel: Optional[str] = None,
        username: Optional[str] = "DevGuard",
        icon_emoji: Optional[str] = ":robot_face:",
        supported_levels: Optional[List[NotificationLevel]] = None,
        enabled: bool = True
    ):
        """Initialize Slack provider.
        
        Args:
            webhook_url: Slack webhook URL
            channel: Override channel (if different from webhook default)
            username: Bot username for messages
            icon_emoji: Bot icon emoji
            supported_levels: Notification levels to handle
            enabled: Whether provider is enabled
        """
        super().__init__("slack", enabled)
        self.webhook_url = webhook_url
        self.channel = channel
        self.username = username
        self.icon_emoji = icon_emoji
        self.supported_levels = supported_levels or [
            NotificationLevel.WARNING,
            NotificationLevel.ERROR,
            NotificationLevel.CRITICAL
        ]
    
    def supports_level(self, level: NotificationLevel) -> bool:
        """Check if provider supports notification level.
        
        Args:
            level: Notification level to check
            
        Returns:
            True if level is supported
        """
        return level in self.supported_levels
    
    async def send_message(
        self,
        message: NotificationMessage
    ) -> NotificationResult:
        """Send notification to Slack.
        
        Args:
            message: Message to send
            
        Returns:
            Result of sending attempt
        """
        if not self.enabled:
            return NotificationResult(
                success=False,
                provider=self.name,
                error="Provider disabled"
            )
        
        if not self.supports_level(message.level):
            return NotificationResult(
                success=False,
                provider=self.name,
                error=f"Level {message.level.value} not supported"
            )
        
        try:
            attachment = self._create_attachment(message)
            slack_message = SlackMessage(
                text=f"*{message.title}*",
                attachments=[attachment]
            )
            
            payload = slack_message.dict(exclude_none=True)
            if self.channel:
                payload["channel"] = self.channel
            if self.username:
                payload["username"] = self.username
            if self.icon_emoji:
                payload["icon_emoji"] = self.icon_emoji
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload
                ) as response:
                    if response.status == 200:
                        response_text = await response.text()
                        if response_text == "ok":
                            logger.debug(
                                f"Slack notification sent: {message.title}"
                            )
                            return NotificationResult(
                                success=True,
                                provider=self.name,
                                message_id="slack_webhook"
                            )
                        else:
                            logger.error(
                                f"Slack notification failed: {response_text}"
                            )
                            return NotificationResult(
                                success=False,
                                provider=self.name,
                                error=response_text
                            )
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"Slack notification failed: {response.status} - {error_text}"
                        )
                        return NotificationResult(
                            success=False,
                            provider=self.name,
                            error=f"HTTP {response.status}: {error_text}"
                        )
        
        except Exception as e:
            logger.error(f"Slack notification error: {e}")
            return NotificationResult(
                success=False,
                provider=self.name,
                error=str(e)
            )
    
    def _create_attachment(self, message: NotificationMessage) -> Dict:
        """Create Slack attachment from notification message.
        
        Args:
            message: Notification message
            
        Returns:
            Slack attachment dictionary
        """
        # Color based on notification level
        color_map = {
            NotificationLevel.DEBUG: "#808080",      # Gray
            NotificationLevel.INFO: "#00BFFF",       # Blue
            NotificationLevel.WARNING: "#FFA500",    # Orange
            NotificationLevel.ERROR: "#FF6B6B",      # Red
            NotificationLevel.CRITICAL: "#8B0000"    # Dark Red
        }
        
        # Icon based on notification level
        icon_map = {
            NotificationLevel.DEBUG: ":mag:",
            NotificationLevel.INFO: ":information_source:",
            NotificationLevel.WARNING: ":warning:",
            NotificationLevel.ERROR: ":x:",
            NotificationLevel.CRITICAL: ":rotating_light:"
        }
        
        attachment = {
            "color": color_map.get(message.level, "#00BFFF"),
            "text": message.content,
            "fields": [
                {
                    "title": "Level",
                    "value": f"{icon_map.get(message.level, ':question:')} {message.level.value}",
                    "short": True
                },
                {
                    "title": "Source",
                    "value": message.source,
                    "short": True
                },
                {
                    "title": "Time",
                    "value": message.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    "short": True
                }
            ],
            "footer": "DevGuard Notification System",
            "ts": int(message.timestamp.timestamp())
        }
        
        # Add tags field if present
        if message.tags:
            attachment["fields"].append({
                "title": "Tags",
                "value": ", ".join(message.tags),
                "short": True
            })
        
        # Add metadata fields if present
        if message.metadata:
            for key, value in message.metadata.items():
                if len(attachment["fields"]) < 10:  # Keep reasonable
                    attachment["fields"].append({
                        "title": key.replace("_", " ").title(),
                        "value": str(value)[:500],  # Slack field limit
                        "short": len(str(value)) < 50
                    })
        
        # Add template info to footer if available
        if message.template_name:
            attachment["footer"] += f" | Template: {message.template_name}"
        
        return attachment
