"""Discord notification provider."""

import logging

import aiohttp
from pydantic import BaseModel

from .base import NotificationLevel, NotificationMessage, NotificationProvider, NotificationResult

logger = logging.getLogger(__name__)


class DiscordWebhookMessage(BaseModel):
    """Discord webhook message format."""
    
    content: str | None = None
    embeds: list[dict] | None = None


class DiscordProvider(NotificationProvider):
    """Discord webhook notification provider."""
    
    def __init__(
        self,
        webhook_url: str,
        username: str | None = "DevGuard",
        avatar_url: str | None = None,
        supported_levels: list[NotificationLevel] | None = None,
        enabled: bool = True
    ):
        """Initialize Discord provider.
        
        Args:
            webhook_url: Discord webhook URL
            username: Bot username for messages
            avatar_url: Bot avatar URL
            supported_levels: Notification levels to handle
            enabled: Whether provider is enabled
        """
        super().__init__("discord", enabled)
        self.webhook_url = webhook_url
        self.username = username
        self.avatar_url = avatar_url
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
        """Send notification to Discord.
        
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
            embed = self._create_embed(message)
            webhook_message = DiscordWebhookMessage(
                embeds=[embed]
            )
            
            payload = webhook_message.dict(exclude_none=True)
            if self.username:
                payload["username"] = self.username
            if self.avatar_url:
                payload["avatar_url"] = self.avatar_url
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload
                ) as response:
                    if response.status == 204:
                        logger.debug(
                            f"Discord notification sent: {message.title}"
                        )
                        return NotificationResult(
                            success=True,
                            provider=self.name,
                            message_id=str(response.headers.get('X-RateLimit-Remaining', ''))
                        )
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"Discord notification failed: {response.status} - {error_text}"
                        )
                        return NotificationResult(
                            success=False,
                            provider=self.name,
                            error=f"HTTP {response.status}: {error_text}"
                        )
        
        except Exception as e:
            logger.error(f"Discord notification error: {e}")
            return NotificationResult(
                success=False,
                provider=self.name,
                error=str(e)
            )
    
    def _create_embed(self, message: NotificationMessage) -> dict:
        """Create Discord embed from notification message.
        
        Args:
            message: Notification message
            
        Returns:
            Discord embed dictionary
        """
        # Color based on notification level
        color_map = {
            NotificationLevel.DEBUG: 0x808080,      # Gray
            NotificationLevel.INFO: 0x00BFFF,       # Blue
            NotificationLevel.WARNING: 0xFFA500,    # Orange
            NotificationLevel.ERROR: 0xFF6B6B,      # Red
            NotificationLevel.CRITICAL: 0x8B0000    # Dark Red
        }
        
        embed = {
            "title": message.title,
            "description": message.content,
            "color": color_map.get(message.level, 0x00BFFF),
            "timestamp": message.timestamp.isoformat(),
            "fields": [
                {
                    "name": "Source",
                    "value": message.source,
                    "inline": True
                },
                {
                    "name": "Level",
                    "value": message.level.value,
                    "inline": True
                }
            ]
        }
        
        # Add tags field if present
        if message.tags:
            embed["fields"].append({
                "name": "Tags",
                "value": ", ".join(message.tags),
                "inline": True
            })
        
        # Add metadata fields if present
        if message.metadata:
            for key, value in message.metadata.items():
                if len(embed["fields"]) < 25:  # Discord limit
                    embed["fields"].append({
                        "name": key.replace("_", " ").title(),
                        "value": str(value)[:1024],  # Discord field limit
                        "inline": True
                    })
        
        # Add footer with template info if available
        if message.template_name:
            embed["footer"] = {
                "text": f"Template: {message.template_name}"
            }
        
        return embed
