"""Telegram notification provider."""

import logging
from typing import List, Optional

import aiohttp

from .base import (
    NotificationLevel,
    NotificationMessage,
    NotificationProvider,
    NotificationResult
)

logger = logging.getLogger(__name__)


class TelegramProvider(NotificationProvider):
    """Telegram bot notification provider."""
    
    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        parse_mode: str = "Markdown",
        supported_levels: Optional[List[NotificationLevel]] = None,
        enabled: bool = True
    ):
        """Initialize Telegram provider.
        
        Args:
            bot_token: Telegram bot token
            chat_id: Telegram chat ID to send messages to
            parse_mode: Message parsing mode (Markdown, HTML, or None)
            supported_levels: Notification levels to handle
            enabled: Whether provider is enabled
        """
        super().__init__("telegram", enabled)
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.parse_mode = parse_mode
        self.api_url = f"https://api.telegram.org/bot{bot_token}"
        self.supported_levels = supported_levels or [
            NotificationLevel.INFO,
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
        """Send notification to Telegram.
        
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
            formatted_message = self._format_message(message)
            
            payload = {
                "chat_id": self.chat_id,
                "text": formatted_message,
                "parse_mode": self.parse_mode
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/sendMessage",
                    json=payload
                ) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        if response_data.get("ok"):
                            message_id = response_data["result"]["message_id"]
                            logger.debug(
                                f"Telegram notification sent: {message.title}"
                            )
                            return NotificationResult(
                                success=True,
                                provider=self.name,
                                message_id=str(message_id)
                            )
                        else:
                            error_desc = response_data.get(
                                "description", "Unknown error"
                            )
                            logger.error(
                                f"Telegram API error: {error_desc}"
                            )
                            return NotificationResult(
                                success=False,
                                provider=self.name,
                                error=error_desc
                            )
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"Telegram notification failed: {response.status} - {error_text}"
                        )
                        return NotificationResult(
                            success=False,
                            provider=self.name,
                            error=f"HTTP {response.status}: {error_text}"
                        )
        
        except Exception as e:
            logger.error(f"Telegram notification error: {e}")
            return NotificationResult(
                success=False,
                provider=self.name,
                error=str(e)
            )
    
    def _format_message(self, message: NotificationMessage) -> str:
        """Format message for Telegram.
        
        Args:
            message: Notification message
            
        Returns:
            Formatted Telegram message
        """
        # Icon based on notification level
        icon_map = {
            NotificationLevel.DEBUG: "🔍",
            NotificationLevel.INFO: "ℹ️",
            NotificationLevel.WARNING: "⚠️",
            NotificationLevel.ERROR: "❌",
            NotificationLevel.CRITICAL: "🚨"
        }
        
        icon = icon_map.get(message.level, "📢")
        
        if self.parse_mode == "Markdown":
            lines = [
                f"{icon} *{message.title}*",
                "",
                f"**Level:** {message.level.value}",
                f"**Source:** {message.source}",
                f"**Time:** {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                f"**Message:**",
                message.content,
            ]
            
            if message.tags:
                lines.extend([
                    "",
                    f"**Tags:** {', '.join(message.tags)}"
                ])
            
            if message.metadata:
                lines.append("")
                lines.append("**Additional Info:**")
                for key, value in message.metadata.items():
                    lines.append(
                        f"• {key.replace('_', ' ').title()}: {value}"
                    )
            
            if message.template_name:
                lines.extend([
                    "",
                    f"_Template: {message.template_name}_"
                ])
        
        elif self.parse_mode == "HTML":
            lines = [
                f"{icon} <b>{message.title}</b>",
                "",
                f"<b>Level:</b> {message.level.value}",
                f"<b>Source:</b> {message.source}",
                f"<b>Time:</b> {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                f"<b>Message:</b>",
                message.content,
            ]
            
            if message.tags:
                lines.extend([
                    "",
                    f"<b>Tags:</b> {', '.join(message.tags)}"
                ])
            
            if message.metadata:
                lines.append("")
                lines.append("<b>Additional Info:</b>")
                for key, value in message.metadata.items():
                    lines.append(
                        f"• {key.replace('_', ' ').title()}: {value}"
                    )
            
            if message.template_name:
                lines.extend([
                    "",
                    f"<i>Template: {message.template_name}</i>"
                ])
        
        else:  # Plain text
            lines = [
                f"{icon} {message.title}",
                "",
                f"Level: {message.level.value}",
                f"Source: {message.source}",
                f"Time: {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                f"Message:",
                message.content,
            ]
            
            if message.tags:
                lines.extend([
                    "",
                    f"Tags: {', '.join(message.tags)}"
                ])
            
            if message.metadata:
                lines.append("")
                lines.append("Additional Info:")
                for key, value in message.metadata.items():
                    lines.append(
                        f"• {key.replace('_', ' ').title()}: {value}"
                    )
            
            if message.template_name:
                lines.extend([
                    "",
                    f"Template: {message.template_name}"
                ])
        
        formatted = '\n'.join(lines)
        
        # Telegram has a 4096 character limit
        if len(formatted) > 4096:
            formatted = formatted[:4093] + "..."
        
        return formatted
