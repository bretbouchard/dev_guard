"""DevGuard Notification System.

Multi-channel notification system supporting Discord, Slack, Telegram, and 
Email. Provides template-based messaging with customization and filtering 
capabilities.
"""

from .base import NotificationLevel, NotificationMessage, NotificationProvider
from .discord_provider import DiscordProvider
from .email_provider import EmailProvider
from .notification_manager import NotificationManager
from .slack_provider import SlackProvider
from .telegram_provider import TelegramProvider
from .templates import NotificationTemplate, TemplateManager

__all__ = [
    "NotificationProvider",
    "NotificationMessage",
    "NotificationLevel",
    "DiscordProvider",
    "EmailProvider",
    "NotificationManager",
    "SlackProvider",
    "TelegramProvider",
    "NotificationTemplate",
    "TemplateManager",
]
