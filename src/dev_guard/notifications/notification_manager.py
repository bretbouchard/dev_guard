"""Main notification manager orchestrating all providers."""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from ..core.config import NotificationConfig
from .base import NotificationFilter, NotificationMessage, NotificationResult
from .discord_provider import DiscordProvider
from .email_provider import EmailProvider
from .slack_provider import SlackProvider
from .telegram_provider import TelegramProvider
from .templates import TemplateManager

logger = logging.getLogger(__name__)


class NotificationManager:
    """Main notification manager coordinating all providers."""
    
    def __init__(self, config: NotificationConfig):
        """Initialize notification manager.
        
        Args:
            config: Notification configuration
        """
        self.config = config
        self.providers = []
        self.template_manager = TemplateManager()
        self.global_filter = NotificationFilter(
            levels=[getattr(level, 'name', level) for level in config.notification_levels]
        )
        
        self._setup_providers()
    
    def _setup_providers(self) -> None:
        """Set up notification providers based on configuration."""
        # Discord provider
        if self.config.discord_webhook:
            discord_provider = DiscordProvider(
                webhook_url=self.config.discord_webhook,
                enabled=self.config.enabled
            )
            self.providers.append(discord_provider)
            logger.info("Discord notification provider configured")
        
        # Slack provider
        if self.config.slack_webhook:
            slack_provider = SlackProvider(
                webhook_url=self.config.slack_webhook,
                enabled=self.config.enabled
            )
            self.providers.append(slack_provider)
            logger.info("Slack notification provider configured")
        
        # Telegram provider
        if self.config.get_telegram_bot_token() and self.config.telegram_chat_id:
            telegram_provider = TelegramProvider(
                bot_token=self.config.get_telegram_bot_token(),
                chat_id=self.config.telegram_chat_id,
                enabled=self.config.enabled
            )
            self.providers.append(telegram_provider)
            logger.info("Telegram notification provider configured")
        
        # Email provider
        if (self.config.email_smtp_server and 
            self.config.email_username and
            self.config.get_email_password() and
            self.config.email_to):
            
            email_provider = EmailProvider(
                smtp_server=self.config.email_smtp_server,
                smtp_port=self.config.email_smtp_port,
                username=self.config.email_username,
                password=self.config.get_email_password(),
                use_tls=self.config.email_use_tls,
                from_address=self.config.email_from or self.config.email_username,
                to_addresses=self.config.email_to,
                enabled=self.config.enabled
            )
            self.providers.append(email_provider)
            logger.info("Email notification provider configured")
        
        if not self.providers:
            logger.warning("No notification providers configured")
    
    async def send_notification(
        self,
        message: NotificationMessage,
        custom_filter: Optional[NotificationFilter] = None
    ) -> List[NotificationResult]:
        """Send notification through configured providers.
        
        Args:
            message: Message to send
            custom_filter: Custom filter to apply (overrides global)
            
        Returns:
            List of results from all providers
        """
        if not self.config.enabled:
            logger.debug("Notification system disabled")
            return []
        
        # Apply filters
        filter_to_use = custom_filter or self.global_filter
        if not filter_to_use.matches(message):
            logger.debug(f"Message filtered out: {message.title}")
            return []
        
        # Send to all applicable providers
        tasks = []
        for provider in self.providers:
            if provider.enabled and provider.supports_level(message.level):
                tasks.append(provider.send_message(message))
        
        if not tasks:
            logger.warning(f"No providers available for level {message.level.value}")
            return []
        
        # Execute all provider sends concurrently
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    processed_results.append(NotificationResult(
                        success=False,
                        provider="unknown",
                        error=str(result)
                    ))
                else:
                    processed_results.append(result)
            
            # Log summary
            successful = sum(1 for r in processed_results if r.success)
            total = len(processed_results)
            
            if successful > 0:
                logger.info(
                    f"Notification sent successfully to {successful}/{total} providers: {message.title}"
                )
            else:
                logger.error(
                    f"Failed to send notification to all providers: {message.title}"
                )
            
            return processed_results
        
        except Exception as e:
            logger.error(f"Error sending notifications: {e}")
            return [NotificationResult(
                success=False,
                provider="notification_manager",
                error=str(e)
            )]
    
    async def send_templated_notification(
        self,
        template_name: str,
        context: Dict[str, Any],
        custom_filter: Optional[NotificationFilter] = None,
        **kwargs
    ) -> List[NotificationResult]:
        """Send notification using template.
        
        Args:
            template_name: Name of template to use
            context: Template context variables
            custom_filter: Custom filter to apply
            **kwargs: Additional message parameters
            
        Returns:
            List of results from all providers
        """
        message = self.template_manager.render_template(
            template_name=template_name,
            context=context,
            **kwargs
        )
        
        if not message:
            logger.error(f"Template not found: {template_name}")
            return [NotificationResult(
                success=False,
                provider="template_manager",
                error=f"Template '{template_name}' not found"
            )]
        
        return await self.send_notification(message, custom_filter)
    
    async def test_providers(self) -> Dict[str, bool]:
        """Test all configured providers.
        
        Returns:
            Dictionary mapping provider names to test results
        """
        results = {}
        
        for provider in self.providers:
            try:
                success = await provider.test_connection()
                results[provider.name] = success
                logger.info(f"Provider {provider.name} test: {'PASSED' if success else 'FAILED'}")
            except Exception as e:
                results[provider.name] = False
                logger.error(f"Provider {provider.name} test failed: {e}")
        
        return results
    
    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all providers.
        
        Returns:
            Dictionary with provider status information
        """
        status = {}
        
        for provider in self.providers:
            status[provider.name] = {
                "enabled": provider.enabled,
                "supported_levels": [level.value for level in provider.supported_levels],
                "type": provider.__class__.__name__
            }
        
        return status
    
    def register_custom_template(self, template_name: str, title: str, content: str, **kwargs) -> None:
        """Register a custom notification template.
        
        Args:
            template_name: Unique template name
            title: Title template string
            content: Content template string
            **kwargs: Additional template parameters
        """
        from .templates import NotificationTemplate
        
        template = NotificationTemplate(
            name=template_name,
            title_template=title,
            content_template=content,
            **kwargs
        )
        
        self.template_manager.register_template(template)
        logger.info(f"Custom template registered: {template_name}")
    
    def list_templates(self) -> List[str]:
        """List all available templates.
        
        Returns:
            List of template names
        """
        return self.template_manager.list_templates()
    
    def update_global_filter(self, filter_config: Dict[str, Any]) -> None:
        """Update global notification filter.
        
        Args:
            filter_config: Filter configuration dictionary
        """
        self.global_filter = NotificationFilter(**filter_config)
        logger.info("Global notification filter updated")
    
    async def shutdown(self) -> None:
        """Shutdown notification manager and cleanup resources."""
        logger.info("Shutting down notification manager")
        # Any cleanup needed for providers can be added here
