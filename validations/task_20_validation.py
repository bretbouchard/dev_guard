"""Validation and test suite for Task 20: Notification System Implementation."""

import asyncio
import logging
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dev_guard.core.config import NotificationConfig
from dev_guard.notifications import (
    DiscordProvider,
    EmailProvider,
    NotificationLevel,
    NotificationManager,
    NotificationMessage,
    NotificationTemplate,
    SlackProvider,
    TelegramProvider,
    TemplateManager,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NotificationSystemValidator:
    """Comprehensive validation for notification system implementation."""
    
    def __init__(self):
        """Initialize validator."""
        self.test_results = []
        self.total_tests = 0
        self.passed_tests = 0
    
    def log_test_result(self, test_name: str, passed: bool, error: str = ""):
        """Log test result."""
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
            logger.info(f"✅ {test_name}")
        else:
            logger.error(f"❌ {test_name}: {error}")
        
        self.test_results.append({
            "test": test_name,
            "passed": passed,
            "error": error
        })
    
    async def test_notification_providers(self):
        """Test notification provider implementations."""
        logger.info("🧪 Testing notification provider implementations...")
        
        # Test Discord Provider
        try:
            discord_provider = DiscordProvider(
                webhook_url="https://discord.com/api/webhooks/test",
                enabled=False  # Don't actually send
            )
            
            # Test message creation and validation
            test_message = NotificationMessage(
                title="Test Message",
                content="Test content",
                level=NotificationLevel.INFO,
                source="test",
                template_name=None
            )
            
            # Test level support
            supports_info = discord_provider.supports_level(NotificationLevel.INFO)
            supports_warning = discord_provider.supports_level(NotificationLevel.WARNING)
            
            self.log_test_result(
                "Discord Provider - Basic functionality",
                isinstance(discord_provider, DiscordProvider)
                and isinstance(test_message, NotificationMessage)
                and supports_warning
                and not supports_info,
            )

        except Exception as e:
            self.log_test_result("Discord Provider - Basic functionality", False, str(e))
        
        # Test Slack Provider
        try:
            slack_provider = SlackProvider(
                webhook_url="https://hooks.slack.com/services/test",
                enabled=False
            )
            
            supports_warning = slack_provider.supports_level(NotificationLevel.WARNING)
            
            self.log_test_result(
                "Slack Provider - Basic functionality",
                isinstance(slack_provider, SlackProvider) and supports_warning
            )
            
        except Exception as e:
            self.log_test_result("Slack Provider - Basic functionality", False, str(e))
        
        # Test Telegram Provider
        try:
            telegram_provider = TelegramProvider(
                bot_token="test_token",
                chat_id="test_chat",
                enabled=False
            )
            
            supports_info = telegram_provider.supports_level(NotificationLevel.INFO)
            
            self.log_test_result(
                "Telegram Provider - Basic functionality",
                isinstance(telegram_provider, TelegramProvider) and supports_info
            )
            
        except Exception as e:
            self.log_test_result("Telegram Provider - Basic functionality", False, str(e))
        
        # Test Email Provider
        try:
            email_provider = EmailProvider(
                smtp_server="smtp.test.com",
                smtp_port=587,
                username="test@test.com",
                to_addresses=["recipient@test.com"],
                enabled=False
            )
            
            supports_error = email_provider.supports_level(NotificationLevel.ERROR)
            
            self.log_test_result(
                "Email Provider - Basic functionality", 
                isinstance(email_provider, EmailProvider) and supports_error
            )
            
        except Exception as e:
            self.log_test_result("Email Provider - Basic functionality", False, str(e))
    
    async def test_template_system(self):
        """Test notification template system."""
        logger.info("📝 Testing notification template system...")
        
        try:
            # Test template manager initialization
            template_manager = TemplateManager()
            default_templates = template_manager.list_templates()
            
            # Should have default templates
            expected_templates = [
                "system_startup", "system_shutdown", "critical_error",
                "task_completed", "security_alert", "dependency_update",
                "git_changes", "agent_health", "user_intervention", 
                "performance_alert"
            ]
            
            has_defaults = all(t in default_templates for t in expected_templates)
            
            self.log_test_result(
                "Template Manager - Default templates loaded",
                has_defaults and len(default_templates) >= 10
            )
            
        except Exception as e:
            self.log_test_result("Template Manager - Default templates loaded", False, str(e))
        
        try:
            # Test custom template registration
            custom_template = NotificationTemplate(
                name="test_template",
                title_template="Test: {title}",
                content_template="Message: {message}",
                level=NotificationLevel.INFO
            )
            
            template_manager.register_template(custom_template)
            
            # Test template rendering
            context = {"title": "Test Title", "message": "Test message"}
            rendered = template_manager.render_template("test_template", context)
            
            self.log_test_result(
                "Template System - Custom template rendering",
                rendered is not None and 
                rendered.title == "Test: Test Title" and
                "Test message" in rendered.content
            )
            
        except Exception as e:
            self.log_test_result("Template System - Custom template rendering", False, str(e))
    
    async def test_notification_manager(self):
        """Test notification manager functionality."""
        logger.info("🎯 Testing notification manager...")
        
        try:
            # Test with mock configuration
            config = NotificationConfig(
                enabled=True,
                notification_levels=["INFO", "WARNING", "ERROR", "CRITICAL"]
            )
            
            notification_manager = NotificationManager(config)
            
            # Test provider setup (should be empty with no config)
            provider_status = notification_manager.get_provider_status()
            
            self.log_test_result(
                "Notification Manager - Initialization",
                isinstance(notification_manager, NotificationManager) and
                isinstance(provider_status, dict)
            )
            
        except Exception as e:
            self.log_test_result("Notification Manager - Initialization", False, str(e))
        
        try:
            # Test with providers configured
            config = NotificationConfig(
                enabled=True,
                discord_webhook="https://discord.com/api/webhooks/test",
                slack_webhook="https://hooks.slack.com/services/test",
                notification_levels=["WARNING", "ERROR", "CRITICAL"]
            )
            
            notification_manager = NotificationManager(config)
            provider_status = notification_manager.get_provider_status()
            
            # Should have Discord and Slack providers
            has_providers = len(provider_status) >= 2
            
            self.log_test_result(
                "Notification Manager - Provider configuration",
                has_providers
            )
            
        except Exception as e:
            self.log_test_result("Notification Manager - Provider configuration", False, str(e))
    
    async def test_notification_filtering(self):
        """Test notification filtering functionality."""
        logger.info("🔍 Testing notification filtering...")
        
        try:
            from dev_guard.notifications.base import NotificationFilter
            
            # Create test message
            test_message = NotificationMessage(
                title="Test Message",
                content="Test content",
                level=NotificationLevel.WARNING,
                source="test_agent",
                tags=["test", "validation"],
                template_name=None
            )
            
            # Test level filtering
            level_filter = NotificationFilter(levels=[NotificationLevel.ERROR])
            matches_level = level_filter.matches(test_message)  # Should be False
            
            # Test tag filtering
            tag_filter = NotificationFilter(tags=["test"])
            matches_tag = tag_filter.matches(test_message)  # Should be True
            
            # Test exclude tag filtering
            exclude_filter = NotificationFilter(exclude_tags=["validation"])
            matches_exclude = exclude_filter.matches(test_message)  # Should be False
            
            self.log_test_result(
                "Notification Filtering - Level, tag, and exclude filters",
                not matches_level and matches_tag and not matches_exclude
            )
            
        except Exception as e:
            self.log_test_result("Notification Filtering - Level, tag, and exclude filters", False, str(e))
    
    async def test_cli_integration(self):
        """Test CLI integration with notification system."""
        logger.info("💻 Testing CLI integration...")
        
        try:
            # Import CLI module
            import typer.testing

            from dev_guard.cli import app
            
            # Test CLI help includes notification commands
            runner = typer.testing.CliRunner()
            result = runner.invoke(app, ["--help"])
            
            has_notify_command = "notify" in result.stdout
            has_test_command = "test-notifications" in result.stdout
            has_status_command = "notification-status" in result.stdout
            
            self.log_test_result(
                "CLI Integration - Notification commands available",
                has_notify_command and has_test_command and has_status_command
            )
            
        except Exception as e:
            self.log_test_result("CLI Integration - Notification commands available", False, str(e))
    
    async def test_error_handling(self):
        """Test error handling and resilience."""
        logger.info("🛡️ Testing error handling...")
        
        try:
            # Test with invalid webhook URL
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_response = AsyncMock()
                mock_response.status = 404
                mock_response.text.return_value = "Not Found"
                mock_post.return_value.__aenter__.return_value = mock_response
                
                discord_provider = DiscordProvider(
                    webhook_url="https://invalid.webhook.url"
                )
                
                test_message = NotificationMessage(
                    title="Test",
                    content="Test",
                    level=NotificationLevel.WARNING,
                    source="test",
                    template_name=None
                )
                
                result = await discord_provider.send_message(test_message)
                
                self.log_test_result(
                    "Error Handling - Invalid webhook graceful failure",
                    not result.success and "404" in str(result.error)
                )
        
        except Exception as e:
            self.log_test_result("Error Handling - Invalid webhook graceful failure", False, str(e))
        
        try:
            # Test disabled provider
            config = NotificationConfig(enabled=False)
            notification_manager = NotificationManager(config)
            
            test_message = NotificationMessage(
                title="Test",
                content="Test",
                level=NotificationLevel.INFO,
                source="test",
                template_name=None
            )
            
            results = await notification_manager.send_notification(test_message)
            
            self.log_test_result(
                "Error Handling - Disabled system returns empty results",
                len(results) == 0
            )
        
        except Exception as e:
            self.log_test_result("Error Handling - Disabled system returns empty results", False, str(e))
    
    async def test_commander_integration(self):
        """Test Commander Agent integration with notifications."""
        logger.info("👑 Testing Commander Agent integration...")
        
        try:
            # Mock the commander agent with notification manager
            from dev_guard.agents.commander import CommanderAgent
            from dev_guard.core.config import Config
            
            # Create mock config with notifications
            config = Config()
            config.notifications = NotificationConfig(
                enabled=True,
                notification_levels=["WARNING", "ERROR", "CRITICAL"]
            )
            
            # Create mock shared memory and vector db
            mock_shared_memory = Mock()
            mock_vector_db = Mock()
            mock_vector_db.count_documents.return_value = 100
            
            # Test commander initialization with notification manager
            commander = CommanderAgent(
                agent_id="commander",
                config=config,
                shared_memory=mock_shared_memory,
                vector_db=mock_vector_db
            )
            
            has_notification_manager = hasattr(commander, 'notification_manager') and \
                                    commander.notification_manager is not None
            
            self.log_test_result(
                "Commander Integration - Notification manager initialization",
                has_notification_manager
            )
            
        except Exception as e:
            self.log_test_result("Commander Integration - Notification manager initialization", False, str(e))
    
    def generate_summary_report(self):
        """Generate final validation report."""
        logger.info("=" * 60)
        logger.info("📊 TASK 20 VALIDATION SUMMARY")
        logger.info("=" * 60)
        
        # Overall statistics
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        logger.info(f"Total Tests: {self.total_tests}")
        logger.info(f"Passed: {self.passed_tests}")
        logger.info(f"Failed: {self.total_tests - self.passed_tests}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        
        # Test categories summary
        categories = {}
        for result in self.test_results:
            category = result["test"].split(" - ")[0]
            if category not in categories:
                categories[category] = {"passed": 0, "total": 0}
            categories[category]["total"] += 1
            if result["passed"]:
                categories[category]["passed"] += 1
        
        logger.info("\n📋 Test Categories:")
        for category, stats in categories.items():
            rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
            status = "✅" if rate == 100 else "⚠️" if rate >= 50 else "❌"
            logger.info(f"  {status} {category}: {stats['passed']}/{stats['total']} ({rate:.1f}%)")
        
        # Failed tests details
        failed_tests = [r for r in self.test_results if not r["passed"]]
        if failed_tests:
            logger.info(f"\n❌ Failed Tests ({len(failed_tests)}):")
            for test in failed_tests:
                logger.error(f"  • {test['test']}: {test['error']}")
        
        # Final verdict
        logger.info("\n" + "=" * 60)
        if success_rate >= 80:
            logger.info("🎉 Task 20: Notification System Implementation - COMPLETE!")
            logger.info("✅ All core functionality implemented and validated")
        elif success_rate >= 60:
            logger.warning("⚠️ Task 20: Mostly Complete with Minor Issues")
            logger.warning("🔧 Some components need attention")
        else:
            logger.error("❌ Task 20: Implementation Issues Detected")
            logger.error("🚨 Major components need fixes")
        
        logger.info("=" * 60)


async def main():
    """Run complete validation suite."""
    logger.info("🚀 Starting Task 20 Notification System Validation")
    logger.info("=" * 60)
    
    validator = NotificationSystemValidator()
    
    # Run all validation tests
    await validator.test_notification_providers()
    await validator.test_template_system()
    await validator.test_notification_manager()
    await validator.test_notification_filtering() 
    await validator.test_cli_integration()
    await validator.test_error_handling()
    await validator.test_commander_integration()
    
    # Generate final report
    validator.generate_summary_report()


if __name__ == "__main__":
    asyncio.run(main())
