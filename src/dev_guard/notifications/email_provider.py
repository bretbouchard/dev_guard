"""Email notification provider."""

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from .base import NotificationLevel, NotificationMessage, NotificationProvider, NotificationResult

logger = logging.getLogger(__name__)


class EmailProvider(NotificationProvider):
    """Email notification provider using SMTP."""
    
    def __init__(
        self,
        smtp_server: str,
        smtp_port: int = 587,
        username: str | None = None,
        password: str | None = None,
        use_tls: bool = True,
        from_address: str | None = None,
        to_addresses: list[str] | None = None,
        supported_levels: list[NotificationLevel] | None = None,
        enabled: bool = True
    ):
        """Initialize email provider.
        
        Args:
            smtp_server: SMTP server hostname
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            use_tls: Whether to use TLS
            from_address: Sender email address
            to_addresses: List of recipient email addresses
            supported_levels: Notification levels to handle
            enabled: Whether provider is enabled
        """
        super().__init__("email", enabled)
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.use_tls = use_tls
        self.from_address = from_address or username
        self.to_addresses = to_addresses or []
        self.supported_levels = supported_levels or [
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
        """Send notification via email.
        
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
        
        if not self.to_addresses:
            return NotificationResult(
                success=False,
                provider=self.name,
                error="No recipient addresses configured"
            )
        
        try:
            # Create email message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[DevGuard] {message.title}"
            msg['From'] = self.from_address
            msg['To'] = ', '.join(self.to_addresses)
            
            # Create plain text content
            text_content = self._create_text_content(message)
            text_part = MIMEText(text_content, 'plain')
            msg.attach(text_part)
            
            # Create HTML content
            html_content = self._create_html_content(message)
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                
                if self.username and self.password:
                    server.login(self.username, self.password)
                
                server.send_message(msg)
            
            logger.debug(f"Email notification sent: {message.title}")
            return NotificationResult(
                success=True,
                provider=self.name,
                message_id=msg['Message-ID']
            )
        
        except Exception as e:
            logger.error(f"Email notification error: {e}")
            return NotificationResult(
                success=False,
                provider=self.name,
                error=str(e)
            )
    
    def _create_text_content(self, message: NotificationMessage) -> str:
        """Create plain text email content.
        
        Args:
            message: Notification message
            
        Returns:
            Plain text email content
        """
        lines = [
            "DevGuard Notification",
            "===================",
            "",
            f"Title: {message.title}",
            f"Level: {message.level.value}",
            f"Source: {message.source}",
            f"Time: {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Content:",
            "--------",
            message.content,
        ]
        
        if message.tags:
            lines.extend([
                "",
                f"Tags: {', '.join(message.tags)}"
            ])
        
        if message.metadata:
            lines.extend([
                "",
                "Additional Information:",
                "----------------------"
            ])
            for key, value in message.metadata.items():
                lines.append(f"{key.replace('_', ' ').title()}: {value}")
        
        if message.template_name:
            lines.extend([
                "",
                f"Template: {message.template_name}"
            ])
        
        return '\n'.join(lines)
    
    def _create_html_content(self, message: NotificationMessage) -> str:
        """Create HTML email content.
        
        Args:
            message: Notification message
            
        Returns:
            HTML email content
        """
        # Level colors
        level_colors = {
            NotificationLevel.DEBUG: '#808080',
            NotificationLevel.INFO: '#00BFFF',
            NotificationLevel.WARNING: '#FFA500',
            NotificationLevel.ERROR: '#FF6B6B',
            NotificationLevel.CRITICAL: '#8B0000'
        }
        
        level_color = level_colors.get(message.level, '#00BFFF')
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ 
                    background-color: {level_color}; 
                    color: white; 
                    padding: 15px; 
                    border-radius: 5px; 
                }}
                .content {{ 
                    background-color: #f9f9f9; 
                    padding: 15px; 
                    margin: 10px 0; 
                    border-left: 4px solid {level_color}; 
                }}
                .metadata {{ 
                    background-color: #e9e9e9; 
                    padding: 10px; 
                    margin: 10px 0; 
                    border-radius: 3px; 
                }}
                .tags {{ display: inline-block; background-color: {level_color}; 
                        color: white; padding: 3px 8px; margin: 2px; 
                        border-radius: 15px; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>DevGuard Notification</h2>
                <h3>{message.title}</h3>
            </div>
            
            <div class="metadata">
                <strong>Level:</strong> {message.level.value}<br>
                <strong>Source:</strong> {message.source}<br>
                <strong>Time:</strong> {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
            </div>
            
            <div class="content">
                <h4>Message:</h4>
                <p>{message.content.replace(chr(10), '<br>')}</p>
            </div>
        """
        
        if message.tags:
            html += f"""
            <div class="metadata">
                <strong>Tags:</strong><br>
                {' '.join([f'<span class="tags">{tag}</span>' for tag in message.tags])}
            </div>
            """
        
        if message.metadata:
            html += """
            <div class="metadata">
                <strong>Additional Information:</strong><br>
            """
            for key, value in message.metadata.items():
                html += f"<strong>{key.replace('_', ' ').title()}:</strong> {value}<br>"
            html += "</div>"
        
        if message.template_name:
            html += f"""
            <div class="metadata">
                <strong>Template:</strong> {message.template_name}
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html
