"""Alert handlers for deployment orchestration.

This module provides alert handling capabilities for deployment orchestration,
including base alert handler interface and concrete implementations for
logging, email, and Slack notifications.
"""

import json
import logging
import smtplib
import time
from abc import ABC, abstractmethod
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import requests

from .orchestration import DeploymentMetadata


class AlertHandler(ABC):
    """Base class for alert handlers."""

    @abstractmethod
    def send_alert(
        self, alert_type: str, metadata: DeploymentMetadata, **kwargs
    ) -> None:
        """Send alert.

        Args:
            alert_type: Type of alert
            metadata: Deployment metadata
            **kwargs: Additional alert data
        """
        pass


class LoggingAlertHandler(AlertHandler):
    """Alert handler that logs alerts."""

    def __init__(self) -> None:
        """Initialize logging alert handler."""
        self.logger = logging.getLogger(__name__)

    def send_alert(
        self, alert_type: str, metadata: DeploymentMetadata, **kwargs
    ) -> None:
        """Send alert via logging.

        Args:
            alert_type: Type of alert
            metadata: Deployment metadata
            **kwargs: Additional alert data
        """
        if alert_type == "deployment_success":
            duration = (metadata.end_time or time.time()) - metadata.start_time
            self.logger.info(
                f"✅ Deployment {metadata.deployment_id} completed "
                f"successfully in {duration:.2f}s"
            )
        elif alert_type == "deployment_failure":
            error_msg = kwargs.get("error", "Unknown error")
            self.logger.error(
                f"❌ Deployment {metadata.deployment_id} failed: {error_msg}"
            )
        elif alert_type == "rollback_triggered":
            self.logger.warning(
                f"🔄 Rollback triggered for deployment "
                f"{metadata.deployment_id}: {metadata.rollback_reason}"
            )


class EmailAlertHandler(AlertHandler):
    """Alert handler that sends email alerts."""

    def __init__(
        self, smtp_server: str, smtp_port: int, username: str, password: str
    ) -> None:
        """Initialize email alert handler.

        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            username: Email username
            password: Email password
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.logger = logging.getLogger(__name__)

    def send_alert(
        self, alert_type: str, metadata: DeploymentMetadata, **kwargs
    ) -> None:
        """Send alert via email.

        Args:
            alert_type: Type of alert
            metadata: Deployment metadata
            **kwargs: Additional alert data
        """
        try:
            # Create email message
            msg = MIMEMultipart()
            msg["From"] = self.username
            msg["To"] = "admin@crackseg.com"  # Configure recipient
            msg["Subject"] = (
                f"Deployment Alert: {alert_type.replace('_', ' ').title()}"
            )

            # Create email body
            duration = (metadata.end_time or time.time()) - metadata.start_time
            body = f"""
Deployment Alert: {alert_type.replace("_", " ").title()}

Deployment ID: {metadata.deployment_id}
Artifact ID: {metadata.artifact_id}
Strategy: {metadata.strategy.value}
Status: {metadata.state.value}
Duration: {duration:.2f}s

"""

            if alert_type == "deployment_failure":
                error_msg = kwargs.get("error", "Unknown error")
                body += f"Error: {error_msg}\n"
            elif alert_type == "rollback_triggered":
                body += f"Rollback Reason: {metadata.rollback_reason}\n"

            msg.attach(MIMEText(body, "plain"))

            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()

            self.logger.info(f"Email alert sent for {alert_type}")

        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")


class SlackAlertHandler(AlertHandler):
    """Alert handler that sends Slack alerts."""

    def __init__(self, webhook_url: str) -> None:
        """Initialize Slack alert handler.

        Args:
            webhook_url: Slack webhook URL
        """
        self.webhook_url = webhook_url
        self.logger = logging.getLogger(__name__)

    def send_alert(
        self, alert_type: str, metadata: DeploymentMetadata, **kwargs
    ) -> None:
        """Send alert via Slack.

        Args:
            alert_type: Type of alert
            metadata: Deployment metadata
            **kwargs: Additional alert data
        """
        try:
            # Create Slack message
            duration = (metadata.end_time or time.time()) - metadata.start_time
            if alert_type == "deployment_success":
                color = "good"
                emoji = "✅"
                title = "Deployment Successful"
            elif alert_type == "deployment_failure":
                color = "danger"
                emoji = "❌"
                title = "Deployment Failed"
            elif alert_type == "rollback_triggered":
                color = "warning"
                emoji = "🔄"
                title = "Rollback Triggered"
            else:
                color = "good"
                emoji = "ℹ️"
                title = "Deployment Alert"

            message = {
                "attachments": [
                    {
                        "color": color,
                        "title": f"{emoji} {title}",
                        "fields": [
                            {
                                "title": "Deployment ID",
                                "value": metadata.deployment_id,
                                "short": True,
                            },
                            {
                                "title": "Artifact ID",
                                "value": metadata.artifact_id,
                                "short": True,
                            },
                            {
                                "title": "Strategy",
                                "value": metadata.strategy.value,
                                "short": True,
                            },
                            {
                                "title": "Status",
                                "value": metadata.state.value,
                                "short": True,
                            },
                            {
                                "title": "Duration",
                                "value": f"{duration:.2f}s",
                                "short": True,
                            },
                        ],
                    }
                ]
            }

            if alert_type == "deployment_failure":
                error_msg = kwargs.get("error", "Unknown error")
                message["attachments"][0]["fields"].append(
                    {"title": "Error", "value": error_msg, "short": False}
                )

            # Send to Slack
            response = requests.post(
                self.webhook_url,
                data=json.dumps(message),
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                self.logger.info(f"Slack alert sent for {alert_type}")
            else:
                self.logger.warning(
                    f"Failed to send Slack alert: {response.status_code}"
                )

        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {e}")
