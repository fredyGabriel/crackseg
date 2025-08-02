"""Docker Compose generation for packaging system.

This module handles creation of Docker Compose configurations.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import DeploymentConfig

logger = logging.getLogger(__name__)


class DockerComposeGenerator:
    """Generates Docker Compose configurations."""

    def __init__(self) -> None:
        """Initialize Docker Compose generator."""
        self.logger = logging.getLogger(__name__)

    def create_docker_compose(
        self, package_dir: Path, config: "DeploymentConfig"
    ) -> str:
        """Create Docker Compose configuration.

        Args:
            package_dir: Package directory
            config: Deployment configuration

        Returns:
            Path to created Docker Compose file
        """
        compose_content = self._generate_compose_content(config)
        compose_path = package_dir / "docker-compose.yml"
        compose_path.write_text(compose_content)

        self.logger.info(f"Created Docker Compose: {compose_path}")
        return str(compose_path)

    def _generate_compose_content(self, config: "DeploymentConfig") -> str:
        """Generate Docker Compose content."""
        return f"""version: '3.8'

services:
  crackseg:
    build:
      context: .
      dockerfile: Dockerfile
    image: crackseg:latest
    container_name: crackseg-app
    ports:
      - "8501:8501"
    environment:
      - CRACKSEG_ENV={config.target_environment}
      - CRACKSEG_DEPLOYMENT_TYPE={config.deployment_type}
      - ENABLE_HEALTH_CHECKS={config.enable_health_checks}
      - ENABLE_METRICS_COLLECTION={config.enable_metrics_collection}
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    networks:
      - crackseg-network

  {self._get_monitoring_service(config)}:
    image: prom/prometheus:latest
    container_name: crackseg-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    restart: unless-stopped
    networks:
      - crackseg-network

  grafana:
    image: grafana/grafana:latest
    container_name: crackseg-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - >-
        ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources:ro
    restart: unless-stopped
    networks:
      - crackseg-network
    depends_on:
      - prometheus

volumes:
  prometheus_data:
  grafana_data:

networks:
  crackseg-network:
    driver: bridge
"""

    def _get_monitoring_service(self, config: "DeploymentConfig") -> str:
        """Get monitoring service name based on environment."""
        if config.enable_metrics_collection:
            return "prometheus"
        else:
            return "prometheus-disabled"

    def create_monitoring_configs(
        self, package_dir: Path, config: "DeploymentConfig"
    ) -> None:
        """Create monitoring configuration files.

        Args:
            package_dir: Package directory
            config: Deployment configuration
        """
        if not config.enable_metrics_collection:
            return

        monitoring_dir = package_dir / "monitoring"
        monitoring_dir.mkdir(exist_ok=True)

        # Create Prometheus configuration
        prometheus_config = self._generate_prometheus_config()
        prometheus_path = monitoring_dir / "prometheus.yml"
        prometheus_path.write_text(prometheus_config)

        # Create Grafana configurations
        grafana_dir = monitoring_dir / "grafana"
        grafana_dir.mkdir(exist_ok=True)

        # Create datasources directory
        datasources_dir = grafana_dir / "datasources"
        datasources_dir.mkdir(exist_ok=True)

        # Create dashboards directory
        dashboards_dir = grafana_dir / "dashboards"
        dashboards_dir.mkdir(exist_ok=True)

        # Create datasource configuration
        datasource_config = self._generate_datasource_config()
        datasource_path = datasources_dir / "datasource.yml"
        datasource_path.write_text(datasource_config)

        # Create dashboard configuration
        dashboard_config = self._generate_dashboard_config()
        dashboard_path = dashboards_dir / "dashboard.yml"
        dashboard_path.write_text(dashboard_config)

        # Create dashboard JSON
        dashboard_json = self._generate_dashboard_json()
        dashboard_json_path = dashboards_dir / "crackseg-dashboard.json"
        dashboard_json_path.write_text(dashboard_json)

        self.logger.info(f"Created monitoring configs in: {monitoring_dir}")

    def _generate_prometheus_config(self) -> str:
        """Generate Prometheus configuration."""
        return """global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'crackseg'
    static_configs:
      - targets: ['crackseg:8501']
    metrics_path: '/metrics'
    scrape_interval: 5s
"""

    def _generate_datasource_config(self) -> str:
        """Generate Grafana datasource configuration."""
        return """apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
"""

    def _generate_dashboard_config(self) -> str:
        """Generate Grafana dashboard configuration."""
        return """apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
"""

    def _generate_dashboard_json(self) -> str:
        """Generate Grafana dashboard JSON."""
        return """{
  "dashboard": {
    "id": null,
    "title": "CrackSeg Dashboard",
    "tags": ["crackseg", "machine-learning"],
    "style": "dark",
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "requests/sec"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 0
        }
      },
      {
        "id": 2,
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_errors_total[5m])",
            "legendFormat": "errors/sec"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 0
        }
      },
      {
        "id": 3,
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "process_resident_memory_bytes",
            "legendFormat": "memory"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 8
        }
      },
      {
        "id": 4,
        "title": "CPU Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(process_cpu_seconds_total[5m])",
            "legendFormat": "cpu"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 8
        }
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "5s"
  }
}"""
