"""Kubernetes manifest generation for packaging system.

This module handles creation of Kubernetes deployment manifests.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import DeploymentConfig

logger = logging.getLogger(__name__)


class KubernetesManifestGenerator:
    """Generates Kubernetes deployment manifests."""

    def __init__(self) -> None:
        """Initialize Kubernetes manifest generator."""
        self.logger = logging.getLogger(__name__)

    def create_kubernetes_manifests(
        self, package_dir: Path, config: "DeploymentConfig"
    ) -> list[str]:
        """Create Kubernetes deployment manifests.

        Args:
            package_dir: Package directory
            config: Deployment configuration

        Returns:
            List of created manifest file paths
        """
        k8s_dir = package_dir / "k8s"
        k8s_dir.mkdir(exist_ok=True)

        manifests = []

        # Deployment manifest
        deployment_manifest = self._generate_deployment_manifest(config)
        deployment_path = k8s_dir / "deployment.yaml"
        deployment_path.write_text(deployment_manifest)
        manifests.append(str(deployment_path))

        # Service manifest
        service_manifest = self._generate_service_manifest(config)
        service_path = k8s_dir / "service.yaml"
        service_path.write_text(service_manifest)
        manifests.append(str(service_path))

        # Ingress manifest (for production)
        if config.target_environment == "production":
            ingress_manifest = self._generate_ingress_manifest(config)
            ingress_path = k8s_dir / "ingress.yaml"
            ingress_path.write_text(ingress_manifest)
            manifests.append(str(ingress_path))

        # Horizontal Pod Autoscaler
        if config.target_environment == "production":
            hpa_manifest = self._generate_hpa_manifest(config)
            hpa_path = k8s_dir / "hpa.yaml"
            hpa_path.write_text(hpa_manifest)
            manifests.append(str(hpa_path))

        # ConfigMap
        configmap_manifest = self._generate_configmap_manifest(config)
        configmap_path = k8s_dir / "configmap.yaml"
        configmap_path.write_text(configmap_manifest)
        manifests.append(str(configmap_path))

        # Secret (for production)
        if config.target_environment == "production":
            secret_manifest = self._generate_secret_manifest(config)
            secret_path = k8s_dir / "secret.yaml"
            secret_path.write_text(secret_manifest)
            manifests.append(str(secret_path))

        self.logger.info(f"Created {len(manifests)} Kubernetes manifests")
        return manifests

    def _generate_deployment_manifest(self, config: "DeploymentConfig") -> str:
        """Generate Kubernetes deployment manifest."""
        return f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: crackseg
  labels:
    app: crackseg
    version: v1.0.0
spec:
  replicas: {self._get_replica_count(config)}
  selector:
    matchLabels:
      app: crackseg
  template:
    metadata:
      labels:
        app: crackseg
    spec:
      containers:
      - name: crackseg
        image: crackseg:latest
        ports:
        - containerPort: 8501
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        env:
        - name: CRACKSEG_ENV
          value: "{config.target_environment}"
        - name: CRACKSEG_DEPLOYMENT_TYPE
          value: "{config.deployment_type}"
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8501
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /healthz
            port: 8501
          initialDelaySeconds: 5
          periodSeconds: 5
"""

    def _generate_service_manifest(self, config: "DeploymentConfig") -> str:
        """Generate Kubernetes service manifest."""
        return """apiVersion: v1
kind: Service
metadata:
  name: crackseg-service
  labels:
    app: crackseg
spec:
  selector:
    app: crackseg
  ports:
  - port: 80
    targetPort: 8501
    protocol: TCP
  type: ClusterIP
"""

    def _generate_ingress_manifest(self, config: "DeploymentConfig") -> str:
        """Generate Kubernetes ingress manifest."""
        return """apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: crackseg-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  rules:
  - host: crackseg.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: crackseg-service
            port:
              number: 80
"""

    def _generate_hpa_manifest(self, config: "DeploymentConfig") -> str:
        """Generate Horizontal Pod Autoscaler manifest."""
        return """apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: crackseg-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: crackseg
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
"""

    def _generate_configmap_manifest(self, config: "DeploymentConfig") -> str:
        """Generate ConfigMap manifest."""
        return f"""apiVersion: v1
kind: ConfigMap
metadata:
  name: crackseg-config
data:
  environment: "{config.target_environment}"
  deployment_type: "{config.deployment_type}"
  log_level: "INFO"
  enable_health_checks: "{config.enable_health_checks}"
  enable_metrics_collection: "{config.enable_metrics_collection}"
"""

    def _generate_secret_manifest(self, config: "DeploymentConfig") -> str:
        """Generate Secret manifest."""
        return """apiVersion: v1
kind: Secret
metadata:
  name: crackseg-secret
type: Opaque
data:
  # Base64 encoded values (replace with actual values)
  api_key: YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXo=  # abcdefghijklmnopqrstuvwxyz
  database_url: cG9zdGdyZXM6Ly9sb2NhbGhvc3Q6NTQzMi9jcmFja3NlZw==  # postgres://localhost:5432/crackseg
"""

    def _get_replica_count(self, config: "DeploymentConfig") -> int:
        """Get replica count based on environment."""
        if config.target_environment == "production":
            return 3
        elif config.target_environment == "staging":
            return 2
        else:
            return 1
