"""Helm chart generation for packaging system.

This module handles creation of Helm charts for Kubernetes deployments.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import DeploymentConfig

logger = logging.getLogger(__name__)


class HelmChartGenerator:
    """Generates Helm charts for Kubernetes deployments."""

    def __init__(self) -> None:
        """Initialize Helm chart generator."""
        self.logger = logging.getLogger(__name__)

    def create_helm_chart(
        self, package_dir: Path, config: "DeploymentConfig"
    ) -> str:
        """Create Helm chart for deployment.

        Args:
            package_dir: Package directory
            config: Deployment configuration

        Returns:
            Path to created Helm chart
        """
        helm_dir = package_dir / "helm" / "crackseg"
        helm_dir.mkdir(parents=True, exist_ok=True)

        # Create Chart.yaml
        chart_yaml = self._generate_chart_yaml(config)
        chart_path = helm_dir / "Chart.yaml"
        chart_path.write_text(chart_yaml)

        # Create values.yaml
        values_yaml = self._generate_values_yaml(config)
        values_path = helm_dir / "values.yaml"
        values_path.write_text(values_yaml)

        # Create templates directory
        templates_dir = helm_dir / "templates"
        templates_dir.mkdir(exist_ok=True)

        # Create deployment template
        deployment_template = self._generate_deployment_template(config)
        deployment_path = templates_dir / "deployment.yaml"
        deployment_path.write_text(deployment_template)

        # Create service template
        service_template = self._generate_service_template(config)
        service_path = templates_dir / "service.yaml"
        service_path.write_text(service_template)

        # Create ingress template (for production)
        if config.target_environment == "production":
            ingress_template = self._generate_ingress_template(config)
            ingress_path = templates_dir / "ingress.yaml"
            ingress_path.write_text(ingress_template)

        # Create HPA template (for production)
        if config.target_environment == "production":
            hpa_template = self._generate_hpa_template(config)
            hpa_path = templates_dir / "hpa.yaml"
            hpa_path.write_text(hpa_template)

        # Create ConfigMap template
        configmap_template = self._generate_configmap_template(config)
        configmap_path = templates_dir / "configmap.yaml"
        configmap_path.write_text(configmap_template)

        # Create Secret template (for production)
        if config.target_environment == "production":
            secret_template = self._generate_secret_template(config)
            secret_path = templates_dir / "secret.yaml"
            secret_path.write_text(secret_template)

        # Create NOTES.txt
        notes_txt = self._generate_notes_txt(config)
        notes_path = templates_dir / "NOTES.txt"
        notes_path.write_text(notes_txt)

        self.logger.info(f"Created Helm chart in: {helm_dir}")
        return str(helm_dir)

    def _generate_chart_yaml(self, config: "DeploymentConfig") -> str:
        """Generate Chart.yaml content."""
        return """apiVersion: v2
name: crackseg
description: A Helm chart for CrackSeg crack detection system
type: application
version: 1.0.0
appVersion: "1.0.0"
keywords:
  - crackseg
  - machine-learning
  - computer-vision
home: https://github.com/crackseg/crackseg
sources:
  - https://github.com/crackseg/crackseg
maintainers:
  - name: CrackSeg Team
    email: team@crackseg.com
"""

    def _generate_values_yaml(self, config: "DeploymentConfig") -> str:
        """Generate values.yaml content."""
        return f"""# Default values for crackseg
# This is a YAML-formatted file.

replicaCount: {self._get_replica_count(config)}

image:
  repository: crackseg
  pullPolicy: IfNotPresent
  tag: "latest"

imagePullSecrets: []
nameOverride: ""
fullnameOverride: ""

serviceAccount:
  create: true
  annotations: {{}}
  name: ""

podAnnotations: {{}}

podSecurityContext: {{}}

securityContext: {{}}

service:
  type: ClusterIP
  port: 80

ingress:
  enabled: {str(config.target_environment == "production").lower()}
  className: ""
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
  hosts:
    - host: crackseg.example.com
      paths:
        - path: /
          pathType: Prefix
  tls: []

resources:
  limits:
    cpu: 1000m
    memory: 2Gi
  requests:
    cpu: 250m
    memory: 512Mi

autoscaling:
  enabled: {str(config.target_environment == "production").lower()}
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

nodeSelector: {{}}

tolerations: []

affinity: {{}}

env:
  CRACKSEG_ENV: "{config.target_environment}"
  CRACKSEG_DEPLOYMENT_TYPE: "{config.deployment_type}"
  ENABLE_HEALTH_CHECKS: "{config.enable_health_checks}"
  ENABLE_METRICS_COLLECTION: "{config.enable_metrics_collection}"
"""

    def _generate_deployment_template(self, config: "DeploymentConfig") -> str:
        """Generate deployment template."""
        return """apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "crackseg.fullname" . }}
  labels:
    {{ include "crackseg.labels" . }}
spec:
  {{ if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{ end }}
  selector:
    matchLabels:
      {{ include "crackseg.selectorLabels" . }}
  template:
    metadata:
      {{ with .Values.podAnnotations }}
      annotations:
        {{ toYaml . }}
      {{ end }}
      labels:
        {{ include "crackseg.selectorLabels" . }}
    spec:
      {{ with .Values.imagePullSecrets }}
      imagePullSecrets:
        {{ toYaml . }}
      {{ end }}
      serviceAccountName: {{ include "crackseg.serviceAccountName" . }}
      securityContext:
        {{ toYaml .Values.podSecurityContext | nindent 8 }}
      containers:
        - name: {{ .Chart.Name }}
          securityContext:
            {{ toYaml .Values.securityContext | nindent 12 }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"  # noqa: E501
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - name: http
              containerPort: 8501
              protocol: TCP
          livenessProbe:
            httpGet:
              path: /healthz
              port: http
          readinessProbe:
            httpGet:
              path: /healthz
              port: http
          resources:
            {{ toYaml .Values.resources | nindent 12 }}
          env:
            {{ toYaml .Values.env | nindent 12 }}
"""

    def _generate_service_template(self, config: "DeploymentConfig") -> str:
        """Generate service template."""
        return """apiVersion: v1
kind: Service
metadata:
  name: {{ include "crackseg.fullname" . }}
  labels:
    {{ include "crackseg.labels" . }}
spec:
  type: {{ .Values.service.type }}
  ports:
    - port: {{ .Values.service.port }}
      targetPort: http
      protocol: TCP
      name: http
  selector:
    {{ include "crackseg.selectorLabels" . }}
"""

    def _generate_ingress_template(self, config: "DeploymentConfig") -> str:
        """Generate ingress template."""
        return """{{- if .Values.ingress.enabled -}}
{{- $fullName := include "crackseg.fullname" . -}}
{{- $svcPort := .Values.service.port -}}
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {{ $fullName }}
  labels:
    {{ include "crackseg.labels" . }}
  {{- with .Values.ingress.annotations }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
spec:
  {{- if .Values.ingress.className }}
  ingressClassName: {{ .Values.ingress.className }}
  {{- end }}
  {{- if .Values.ingress.tls }}
  tls:
    {{- range .Values.ingress.tls }}
    - hosts:
        {{- range .hosts }}
        - {{ . | quote }}
        {{- end }}
      secretName: {{ .secretName }}
    {{- end }}
  {{- end }}
  rules:
    {{- range .Values.ingress.hosts }}
    - host: {{ .host | quote }}
      http:
        paths:
          {{- range .paths }}
          - path: {{ .path }}
            {{- if and .pathType (eq (lower .pathType) "implementationprefix") }}  # noqa: E501
            pathType: ImplementationSpecific
            {{- else if eq (lower .pathType) "prefix" }}
            pathType: Prefix
            {{- else if eq (lower .pathType) "exact" }}
            pathType: Exact
            {{- else }}
            pathType: Prefix
            {{- end }}
            backend:
              service:
                name: {{ $fullName }}
                port:
                  number: {{ $svcPort }}
    {{- end }}
{{- end }}
"""

    def _generate_hpa_template(self, config: "DeploymentConfig") -> str:
        """Generate HPA template."""
        return """{{- if .Values.autoscaling.enabled }}
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{ include "crackseg.fullname" . }}
  labels:
    {{ include "crackseg.labels" . }}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ include "crackseg.fullname" . }}
  minReplicas: {{ .Values.autoscaling.minReplicas }}
  maxReplicas: {{ .Values.autoscaling.maxReplicas }}
  metrics:
    {{- if .Values.autoscaling.targetCPUUtilizationPercentage }}
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: {{ .Values.autoscaling.targetCPUUtilizationPercentage }}
    {{- end }}
    {{- if .Values.autoscaling.targetMemoryUtilizationPercentage }}
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: {{ .Values.autoscaling.targetMemoryUtilizationPercentage }}
    {{- end }}
{{- end }}
"""

    def _generate_configmap_template(self, config: "DeploymentConfig") -> str:
        """Generate ConfigMap template."""
        return """apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ include "crackseg.fullname" . }}-config
  labels:
    {{ include "crackseg.labels" . }}
data:
  environment: {{ .Values.env.CRACKSEG_ENV | quote }}
  deployment_type: {{ .Values.env.CRACKSEG_DEPLOYMENT_TYPE | quote }}
  log_level: "INFO"
  enable_health_checks: {{ .Values.env.ENABLE_HEALTH_CHECKS | quote }}
  enable_metrics_collection: {{ .Values.env.ENABLE_METRICS_COLLECTION | quote }}
"""

    def _generate_secret_template(self, config: "DeploymentConfig") -> str:
        """Generate Secret template."""
        return """apiVersion: v1
kind: Secret
metadata:
  name: {{ include "crackseg.fullname" . }}-secret
  labels:
    {{ include "crackseg.labels" . }}
type: Opaque
data:
  # Base64 encoded values (replace with actual values)
  api_key: YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXo=  # abcdefghijklmnopqrstuvwxyz
  database_url: cG9zdGdyZXM6Ly9sb2NhbGhvc3Q6NTQzMi9jcmFja3NlZw==  # postgres://localhost:5432/crackseg
"""

    def _generate_notes_txt(self, config: "DeploymentConfig") -> str:
        """Generate NOTES.txt content."""
        return """1. Get the application URL by running these commands:
{- if .Values.ingress.enabled }
{- range $host := .Values.ingress.hosts }
  {- range .paths }
  http{ if $.Values.ingress.tls }s{ end }://{ $host.host }{ .path }
  {- end }
{- end }
{- else if contains "NodePort" .Values.service.type }
  export NODE_PORT=$(kubectl get --namespace { .Release.Namespace } -o jsonpath="{.Values.service.port}")
  export NODE_IP=$(kubectl get nodes --namespace { .Release.Namespace } -o jsonpath="{.Values.service.nodePort}")
  echo http://$NODE_IP:$NODE_PORT
{- else if contains "LoadBalancer" .Values.service.type }
     NOTE: It may take a few minutes for the LoadBalancer IP to be available.
           You can watch the status of by running 'kubectl get --namespace { .Release.Namespace } svc -w { include "crackseg.fullname" . }'
  export SERVICE_IP=$(kubectl get svc --namespace { .Release.Namespace } { include "crackseg.fullname" . } --template "{"{ range (index .status.loadBalancer.ingress 0) }{.}{ end }"}")
  echo http://$SERVICE_IP:{ .Values.service.port }
{- else if contains "ClusterIP" .Values.service.type }
  export POD_NAME=$(kubectl get pods --namespace { .Release.Namespace } -l "app.kubernetes.io/name={ include "crackseg.name" . },app.kubernetes.io/instance={ .Release.Name }" -o jsonpath="{.Values.service.port}")
  echo "Visit http://127.0.0.1:8080 to use your application"
  kubectl --namespace { .Release.Namespace } port-forward $POD_NAME 8080:{ .Values.service.port }
{- end }

2. Check the application health:
   kubectl get pods --namespace { .Release.Namespace } -l "app.kubernetes.io/name={ include "crackseg.name" . },app.kubernetes.io/instance={ .Release.Name }"

3. View logs:
   kubectl logs --namespace { .Release.Namespace } -l "app.kubernetes.io/name={ include "crackseg.name" . },app.kubernetes.io/instance={ .Release.Name }"
"""

    def _get_replica_count(self, config: "DeploymentConfig") -> int:
        """Get replica count based on environment."""
        if config.target_environment == "production":
            return 3
        elif config.target_environment == "staging":
            return 2
        else:
            return 1
