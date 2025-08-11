"""Helm chart template helpers for deployment packaging.

Extracted from packaging/helm.py to reduce size and centralize templates.
"""

from __future__ import annotations


def generate_chart_yaml() -> str:
    return (
        "apiVersion: v2\n"
        "name: crackseg\n"
        "description: A Helm chart for CrackSeg crack detection system\n"
        "type: application\n"
        "version: 1.0.0\n"
        'appVersion: "1.0.0"\n'
        "keywords:\n"
        "  - crackseg\n"
        "  - machine-learning\n"
        "  - computer-vision\n"
        "home: https://github.com/crackseg/crackseg\n"
        "sources:\n"
        "  - https://github.com/crackseg/crackseg\n"
        "maintainers:\n"
        "  - name: CrackSeg Team\n"
        "    email: team@crackseg.com\n"
    )


def generate_values_yaml(
    replica_count: int,
    is_production: bool,
    env: str,
    deployment_type: str,
    enable_health_checks: bool,
    enable_metrics_collection: bool,
) -> str:
    return (
        "# Default values for crackseg\n"
        "# This is a YAML-formatted file.\n\n"
        f"replicaCount: {replica_count}\n\n"
        "image:\n"
        "  repository: crackseg\n"
        "  pullPolicy: IfNotPresent\n"
        '  tag: "latest"\n\n'
        "imagePullSecrets: []\n"
        'nameOverride: ""\n'
        'fullnameOverride: ""\n\n'
        "serviceAccount:\n"
        "  create: true\n"
        "  annotations: {}\n"
        '  name: ""\n\n'
        "podAnnotations: {}\n\n"
        "podSecurityContext: {}\n\n"
        "securityContext: {}\n\n"
        "service:\n"
        "  type: ClusterIP\n"
        "  port: 80\n\n"
        "ingress:\n"
        f"  enabled: {str(is_production).lower()}\n"
        '  className: ""\n'
        "  annotations:\n"
        "    nginx.ingress.kubernetes.io/rewrite-target: /\n"
        "  hosts:\n"
        "    - host: crackseg.example.com\n"
        "      paths:\n"
        "        - path: /\n"
        "          pathType: Prefix\n"
        "  tls: []\n\n"
        "resources:\n"
        "  limits:\n"
        "    cpu: 1000m\n"
        "    memory: 2Gi\n"
        "  requests:\n"
        "    cpu: 250m\n"
        "    memory: 512Mi\n\n"
        "autoscaling:\n"
        f"  enabled: {str(is_production).lower()}\n"
        "  minReplicas: 2\n"
        "  maxReplicas: 10\n"
        "  targetCPUUtilizationPercentage: 70\n"
        "  targetMemoryUtilizationPercentage: 80\n\n"
        "nodeSelector: {}\n\n"
        "tolerations: []\n\n"
        "affinity: {}\n\n"
        "env:\n"
        f'  CRACKSEG_ENV: "{env}"\n'
        f'  CRACKSEG_DEPLOYMENT_TYPE: "{deployment_type}"\n'
        f'  ENABLE_HEALTH_CHECKS: "{str(enable_health_checks).lower()}"\n'
        f'  ENABLE_METRICS_COLLECTION: "{str(enable_metrics_collection).lower()}"\n'
    )


def generate_deployment_template() -> str:
    return (
        "apiVersion: apps/v1\n"
        "kind: Deployment\n"
        "metadata:\n"
        '  name: {{ include "crackseg.fullname" . }}\n'
        "  labels:\n"
        '    {{ include "crackseg.labels" . }}\n'
        "spec:\n"
        "  {{ if not .Values.autoscaling.enabled }}\n"
        "  replicas: {{ .Values.replicaCount }}\n"
        "  {{ end }}\n"
        "  selector:\n"
        "    matchLabels:\n"
        '      {{ include "crackseg.selectorLabels" . }}\n'
        "  template:\n"
        "    metadata:\n"
        "      {{ with .Values.podAnnotations }}\n"
        "      annotations:\n"
        "        {{ toYaml . }}\n"
        "      {{ end }}\n"
        "      labels:\n"
        '        {{ include "crackseg.selectorLabels" . }}\n'
        "    spec:\n"
        "      {{ with .Values.imagePullSecrets }}\n"
        "      imagePullSecrets:\n"
        "        {{ toYaml . }}\n"
        "      {{ end }}\n"
        '      serviceAccountName: {{ include "crackseg.serviceAccountName" . }}\n'
        "      securityContext:\n"
        "        {{ toYaml .Values.podSecurityContext | nindent 8 }}\n"
        "      containers:\n"
        "        - name: {{ .Chart.Name }}\n"
        "          securityContext:\n"
        "            {{ toYaml .Values.securityContext | nindent 12 }}\n"
        '          image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"\n'
        "          imagePullPolicy: {{ .Values.image.pullPolicy }}\n"
        "          ports:\n"
        "            - name: http\n"
        "              containerPort: 8501\n"
        "              protocol: TCP\n"
        "          livenessProbe:\n"
        "            httpGet:\n"
        "              path: /healthz\n"
        "              port: http\n"
        "          readinessProbe:\n"
        "            httpGet:\n"
        "              path: /healthz\n"
        "              port: http\n"
        "          resources:\n"
        "            {{ toYaml .Values.resources | nindent 12 }}\n"
        "          env:\n"
        "            {{ toYaml .Values.env | nindent 12 }}\n"
    )


def generate_service_template() -> str:
    return (
        "apiVersion: v1\n"
        "kind: Service\n"
        "metadata:\n"
        '  name: {{ include "crackseg.fullname" . }}\n'
        "  labels:\n"
        '    {{ include "crackseg.labels" . }}\n'
        "spec:\n"
        "  type: {{ .Values.service.type }}\n"
        "  ports:\n"
        "    - port: {{ .Values.service.port }}\n"
        "      targetPort: http\n"
        "      protocol: TCP\n"
        "      name: http\n"
        "  selector:\n"
        '    {{ include "crackseg.selectorLabels" . }}\n'
    )


def generate_ingress_template() -> str:
    return (
        "{{- if .Values.ingress.enabled -}}\n"
        '{{- $fullName := include "crackseg.fullname" . -}}\n'
        "{{- $svcPort := .Values.service.port -}}\n"
        "apiVersion: networking.k8s.io/v1\n"
        "kind: Ingress\n"
        "metadata:\n"
        "  name: {{ $fullName }}\n"
        "  labels:\n"
        '    {{ include "crackseg.labels" . }}\n'
        "  {{- with .Values.ingress.annotations }}\n"
        "  annotations:\n"
        "    {{- toYaml . | nindent 4 }}\n"
        "  {{- end }}\n"
        "spec:\n"
        "  {{- if .Values.ingress.className }}\n"
        "  ingressClassName: {{ .Values.ingress.className }}\n"
        "  {{- end }}\n"
        "  {{- if .Values.ingress.tls }}\n"
        "  tls:\n"
        "    {{- range .Values.ingress.tls }}\n"
        "    - hosts:\n"
        "        {{- range .hosts }}\n"
        "        - {{ . | quote }}\n"
        "        {{- end }}\n"
        "      secretName: {{ .secretName }}\n"
        "    {{- end }}\n"
        "  {{- end }}\n"
        "  rules:\n"
        "    {{- range .Values.ingress.hosts }}\n"
        "    - host: {{ .host | quote }}\n"
        "      http:\n"
        "        paths:\n"
        "          {{- range .paths }}\n"
        "          - path: {{ .path }}\n"
        '            {{- if and .pathType (eq (lower .pathType) "implementationprefix") }}\n'
        "            pathType: ImplementationSpecific\n"
        '            {{- else if eq (lower .pathType) "prefix" }}\n'
        "            pathType: Prefix\n"
        '            {{- else if eq (lower .pathType) "exact" }}\n'
        "            pathType: Exact\n"
        "            {{- else }}\n"
        "            pathType: Prefix\n"
        "            {{- end }}\n"
        "            backend:\n"
        "              service:\n"
        "                name: {{ $fullName }}\n"
        "                port:\n"
        "                  number: {{ $svcPort }}\n"
        "    {{- end }}\n"
        "{{- end }}\n"
    )


def generate_hpa_template() -> str:
    return (
        "{{- if .Values.autoscaling.enabled }}\n"
        "apiVersion: autoscaling/v2\n"
        "kind: HorizontalPodAutoscaler\n"
        "metadata:\n"
        '  name: {{ include "crackseg.fullname" . }}\n'
        "  labels:\n"
        '    {{ include "crackseg.labels" . }}\n'
        "spec:\n"
        "  scaleTargetRef:\n"
        "    apiVersion: apps/v1\n"
        "    kind: Deployment\n"
        '    name: {{ include "crackseg.fullname" . }}\n'
        "  minReplicas: {{ .Values.autoscaling.minReplicas }}\n"
        "  maxReplicas: {{ .Values.autoscaling.maxReplicas }}\n"
        "  metrics:\n"
        "    {{- if .Values.autoscaling.targetCPUUtilizationPercentage }}\n"
        "    - type: Resource\n"
        "      resource:\n"
        "        name: cpu\n"
        "        target:\n"
        "          type: Utilization\n"
        "          averageUtilization: {{ .Values.autoscaling.targetCPUUtilizationPercentage }}\n"
        "    {{- end }}\n"
        "    {{- if .Values.autoscaling.targetMemoryUtilizationPercentage }}\n"
        "    - type: Resource\n"
        "      resource:\n"
        "        name: memory\n"
        "        target:\n"
        "          type: Utilization\n"
        "          averageUtilization: {{ .Values.autoscaling.targetMemoryUtilizationPercentage }}\n"
        "    {{- end }}\n"
        "{{- end }}\n"
    )


def generate_configmap_template() -> str:
    return (
        "apiVersion: v1\n"
        "kind: ConfigMap\n"
        "metadata:\n"
        '  name: {{ include "crackseg.fullname" . }}-config\n'
        "  labels:\n"
        '    {{ include "crackseg.labels" . }}\n'
        "data:\n"
        "  environment: {{ .Values.env.CRACKSEG_ENV | quote }}\n"
        "  deployment_type: {{ .Values.env.CRACKSEG_DEPLOYMENT_TYPE | quote }}\n"
        '  log_level: "INFO"\n'
        "  enable_health_checks: {{ .Values.env.ENABLE_HEALTH_CHECKS | quote }}\n"
        "  enable_metrics_collection: {{ .Values.env.ENABLE_METRICS_COLLECTION | quote }}\n"
    )


def generate_secret_template() -> str:
    return (
        "apiVersion: v1\n"
        "kind: Secret\n"
        "metadata:\n"
        '  name: {{ include "crackseg.fullname" . }}-secret\n'
        "  labels:\n"
        '    {{ include "crackseg.labels" . }}\n'
        "type: Opaque\n"
        "data:\n"
        "  # Base64 encoded values (replace with actual values)\n"
        "  api_key: YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXo=\n"
        "  database_url: cG9zdGdyZXM6Ly9sb2NhbGhvc3Q6NTQzMi9jcmFja3NlZw==\n"
    )


def generate_notes_txt() -> str:
    return (
        "1. Get the application URL by running these commands:\n"
        "{- if .Values.ingress.enabled }\n"
        "{- range $host := .Values.ingress.hosts }\n"
        "  {- range .paths }\n"
        "  http{ if $.Values.ingress.tls }s{ end }://{ $host.host }{ .path }\n"
        "  {- end }\n"
        "{- end }\n"
        '{- else if contains "NodePort" .Values.service.type }\n'
        '  export NODE_PORT=$(kubectl get --namespace { .Release.Namespace } -o jsonpath="{.Values.service.port}")\n'
        '  export NODE_IP=$(kubectl get nodes --namespace { .Release.Namespace } -o jsonpath="{.Values.service.nodePort}")\n'
        "  echo http://$NODE_IP:$NODE_PORT\n"
        '{- else if contains "LoadBalancer" .Values.service.type }\n'
        "     NOTE: It may take a few minutes for the LoadBalancer IP to be available.\n"
        "           You can watch the status of by running 'kubectl get --namespace { .Release.Namespace } svc -w { include \"crackseg.fullname\" . }'\n"
        '  export SERVICE_IP=$(kubectl get svc --namespace { .Release.Namespace } { include "crackseg.fullname" . } --template "{""{ range (index .status.loadBalancer.ingress 0) }{.}{ end }""}")\n'
        "  echo http://$SERVICE_IP:{ .Values.service.port }\n"
        '{- else if contains "ClusterIP" .Values.service.type }\n'
        '  export POD_NAME=$(kubectl get pods --namespace { .Release.Namespace } -l "app.kubernetes.io/name={ include "crackseg.name" . },app.kubernetes.io/instance={ .Release.Name }" -o jsonpath="{.Values.service.port}")\n'
        '  echo "Visit http://127.0.0.1:8080 to use your application"\n'
        "  kubectl --namespace { .Release.Namespace } port-forward $POD_NAME 8080:{ .Values.service.port }\n"
        "{- end }\n\n"
        "2. Check the application health:\n"
        '   kubectl get pods --namespace { .Release.Namespace } -l "app.kubernetes.io/name={ include "crackseg.name" . },app.kubernetes.io/instance={ .Release.Name }"\n\n'
        "3. View logs:\n"
        '   kubectl logs --namespace { .Release.Namespace } -l "app.kubernetes.io/name={ include "crackseg.name" . },app.kubernetes.io/instance={ .Release.Name }"\n'
    )
