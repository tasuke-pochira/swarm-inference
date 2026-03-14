# Swarm Inference Protocol - Deployment Guide

## Overview

This guide covers deploying the Swarm Inference Protocol in production environments, including Kubernetes orchestration, Docker deployment, and operational procedures.

## Prerequisites

- Kubernetes cluster (v1.24+)
- Docker (v20.10+)
- Helm (v3.8+)
- kubectl configured for your cluster

## Quick Start with Kubernetes

### 1. Deploy with Helm

```bash
# Add the swarm-inference helm repository
helm repo add swarm-inference https://charts.swarm-inference.io
helm repo update

# Install the swarm inference cluster
helm install swarm-inference swarm-inference/swarm-inference \
  --namespace swarm-inference \
  --create-namespace \
  --set replicaCount=3 \
  --set coordinator.enabled=true
```

### 2. Verify Deployment

```bash
# Check pod status
kubectl get pods -n swarm-inference

# Check services
kubectl get svc -n swarm-inference

# View logs
kubectl logs -n swarm-inference -l app=swarm-inference
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SWARM_COORDINATOR_ADDR` | Coordinator node address | `127.0.0.1:8080` |
| `SWARM_NODE_ID` | Unique node identifier | Auto-generated |
| `SWARM_LISTEN_ADDR` | Listen address for this node | `0.0.0.0:8081` |
| `SWARM_TLS_CERT` | Path to TLS certificate | Auto-generated |
| `SWARM_TLS_KEY` | Path to TLS private key | Auto-generated |
| `SWARM_LOG_LEVEL` | Logging level (trace, debug, info, warn, error) | `info` |

### Kubernetes Configuration

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: swarm-inference
  namespace: swarm-inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: swarm-inference
  template:
    metadata:
      labels:
        app: swarm-inference
    spec:
      containers:
      - name: swarm-inference
        image: swarm-inference:latest
        ports:
        - containerPort: 8081
          name: quic
        env:
        - name: SWARM_COORDINATOR_ADDR
          value: "swarm-inference-coordinator:8080"
        - name: SWARM_NODE_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        volumeMounts:
        - name: tls-certs
          mountPath: /etc/ssl/certs
        - name: models
          mountPath: /models
  volumeClaimTemplates:
  - metadata:
    name: models
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 50Gi
```

## Docker Deployment

### Build the Image

```bash
# Build the Docker image
docker build -t swarm-inference:latest .

# Or use the multi-stage build
docker build --target production -t swarm-inference:prod .
```

### Run with Docker Compose

```yaml
version: '3.8'
services:
  coordinator:
    image: swarm-inference:latest
    command: ["coordinator", "--listen", "0.0.0.0:8080"]
    ports:
      - "8080:8080"
    environment:
      - SWARM_COORDINATOR_ADDR=127.0.0.1:8080
    volumes:
      - ./models:/models
      - ./config:/config

  node1:
    image: swarm-inference:latest
    command: ["node", "--id", "node1", "--coordinator", "coordinator:8080"]
    depends_on:
      - coordinator
    environment:
      - SWARM_COORDINATOR_ADDR=coordinator:8080
    volumes:
      - ./models:/models

  node2:
    image: swarm-inference:latest
    command: ["node", "--id", "node2", "--coordinator", "coordinator:8080"]
    depends_on:
      - coordinator
    environment:
      - SWARM_COORDINATOR_ADDR=coordinator:8080
    volumes:
      - ./models:/models
```

## Monitoring and Observability

### Health Checks

The system provides several health check endpoints:

- `/health` - Basic health check
- `/metrics` - Prometheus metrics
- `/ready` - Readiness probe

### Monitoring Setup

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'swarm-inference'
    static_configs:
      - targets: ['swarm-inference:8081']
    scrape_interval: 15s
```

### Alerting Rules

```yaml
# alert_rules.yml
groups:
  - name: swarm-inference
    rules:
      - alert: HighLatency
        expr: swarm_inference_request_duration_seconds{quantile="0.95"} > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High request latency detected"

      - alert: NodeDown
        expr: up{job="swarm-inference"} == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Swarm inference node is down"
```

## Scaling

### Horizontal Scaling

```bash
# Scale the deployment
kubectl scale statefulset swarm-inference --replicas=5 -n swarm-inference
```

### Auto-scaling with HPA

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: swarm-inference-hpa
  namespace: swarm-inference
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: StatefulSet
    name: swarm-inference
  minReplicas: 3
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
```

## Backup and Recovery

### Checkpoint Backup

```bash
# Backup checkpoints
kubectl exec -n swarm-inference swarm-inference-0 -- tar czf /tmp/checkpoints.tar.gz /data/checkpoints
kubectl cp swarm-inference/swarm-inference-0:/tmp/checkpoints.tar.gz ./backups/checkpoints.tar.gz
```

### Model Backup

```bash
# Backup models
kubectl exec -n swarm-inference swarm-inference-0 -- tar czf /tmp/models.tar.gz /models
kubectl cp swarm-inference/swarm-inference-0:/tmp/models.tar.gz ./backups/models.tar.gz
```

## Troubleshooting

### Common Issues

1. **Nodes can't connect to coordinator**
   - Check network policies
   - Verify service discovery
   - Check TLS certificates

2. **High latency**
   - Monitor network conditions
   - Check node resource utilization
   - Verify model sharding configuration

3. **Memory issues**
   - Adjust memory limits
   - Enable memory pooling
   - Monitor garbage collection

### Debug Commands

```bash
# Check node status
kubectl exec -n swarm-inference swarm-inference-0 -- curl http://localhost:8081/health

# View audit logs
kubectl logs -n swarm-inference -l app=swarm-inference | grep AUDIT

# Check consensus state
kubectl exec -n swarm-inference swarm-inference-0 -- curl http://localhost:8081/metrics | grep consensus
```

## Security Considerations

- Use TLS certificates for all communications
- Implement network policies to restrict traffic
- Enable RBAC for Kubernetes access
- Regularly rotate TLS certificates
- Monitor audit logs for security events

## Performance Tuning

### GPU Optimization

```yaml
# GPU-enabled deployment
apiVersion: apps/v1
kind: StatefulSet
spec:
  template:
    spec:
      containers:
      - name: swarm-inference
        resources:
          limits:
            nvidia.com/gpu: 1
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
```

### Network Optimization

- Use QUIC for better performance over unreliable networks
- Enable compression for tensor data
- Configure appropriate MTU settings
- Use node affinity for geographic routing</content>
<parameter name="filePath">X:\projects\GIT-LOKI\130326\swarm_inference\DEPLOYMENT.md