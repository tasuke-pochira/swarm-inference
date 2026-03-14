# Swarm Inference Kubernetes Deployment

This directory contains Kubernetes manifests for deploying the Swarm Inference Protocol on a Kubernetes cluster.

## Prerequisites

- Kubernetes cluster (v1.19+)
- kubectl configured to access the cluster
- Docker registry access for the swarm-inference image
- (Optional) Ingress controller (nginx recommended)
- (Optional) cert-manager for TLS certificates
- (Optional) GPU nodes for GPU acceleration

## Quick Start

1. **Build and push the Docker image:**
   ```bash
   docker build -t your-registry/swarm-inference:latest .
   docker push your-registry/swarm-inference:latest
   ```

2. **Update image references in the manifests:**
   ```bash
   find kubernetes/ -name "*.yaml" -exec sed -i 's|swarm-inference:latest|your-registry/swarm-inference:latest|g' {} \;
   ```

3. **Deploy the system:**
   ```bash
   kubectl apply -f kubernetes/
   ```

4. **Check deployment status:**
   ```bash
   kubectl get pods -n swarm-inference
   kubectl get services -n swarm-inference
   ```

## Architecture

The deployment consists of:

- **Coordinator**: Single replica deployment that orchestrates inference requests
- **Nodes**: StatefulSet with configurable replicas for distributed computation
- **Dashboard**: Web interface for monitoring and metrics
- **Services**: Internal networking between components
- **ConfigMaps**: Configuration management
- **RBAC**: Permissions for Kubernetes API access

## Configuration

### Scaling

Adjust the number of nodes by modifying the StatefulSet:

```bash
kubectl scale statefulset swarm-nodes -n swarm-inference --replicas=5
```

### Resource Limits

Update resource requests/limits in the deployment files based on your workload:

```yaml
resources:
  requests:
    cpu: 1000m
    memory: 2Gi
  limits:
    cpu: 4000m
    memory: 8Gi
```

### GPU Support

For GPU-enabled nodes, ensure your cluster has GPU nodes and update the StatefulSet:

```yaml
spec:
  template:
    spec:
      nodeSelector:
        accelerator: nvidia-tesla-k80  # Adjust based on your GPU type
      containers:
      - resources:
          limits:
            nvidia.com/gpu: 1  # Request 1 GPU
```

## Monitoring

### Dashboard Access

Access the monitoring dashboard:

```bash
kubectl port-forward -n swarm-inference svc/swarm-dashboard 9090:9090
# Visit http://localhost:9090
```

### Metrics

View metrics programmatically:

```bash
kubectl exec -n swarm-inference deployment/swarm-coordinator -- ./swarm_inference metrics
```

### Logs

View component logs:

```bash
# Coordinator logs
kubectl logs -n swarm-inference deployment/swarm-coordinator

# Node logs
kubectl logs -n swarm-inference statefulset/swarm-nodes

# Dashboard logs
kubectl logs -n swarm-inference deployment/swarm-dashboard
```

## Troubleshooting

### Common Issues

1. **Pods not starting:**
   ```bash
   kubectl describe pod <pod-name> -n swarm-inference
   ```

2. **Network connectivity issues:**
   ```bash
   kubectl exec -it <pod-name> -n swarm-inference -- nslookup swarm-coordinator.swarm-inference.svc.cluster.local
   ```

3. **Resource constraints:**
   ```bash
   kubectl describe nodes  # Check node capacity
   ```

### Health Checks

The deployment includes liveness and readiness probes. Check probe status:

```bash
kubectl get pods -n swarm-inference -o wide
kubectl describe pod <pod-name> -n swarm-inference | grep -A 5 "Liveness\|Readiness"
```

## Production Considerations

### High Availability

- The coordinator uses a single replica. Consider using a StatefulSet with leader election for HA.
- Nodes use a StatefulSet with persistent storage for model data.
- PodDisruptionBudgets ensure minimum availability during maintenance.

### Security

- RBAC is configured with minimal required permissions.
- Network policies restrict traffic between pods.
- Consider using secrets for sensitive configuration.

### Storage

- Model data is stored on persistent volumes.
- Configure appropriate storage classes for your environment.
- Consider backup strategies for model data.

### Networking

- Services use ClusterIP for internal communication.
- Ingress provides external access (configure with your domain).
- TLS is supported via cert-manager.

## Development

### Local Development

For local development with minikube:

```bash
minikube start --kubernetes-version=v1.25.0
minikube addons enable ingress
eval $(minikube docker-env)  # Build images directly in minikube
```

### CI/CD Integration

The manifests can be integrated into CI/CD pipelines:

```bash
# Update image tag
sed -i 's|swarm-inference:latest|swarm-inference:v1.2.3|g' kubernetes/*.yaml

# Deploy
kubectl apply -f kubernetes/
```

## Cleanup

Remove all resources:

```bash
kubectl delete namespace swarm-inference
```

Or remove specific components:

```bash
kubectl delete -f kubernetes/ingress.yaml
kubectl delete -f kubernetes/hpa.yaml
# etc.
```