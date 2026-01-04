# Kubernetes Deployment

This directory contains Kubernetes manifests for deploying the Heart Disease Prediction API.

## Prerequisites

- Kubernetes cluster (GKE, EKS, AKS, Minikube, or Docker Desktop with Kubernetes enabled)
- `kubectl` CLI installed and configured
- Docker image pushed to DockerHub (or another container registry)

## Files

- `deployment.yaml` - Deployment with 2 replicas, health checks, resource limits, and HPA
- `service.yaml` - LoadBalancer service exposing the API on port 80

## Quick Start

### 1. Update Docker Image

Edit `deployment.yaml` and replace `YOUR_DOCKERHUB_USERNAME` with your actual DockerHub username:

```yaml
image: YOUR_DOCKERHUB_USERNAME/heart-predict-api:latest
```

Or use `sed` to replace it:

```bash
sed -i 's/YOUR_DOCKERHUB_USERNAME/your-username/g' k8s/deployment.yaml
```

### 2. Apply Manifests

Deploy to your Kubernetes cluster:

```bash
kubectl apply -f k8s/
```

This creates:
- Deployment with 2 pod replicas
- HorizontalPodAutoscaler (scales from 2 to 10 replicas based on CPU/memory)
- LoadBalancer service

### 3. Verify Deployment

Check that pods are running:

```bash
kubectl get pods -l app=heart-predict-api
```

Expected output:
```
NAME                                  READY   STATUS    RESTARTS   AGE
heart-predict-api-xxxxxxxxxx-xxxxx    1/1     Running   0          30s
heart-predict-api-xxxxxxxxxx-xxxxx    1/1     Running   0          30s
```

### 4. Get Service External IP

Wait for the LoadBalancer to assign an external IP:

```bash
kubectl get svc heart-predict-api
```

Expected output:
```
NAME                TYPE           CLUSTER-IP      EXTERNAL-IP      PORT(S)        AGE
heart-predict-api   LoadBalancer   10.x.x.x        x.x.x.x          80:xxxxx/TCP   1m
```

**Note:** On Minikube or Docker Desktop, you may need to use:
```bash
minikube service heart-predict-api --url
# or
kubectl port-forward svc/heart-predict-api 8080:80
```

### 5. Test the API

Once you have the external IP (or port-forward):

**Health check:**
```bash
curl http://<EXTERNAL-IP>/health
```

**Prediction request:**
```bash
curl -X POST http://<EXTERNAL-IP>/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1
  }'
```

Expected response:
```json
{
  "prediction": 1,
  "probability": 0.85
}
```

## Monitoring

View logs:
```bash
kubectl logs -l app=heart-predict-api --tail=100 -f
```

Check HPA status:
```bash
kubectl get hpa heart-predict-api-hpa
```

Describe deployment:
```bash
kubectl describe deployment heart-predict-api
```

## Scaling

Manual scaling (overrides HPA temporarily):
```bash
kubectl scale deployment heart-predict-api --replicas=5
```

## Updating

To update to a new image version:

```bash
kubectl set image deployment/heart-predict-api \
  heart-predict-api=YOUR_DOCKERHUB_USERNAME/heart-predict-api:v2
```

Or edit the YAML and reapply:
```bash
kubectl apply -f k8s/deployment.yaml
```

## Cleanup

Remove all resources:

```bash
kubectl delete -f k8s/
```

## Cloud-Specific Notes

### Google Kubernetes Engine (GKE)
- LoadBalancer automatically provisions a Google Cloud Load Balancer
- External IP appears within 1-2 minutes

### Amazon EKS
- LoadBalancer provisions an AWS Classic or Network Load Balancer
- Requires proper IAM permissions

### Azure AKS
- LoadBalancer provisions an Azure Load Balancer
- External IP assigned from cluster's resource group

### Minikube / Docker Desktop
- LoadBalancer stays in "Pending" state (no cloud provider)
- Use `minikube service` or `kubectl port-forward` instead

## Production Considerations

For production deployments, consider:

1. **Ingress Controller**: Use NGINX Ingress or Traefik instead of LoadBalancer for better control
2. **TLS/SSL**: Add cert-manager for automatic HTTPS certificates
3. **Resource Limits**: Adjust based on actual usage patterns
4. **Monitoring**: Deploy Prometheus + Grafana for metrics
5. **Logging**: Use EFK/ELK stack or cloud-native logging
6. **Secrets Management**: Use Kubernetes Secrets or external secret managers
7. **Network Policies**: Restrict pod-to-pod communication
8. **Pod Disruption Budgets**: Ensure high availability during updates
