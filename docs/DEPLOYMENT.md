# Deployment Guide

Complete guide for deploying the Petroleum RAG Benchmark system to production environments.

## Table of Contents

- [Overview](#overview)
- [Docker Deployment](#docker-deployment)
- [Google Cloud Platform (GCP)](#google-cloud-platform-gcp)
- [Environment Variables](#environment-variables)
- [Security Best Practices](#security-best-practices)
- [Monitoring & Logging](#monitoring--logging)
- [Scaling Considerations](#scaling-considerations)
- [Backup & Recovery](#backup--recovery)

## Overview

The Petroleum RAG Benchmark can be deployed in several configurations:

1. **Local Development**: Docker Compose (current setup)
2. **Single Server**: Docker on dedicated VM
3. **Cloud Native**: GCP Cloud Run + managed services
4. **Enterprise**: Kubernetes cluster with auto-scaling

This guide covers the first three scenarios. For Kubernetes deployment, see the Kubernetes-specific documentation.

## Docker Deployment

### Local Development (Current Setup)

**Already configured** via `docker-compose.yml`:

```bash
# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Stop and remove data
docker-compose down -v
```

### Production Docker Deployment

For production deployment on a dedicated server:

#### 1. Server Requirements

**Minimum**:
- 4 CPU cores
- 8 GB RAM
- 50 GB SSD storage
- Ubuntu 20.04 LTS or similar

**Recommended**:
- 8+ CPU cores
- 16 GB RAM
- 100 GB SSD storage
- Ubuntu 22.04 LTS

#### 2. Install Docker

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation
docker --version
docker-compose --version
```

#### 3. Deploy Application

```bash
# Clone repository
git clone <repository-url>
cd petroleum-rag

# Set up environment
cp .env.example .env
nano .env  # Configure production settings

# Create production docker-compose
cat > docker-compose.prod.yml << 'EOF'
version: '3.8'

services:
  chroma:
    image: chromadb/chroma:latest
    container_name: petroleum-rag-chroma
    ports:
      - "127.0.0.1:8000:8000"  # Bind to localhost only
    volumes:
      - chroma_data:/chroma/chroma
    environment:
      - IS_PERSISTENT=TRUE
      - ANONYMIZED_TELEMETRY=FALSE
    restart: always
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  weaviate:
    image: semitechnologies/weaviate:1.24.4
    container_name: petroleum-rag-weaviate
    ports:
      - "127.0.0.1:8080:8080"  # Bind to localhost only
    environment:
      - AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true
      - PERSISTENCE_DATA_PATH=/var/lib/weaviate
      - DEFAULT_VECTORIZER_MODULE=none
      - QUERY_DEFAULTS_LIMIT=25
      - CLUSTER_HOSTNAME=node1
    volumes:
      - weaviate_data:/var/lib/weaviate
    restart: always
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  falkordb:
    image: falkordb/falkordb:latest
    container_name: petroleum-rag-falkordb
    ports:
      - "127.0.0.1:6379:6379"  # Bind to localhost only
    volumes:
      - falkordb_data:/data
    command: ["redis-server", "--loadmodule", "/usr/lib/redis/modules/falkordb.so"]
    restart: always
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

volumes:
  chroma_data:
    driver: local
  weaviate_data:
    driver: local
  falkordb_data:
    driver: local

networks:
  default:
    name: petroleum-rag-network
EOF

# Start production services
docker-compose -f docker-compose.prod.yml up -d

# Verify
docker-compose -f docker-compose.prod.yml ps
```

#### 4. Set Up Application

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .

# Verify setup
python verify_setup.py
```

#### 5. Configure Systemd Service

Create systemd service for automatic startup:

```bash
# Create service file
sudo nano /etc/systemd/system/petroleum-rag.service
```

Add:
```ini
[Unit]
Description=Petroleum RAG Benchmark Services
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/petroleum-rag
ExecStart=/usr/local/bin/docker-compose -f docker-compose.prod.yml up -d
ExecStop=/usr/local/bin/docker-compose -f docker-compose.prod.yml down
User=petroleum-rag
Group=petroleum-rag

[Install]
WantedBy=multi-user.target
```

Enable service:
```bash
sudo systemctl enable petroleum-rag
sudo systemctl start petroleum-rag
sudo systemctl status petroleum-rag
```

## Google Cloud Platform (GCP)

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Google Cloud Platform                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐        ┌──────────────┐                  │
│  │  Cloud Run   │        │   Cloud      │                  │
│  │  (API/Worker)│───────▶│   Storage    │                  │
│  │              │        │   (PDFs)     │                  │
│  └──────────────┘        └──────────────┘                  │
│         │                                                    │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────────────────────────────────┐              │
│  │       Managed Storage Backends            │              │
│  │  ┌──────────┐  ┌──────────┐  ┌────────┐ │              │
│  │  │MemoryStore│  │Weaviate  │  │Redis   │ │              │
│  │  │(Chroma)   │  │Cloud     │  │(Falkor)│ │              │
│  │  └──────────┘  └──────────┘  └────────┘ │              │
│  └──────────────────────────────────────────┘              │
│                                                              │
│  ┌──────────────────────────────────────────┐              │
│  │       External APIs                       │              │
│  │  - OpenAI (embeddings)                    │              │
│  │  - Anthropic (evaluation)                 │              │
│  │  - LlamaParse                             │              │
│  │  - Vertex Document AI                     │              │
│  └──────────────────────────────────────────┘              │
│                                                              │
│  ┌──────────────────────────────────────────┐              │
│  │       Monitoring & Logging                │              │
│  │  - Cloud Logging                          │              │
│  │  - Cloud Monitoring                       │              │
│  │  - Cloud Trace                            │              │
│  └──────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

### Setup Guide

#### 1. Prerequisites

```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init

# Set project
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    storage-api.googleapis.com \
    redis.googleapis.com \
    secretmanager.googleapis.com \
    logging.googleapis.com \
    monitoring.googleapis.com
```

#### 2. Set Up Storage Backends

##### Cloud Storage (for PDFs)

```bash
# Create bucket
gsutil mb -l us-central1 gs://petroleum-rag-documents

# Upload sample documents
gsutil cp data/input/*.pdf gs://petroleum-rag-documents/input/
```

##### Memorystore for Redis (FalkorDB)

```bash
# Create Redis instance
gcloud redis instances create petroleum-rag-redis \
    --size=5 \
    --region=us-central1 \
    --redis-version=redis_7_0 \
    --tier=basic

# Get connection details
gcloud redis instances describe petroleum-rag-redis \
    --region=us-central1
```

##### Weaviate Cloud

1. Sign up at https://console.weaviate.cloud
2. Create a cluster
3. Note the endpoint and API key

Or use self-hosted Weaviate on Compute Engine:

```bash
# Create VM for Weaviate
gcloud compute instances create petroleum-rag-weaviate \
    --machine-type=n1-standard-2 \
    --zone=us-central1-a \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=50GB

# SSH and install Weaviate
gcloud compute ssh petroleum-rag-weaviate --zone=us-central1-a

# On VM:
docker run -d \
    -p 8080:8080 \
    -v weaviate_data:/var/lib/weaviate \
    --name weaviate \
    semitechnologies/weaviate:1.24.4 \
    --host 0.0.0.0 \
    --port 8080 \
    --scheme http
```

#### 3. Store Secrets

```bash
# Create secrets
echo -n "your-anthropic-key" | gcloud secrets create anthropic-api-key --data-file=-
echo -n "your-openai-key" | gcloud secrets create openai-api-key --data-file=-
echo -n "your-llama-key" | gcloud secrets create llama-cloud-api-key --data-file=-

# Verify secrets
gcloud secrets list
```

#### 4. Create Dockerfile for Cloud Run

Create `Dockerfile.cloudrun`:

```dockerfile
# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# Expose port (Cloud Run requirement)
EXPOSE 8080

# Run benchmark
CMD ["python", "benchmark.py"]
```

#### 5. Build and Deploy

```bash
# Build container image
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/petroleum-rag

# Deploy to Cloud Run
gcloud run deploy petroleum-rag \
    --image gcr.io/YOUR_PROJECT_ID/petroleum-rag \
    --platform managed \
    --region us-central1 \
    --memory 4Gi \
    --cpu 2 \
    --timeout 3600 \
    --no-allow-unauthenticated \
    --set-env-vars "CHROMA_HOST=REDIS_IP,CHROMA_PORT=8000" \
    --set-secrets "ANTHROPIC_API_KEY=anthropic-api-key:latest,OPENAI_API_KEY=openai-api-key:latest,LLAMA_CLOUD_API_KEY=llama-cloud-api-key:latest"

# Get service URL
gcloud run services describe petroleum-rag \
    --platform managed \
    --region us-central1 \
    --format 'value(status.url)'
```

#### 6. Set Up Cloud Scheduler (Optional)

Run benchmarks on a schedule:

```bash
# Create service account
gcloud iam service-accounts create petroleum-rag-scheduler \
    --display-name="Petroleum RAG Scheduler"

# Grant invoker role
gcloud run services add-iam-policy-binding petroleum-rag \
    --member="serviceAccount:petroleum-rag-scheduler@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/run.invoker" \
    --region=us-central1

# Create scheduled job (daily at 2 AM)
gcloud scheduler jobs create http petroleum-rag-daily \
    --schedule="0 2 * * *" \
    --uri="SERVICE_URL/benchmark" \
    --http-method=POST \
    --oidc-service-account-email="petroleum-rag-scheduler@YOUR_PROJECT_ID.iam.gserviceaccount.com"
```

### Cost Estimation

**Monthly costs for moderate usage**:

| Service | Configuration | Estimated Cost |
|---------|--------------|----------------|
| Cloud Run | 4GB RAM, 2 CPU, 10 runs/day | $50-100 |
| Memorystore Redis | Basic, 5GB | $35 |
| Cloud Storage | 100GB documents | $2 |
| Cloud Logging | 50GB logs | $25 |
| External APIs (OpenAI, Anthropic) | Variable | $200-500 |
| **Total** | | **$312-662/month** |

**Cost Optimization**:
1. Use preemptible Cloud Run instances
2. Implement caching for embeddings
3. Batch process documents
4. Use Cloud Storage lifecycle policies
5. Reduce logging verbosity

## Environment Variables

### Required Variables

```bash
# API Keys (required)
ANTHROPIC_API_KEY=sk-ant-xxxxx
OPENAI_API_KEY=sk-xxxxx
LLAMA_CLOUD_API_KEY=llx-xxxxx

# Storage Backend URLs (required)
CHROMA_HOST=localhost  # or IP address
CHROMA_PORT=8000
WEAVIATE_HOST=localhost
WEAVIATE_PORT=8080
FALKORDB_HOST=localhost
FALKORDB_PORT=6379
```

### Optional Variables

```bash
# Google Cloud (optional, for Vertex Document AI)
GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
GOOGLE_CLOUD_PROJECT=your-project-id
VERTEX_DOCAI_PROCESSOR_ID=processor-id
VERTEX_DOCAI_LOCATION=us

# Chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MIN_CHUNK_SIZE=100
MAX_CHUNK_SIZE=2000

# Embeddings
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
EMBEDDING_BATCH_SIZE=100

# Evaluation
EVAL_LLM_MODEL=claude-sonnet-4-20250514
EVAL_LLM_TEMPERATURE=0.0
EVAL_LLM_MAX_TOKENS=4096

# Retrieval
RETRIEVAL_TOP_K=5
RETRIEVAL_MIN_SCORE=0.5

# Benchmark
BENCHMARK_PARALLEL_PARSERS=true
BENCHMARK_PARALLEL_STORAGE=true
BENCHMARK_SAVE_INTERMEDIATE_RESULTS=true

# Logging
LOG_LEVEL=INFO
LOG_FILE=benchmark.log
DEBUG=false
```

### Managing Secrets

#### Local Development

```bash
# Use .env file
cp .env.example .env
nano .env
```

#### Docker

```bash
# Use environment file
docker-compose --env-file .env.prod up -d
```

#### GCP

```bash
# Use Secret Manager
gcloud secrets create my-secret --data-file=-
gcloud run services update petroleum-rag \
    --set-secrets "ENV_VAR=secret-name:latest"
```

#### Kubernetes

```bash
# Create secret
kubectl create secret generic petroleum-rag-secrets \
    --from-literal=anthropic-api-key=xxxxx \
    --from-literal=openai-api-key=xxxxx

# Reference in deployment
env:
  - name: ANTHROPIC_API_KEY
    valueFrom:
      secretKeyRef:
        name: petroleum-rag-secrets
        key: anthropic-api-key
```

## Security Best Practices

### 1. API Key Management

**Do**:
- Use secret management services (GCP Secret Manager, AWS Secrets Manager)
- Rotate keys regularly (every 90 days)
- Use different keys for dev/staging/prod
- Never commit keys to version control
- Use environment variables or secret files

**Don't**:
- Hardcode keys in source code
- Share keys between environments
- Log API keys
- Use default/demo keys in production

### 2. Network Security

**Firewall Rules**:
```bash
# Allow only necessary ports
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable

# Block direct access to storage backends
# Access only via localhost or VPN
```

**Docker Network Isolation**:
```yaml
# In docker-compose.prod.yml
services:
  chroma:
    ports:
      - "127.0.0.1:8000:8000"  # Localhost only
    networks:
      - backend
networks:
  backend:
    internal: true  # No external access
```

### 3. Authentication & Authorization

**API Authentication**:
```python
# Add API key validation
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

def validate_api_key(api_key: str = Security(api_key_header)):
    if api_key != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key
```

**GCP IAM**:
```bash
# Principle of least privilege
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:petroleum-rag@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer"  # Read-only access
```

### 4. Data Protection

**Encryption at Rest**:
- Use encrypted volumes for Docker
- Enable GCP encryption (automatic)
- Encrypt sensitive data before storage

**Encryption in Transit**:
- Use HTTPS/TLS for all API calls
- Enable TLS for storage backends
- Use VPN for inter-service communication

**Data Retention**:
```bash
# Set retention policies
gsutil retention set 90d gs://petroleum-rag-documents

# Lifecycle policy for old data
gsutil lifecycle set lifecycle.json gs://petroleum-rag-documents
```

### 5. Logging & Auditing

**Security Logging**:
```python
# Log security events
logger.info(f"Authentication attempt from {ip_address}")
logger.warning(f"Invalid API key used: {masked_key}")
logger.error(f"Unauthorized access attempt to {resource}")
```

**GCP Audit Logs**:
```bash
# Enable audit logging
gcloud logging read "resource.type=gce_instance AND logName=projects/YOUR_PROJECT_ID/logs/cloudaudit.googleapis.com%2Factivity"
```

### 6. Dependency Security

**Regular Updates**:
```bash
# Check for vulnerabilities
pip install safety
safety check

# Update dependencies
pip install --upgrade pip
pip install --upgrade -r requirements.txt
```

**Container Scanning**:
```bash
# Scan Docker images
gcloud container images scan gcr.io/YOUR_PROJECT_ID/petroleum-rag

# View vulnerabilities
gcloud container images describe gcr.io/YOUR_PROJECT_ID/petroleum-rag \
    --show-package-vulnerability
```

## Monitoring & Logging

### Application Logging

**Structure Logs**:
```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }
        return json.dumps(log_obj)

# Configure logger
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger = logging.getLogger(__name__)
logger.addHandler(handler)
```

**Log Levels**:
- **ERROR**: Failures requiring immediate attention
- **WARNING**: Potential issues or degraded performance
- **INFO**: Key events (benchmark start/complete, parsing progress)
- **DEBUG**: Detailed diagnostic information

### GCP Cloud Logging

**View Logs**:
```bash
# Real-time logs
gcloud logging tail "resource.type=cloud_run_revision"

# Filter logs
gcloud logging read "resource.type=cloud_run_revision AND severity>=ERROR" \
    --limit 50 \
    --format json

# Export logs
gcloud logging sinks create petroleum-rag-logs \
    storage.googleapis.com/petroleum-rag-logs
```

### Monitoring Metrics

**Key Metrics**:
- Benchmark completion rate
- Average query time
- Error rate
- API usage and costs
- Storage backend health

**GCP Cloud Monitoring**:
```bash
# Create alert policy
gcloud alpha monitoring policies create \
    --notification-channels=CHANNEL_ID \
    --display-name="High Error Rate" \
    --condition-threshold-value=0.05 \
    --condition-threshold-duration=60s
```

### Health Checks

**Application Health Endpoint**:
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers."""
    # Check storage backends
    storage_ok = await check_storage_health()

    # Check API connectivity
    api_ok = await check_api_connectivity()

    if storage_ok and api_ok:
        return {"status": "healthy"}
    else:
        return {"status": "unhealthy"}, 503
```

**Docker Health Check**:
```yaml
services:
  api:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

## Scaling Considerations

### Vertical Scaling

**Increase Resources**:
```bash
# GCP Cloud Run
gcloud run services update petroleum-rag \
    --memory 8Gi \
    --cpu 4

# Docker
# Edit docker-compose.yml
services:
  benchmark:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
```

### Horizontal Scaling

**Load Balancing**:
```bash
# GCP Cloud Run (automatic)
gcloud run services update petroleum-rag \
    --min-instances 1 \
    --max-instances 10 \
    --concurrency 80
```

**Distributed Processing**:
```python
# Process documents in parallel across workers
from multiprocessing import Pool

def process_document(pdf_path):
    # Parse, chunk, embed, store
    ...

if __name__ == "__main__":
    with Pool(processes=4) as pool:
        results = pool.map(process_document, pdf_files)
```

### Storage Scaling

**ChromaDB**:
- Use persistent volumes
- Consider hosted solution for > 1M vectors

**Weaviate**:
- Use Weaviate Cloud for auto-scaling
- Or deploy cluster on Kubernetes

**FalkorDB/Redis**:
- Use GCP Memorystore with replica
- Consider Redis Cluster for large datasets

## Backup & Recovery

### Data Backup

**Docker Volumes**:
```bash
# Backup volumes
docker run --rm \
    --volumes-from petroleum-rag-chroma \
    -v $(pwd):/backup \
    ubuntu tar czf /backup/chroma-backup.tar.gz /chroma/chroma

# Restore volumes
docker run --rm \
    --volumes-from petroleum-rag-chroma \
    -v $(pwd):/backup \
    ubuntu tar xzf /backup/chroma-backup.tar.gz -C /
```

**GCP Cloud Storage**:
```bash
# Backup to Cloud Storage
gsutil -m rsync -r /data/results gs://petroleum-rag-backups/results

# Schedule daily backups
0 3 * * * gsutil -m rsync -r /data/results gs://petroleum-rag-backups/results/$(date +\%Y-\%m-\%d)
```

### Disaster Recovery

**Recovery Time Objective (RTO)**: 4 hours
**Recovery Point Objective (RPO)**: 24 hours

**Recovery Steps**:
1. Restore storage backend data from backups
2. Redeploy application containers
3. Verify storage connectivity
4. Run verification benchmark
5. Resume normal operations

**Testing**:
```bash
# Test recovery quarterly
# 1. Stop services
docker-compose down

# 2. Simulate data loss
docker volume rm petroleum-rag_chroma_data

# 3. Restore from backup
# ... restore steps ...

# 4. Verify
docker-compose up -d
python verify_setup.py
```

---

## Troubleshooting

### Common Deployment Issues

**Issue**: Cloud Run timeouts
**Solution**: Increase timeout to 3600s, use async operations

**Issue**: Out of memory
**Solution**: Increase memory allocation, reduce batch sizes

**Issue**: Storage backend connection failures
**Solution**: Check firewall rules, verify network connectivity

**Issue**: High API costs
**Solution**: Implement caching, reduce query frequency

### Getting Help

- Check logs: `docker logs <container-name>`
- Review metrics: GCP Console → Cloud Run → Metrics
- Test connectivity: `curl -f http://localhost:8000/api/v1/heartbeat`
- Open support ticket or GitHub issue

---

**Next Steps**:
- Set up monitoring dashboards
- Configure alerts for critical metrics
- Implement automated backups
- Document runbooks for common scenarios
- Plan capacity for expected growth
