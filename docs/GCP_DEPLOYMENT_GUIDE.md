# Complete GCP Deployment Guide - Petroleum RAG Benchmark

## 🎯 Recommended Architecture

For your use case (Streamlit UI + background processing + document storage), here's the **best approach**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Google Cloud Platform                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────┐      ┌──────────────────┐                │
│  │  Cloud Run       │      │  Cloud Storage   │                │
│  │  (Streamlit UI)  │─────▶│  (Documents)     │                │
│  │  Port: 8501      │      │  - Input PDFs    │                │
│  └──────────────────┘      │  - Results       │                │
│                             │  - Cache         │                │
│  ┌──────────────────┐      └──────────────────┘                │
│  │  Compute Engine  │                                            │
│  │  (VM Instance)   │      ┌──────────────────────────────────┐│
│  │  - Docker        │      │  Managed Services                 ││
│  │  - Chroma        │      │  ┌─────────────────────────────┐ ││
│  │  - Weaviate      │      │  │ Memorystore Redis           │ ││
│  │  - FalkorDB      │      │  │ (for FalkorDB)              │ ││
│  │  - Processing    │      │  └─────────────────────────────┘ ││
│  └──────────────────┘      │  ┌─────────────────────────────┐ ││
│                             │  │ Secret Manager              │ ││
│                             │  │ (API Keys)                  │ ││
│                             │  └─────────────────────────────┘ ││
│  ┌──────────────────┐      │  ┌─────────────────────────────┐ ││
│  │  Cloud Monitoring│      │  │ Cloud Logging               │ ││
│  │  + Grafana       │      │  │ (Application Logs)          │ ││
│  └──────────────────┘      │  └─────────────────────────────┘ ││
│                             └──────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

**Why this architecture?**
- ✅ **Cloud Run** for Streamlit = Auto-scaling, serverless, cost-effective
- ✅ **Compute Engine VM** for storage backends = Full control, persistent state
- ✅ **Cloud Storage** for documents = Scalable, versioned, accessible
- ✅ **Memorystore Redis** for FalkorDB = Managed, reliable, fast
- ✅ **Secret Manager** for API keys = Secure, rotatable, audited

---

## 💰 Cost Estimate

### Monthly Costs (Moderate Usage)

| Service | Configuration | Est. Monthly Cost |
|---------|--------------|-------------------|
| **Compute Engine** | e2-standard-4 (4 vCPU, 16 GB) | $120 |
| **Cloud Run** | Streamlit UI (minimal traffic) | $20 |
| **Memorystore Redis** | Basic tier, 5 GB | $50 |
| **Cloud Storage** | 100 GB documents + results | $2 |
| **Secret Manager** | ~10 secrets | $1 |
| **Cloud Logging** | 10 GB logs/month | $5 |
| **Egress** | External API calls | $10 |
| **TOTAL** | | **~$208/month** |

**API Costs (per benchmark run):**
- OpenAI Embeddings: ~$0.23 (first run) → $0.01 (cached)
- Anthropic Claude: ~$1.80 (first run) → $0.06 (cached)
- LlamaParse: ~$0.50 per document
- **Total per run**: ~$2.50 first time, ~$0.60 cached

---

## 🚀 Deployment Options

### Option 1: **Simple & Fast** (Recommended for POC)
Single VM with everything, Streamlit on Cloud Run
- ⏱️ Setup time: **30 minutes**
- 💰 Cost: **~$140/month**
- 🎯 Best for: POC, testing, low traffic

### Option 2: **Production Ready** (Recommended for Production)
Separate services, managed databases, monitoring
- ⏱️ Setup time: **2-3 hours**
- 💰 Cost: **~$208/month**
- 🎯 Best for: Production, scaling, high availability

### Option 3: **Enterprise** (Full GKE)
Kubernetes cluster, auto-scaling, multi-region
- ⏱️ Setup time: **1 day**
- 💰 Cost: **~$500+/month**
- 🎯 Best for: Enterprise, multi-tenant, global scale

**This guide covers Options 1 & 2.**

---

## 📋 Prerequisites

### 1. GCP Account & Project

```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Login and set project
gcloud auth login
gcloud projects create petroleum-rag-prod --name="Petroleum RAG Benchmark"
gcloud config set project petroleum-rag-prod

# Enable billing (replace BILLING_ACCOUNT_ID)
gcloud beta billing projects link petroleum-rag-prod \
  --billing-account=BILLING_ACCOUNT_ID
```

### 2. Enable Required APIs

```bash
gcloud services enable \
    compute.googleapis.com \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    storage.googleapis.com \
    redis.googleapis.com \
    secretmanager.googleapis.com \
    logging.googleapis.com \
    monitoring.googleapis.com \
    artifactregistry.googleapis.com \
    cloudresourcemanager.googleapis.com
```

### 3. Set Variables

```bash
# Set your configuration
export PROJECT_ID="petroleum-rag-prod"
export REGION="us-central1"
export ZONE="us-central1-a"
export VM_NAME="petroleum-rag-vm"
export BUCKET_NAME="${PROJECT_ID}-documents"

# Verify
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "VM: $VM_NAME"
```

---

## 🏗️ Option 1: Simple & Fast Deployment

### Step 1: Create Storage Bucket

```bash
# Create bucket for documents and results
gsutil mb -l $REGION gs://$BUCKET_NAME

# Create directory structure
gsutil -m cp -r data/input/* gs://$BUCKET_NAME/input/ 2>/dev/null || true

# Set lifecycle policy (optional - auto-delete old results after 90 days)
cat > lifecycle.json << EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {
          "age": 90,
          "matchesPrefix": ["results/"]
        }
      }
    ]
  }
}
EOF

gsutil lifecycle set lifecycle.json gs://$BUCKET_NAME
```

### Step 2: Store Secrets

```bash
# Store API keys in Secret Manager
echo -n "YOUR_ANTHROPIC_KEY" | gcloud secrets create anthropic-api-key \
  --data-file=- --replication-policy="automatic"

echo -n "YOUR_OPENAI_KEY" | gcloud secrets create openai-api-key \
  --data-file=- --replication-policy="automatic"

echo -n "YOUR_LLAMA_KEY" | gcloud secrets create llama-cloud-api-key \
  --data-file=- --replication-policy="automatic"

# Optional: Vertex AI key
echo -n "YOUR_VERTEX_KEY" | gcloud secrets create vertex-api-key \
  --data-file=- --replication-policy="automatic"

# Verify
gcloud secrets list
```

### Step 3: Create Compute Engine VM

```bash
# Create VM with Docker pre-installed
gcloud compute instances create $VM_NAME \
    --zone=$ZONE \
    --machine-type=e2-standard-4 \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --scopes=cloud-platform \
    --tags=http-server,https-server,petroleum-rag \
    --metadata=startup-script='#!/bin/bash
# Update system
apt-get update
apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install Python 3.11
apt-get install -y python3.11 python3.11-venv python3-pip

# Create app directory
mkdir -p /opt/petroleum-rag
chown -R $USER:$USER /opt/petroleum-rag
'

# Wait for VM to be ready
echo "Waiting for VM to be ready..."
sleep 30

# Create firewall rule for Streamlit
gcloud compute firewall-rules create allow-streamlit \
    --allow=tcp:8501 \
    --source-ranges=0.0.0.0/0 \
    --target-tags=petroleum-rag \
    --description="Allow Streamlit traffic"

# Create firewall rule for monitoring
gcloud compute firewall-rules create allow-monitoring \
    --allow=tcp:9090,tcp:3001,tcp:9091 \
    --source-ranges=0.0.0.0/0 \
    --target-tags=petroleum-rag \
    --description="Allow monitoring traffic"

echo "✓ VM created: $VM_NAME"
echo "✓ Firewall rules created"
```

### Step 4: Deploy Application to VM

```bash
# Copy application code to VM
gcloud compute scp --recurse \
    --zone=$ZONE \
    /Users/gregorydickson/petroleum-rag/* \
    $VM_NAME:/opt/petroleum-rag/

# SSH into VM and set up
gcloud compute ssh $VM_NAME --zone=$ZONE << 'ENDSSH'
cd /opt/petroleum-rag

# Create .env file with secrets
cat > .env << EOF
# Get from Secret Manager
ANTHROPIC_API_KEY=$(gcloud secrets versions access latest --secret=anthropic-api-key)
OPENAI_API_KEY=$(gcloud secrets versions access latest --secret=openai-api-key)
LLAMA_CLOUD_API_KEY=$(gcloud secrets versions access latest --secret=llama-cloud-api-key)

# Storage configuration
CHROMA_HOST=localhost
CHROMA_PORT=8000
WEAVIATE_HOST=localhost
WEAVIATE_PORT=8080
FALKORDB_HOST=localhost
FALKORDB_PORT=6379

# GCS bucket
GCS_BUCKET=petroleum-rag-prod-documents

# Monitoring
ENABLE_CACHE=true
ENABLE_MONITORING=true
EOF

# Create Python virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -e .

# Start Docker services
docker-compose up -d

# Wait for services
sleep 15

# Check Docker status
docker-compose ps

echo "✓ Application deployed"
echo "✓ Docker services running"
ENDSSH

# Get VM external IP
VM_IP=$(gcloud compute instances describe $VM_NAME \
    --zone=$ZONE \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

echo ""
echo "═══════════════════════════════════════════════"
echo "  🎉 Deployment Complete!"
echo "═══════════════════════════════════════════════"
echo ""
echo "VM External IP: $VM_IP"
echo ""
echo "To access services:"
echo "  1. SSH to VM: gcloud compute ssh $VM_NAME --zone=$ZONE"
echo "  2. Run: cd /opt/petroleum-rag && ./start_app.sh"
echo "  3. Access UI: http://$VM_IP:8501"
echo ""
```

### Step 5: Start Application

```bash
# SSH into VM
gcloud compute ssh $VM_NAME --zone=$ZONE

# Navigate to app directory
cd /opt/petroleum-rag

# Download documents from GCS (if needed)
gsutil -m cp -r gs://$BUCKET_NAME/input/* data/input/

# Start application
./start_app.sh

# Application will:
# ✓ Process documents
# ✓ Run benchmark
# ✓ Generate analysis
# ✓ Launch Streamlit UI
```

### Step 6: Access Application

```bash
# Get VM IP
VM_IP=$(gcloud compute instances describe $VM_NAME \
    --zone=$ZONE \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

echo "Access points:"
echo "  Streamlit UI:    http://$VM_IP:8501"
echo "  Monitoring:      http://$VM_IP:9090"
echo "  Grafana:         http://$VM_IP:3001"
echo "  Prometheus:      http://$VM_IP:9091"
```

---

## 🏢 Option 2: Production Ready Deployment

### Additional Steps for Production

#### 1. Use Managed Redis for FalkorDB

```bash
# Create Memorystore Redis instance
gcloud redis instances create petroleum-rag-redis \
    --size=5 \
    --region=$REGION \
    --redis-version=redis_7_0 \
    --tier=basic \
    --network=default

# Get Redis host
REDIS_HOST=$(gcloud redis instances describe petroleum-rag-redis \
    --region=$REGION \
    --format='get(host)')

REDIS_PORT=$(gcloud redis instances describe petroleum-rag-redis \
    --region=$REGION \
    --format='get(port)')

echo "Redis Host: $REDIS_HOST"
echo "Redis Port: $REDIS_PORT"

# Update .env on VM with Redis connection
gcloud compute ssh $VM_NAME --zone=$ZONE << ENDSSH
cd /opt/petroleum-rag
sed -i "s/FALKORDB_HOST=localhost/FALKORDB_HOST=$REDIS_HOST/g" .env
sed -i "s/FALKORDB_PORT=6379/FALKORDB_PORT=$REDIS_PORT/g" .env
ENDSSH
```

#### 2. Deploy Streamlit to Cloud Run

Create `Dockerfile.streamlit`:

```bash
cat > Dockerfile.streamlit << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application
COPY pyproject.toml .
COPY . .

# Install Python packages
RUN pip install --no-cache-dir -e .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "demo_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
EOF
```

Build and deploy:

```bash
# Build container
gcloud builds submit --tag gcr.io/$PROJECT_ID/petroleum-rag-ui \
    -f Dockerfile.streamlit .

# Deploy to Cloud Run
gcloud run deploy petroleum-rag-ui \
    --image gcr.io/$PROJECT_ID/petroleum-rag-ui \
    --platform managed \
    --region $REGION \
    --memory 2Gi \
    --cpu 2 \
    --timeout 3600 \
    --allow-unauthenticated \
    --set-secrets "ANTHROPIC_API_KEY=anthropic-api-key:latest,OPENAI_API_KEY=openai-api-key:latest,LLAMA_CLOUD_API_KEY=llama-cloud-api-key:latest" \
    --set-env-vars "CHROMA_HOST=$VM_IP,WEAVIATE_HOST=$VM_IP,FALKORDB_HOST=$REDIS_HOST"

# Get Cloud Run URL
CLOUD_RUN_URL=$(gcloud run services describe petroleum-rag-ui \
    --region=$REGION \
    --format='value(status.url)')

echo "✓ Streamlit deployed to Cloud Run"
echo "✓ URL: $CLOUD_RUN_URL"
```

#### 3. Set Up Load Balancer (Optional)

```bash
# Reserve static IP
gcloud compute addresses create petroleum-rag-ip \
    --global

# Get reserved IP
STATIC_IP=$(gcloud compute addresses describe petroleum-rag-ip \
    --global \
    --format='get(address)')

echo "Static IP: $STATIC_IP"

# Configure Cloud Armor (DDoS protection)
gcloud compute security-policies create petroleum-rag-policy \
    --description="Security policy for Petroleum RAG"

# Create backend service pointing to Cloud Run
# (Full load balancer setup requires additional steps)
```

#### 4. Set Up Monitoring & Alerts

```bash
# Create notification channel (email)
gcloud alpha monitoring channels create \
    --display-name="Petroleum RAG Alerts" \
    --type=email \
    --channel-labels=email_address=YOUR_EMAIL@example.com

# Create alert for high CPU
gcloud alpha monitoring policies create \
    --notification-channels=CHANNEL_ID \
    --display-name="High CPU Usage" \
    --condition-display-name="CPU > 80%" \
    --condition-threshold-value=0.8 \
    --condition-threshold-duration=300s

# Create alert for errors
gcloud alpha monitoring policies create \
    --notification-channels=CHANNEL_ID \
    --display-name="Application Errors" \
    --condition-display-name="Error rate > 5%" \
    --condition-threshold-value=0.05
```

#### 5. Set Up Backup & Disaster Recovery

```bash
# Create snapshot schedule for VM disk
gcloud compute resource-policies create snapshot-schedule petroleum-rag-daily \
    --region=$REGION \
    --max-retention-days=7 \
    --on-source-disk-delete=keep-auto-snapshots \
    --daily-schedule \
    --start-time=02:00

# Attach to VM disk
gcloud compute disks add-resource-policies $VM_NAME \
    --resource-policies=petroleum-rag-daily \
    --zone=$ZONE

# Backup Cloud Storage bucket
gsutil -m rsync -r gs://$BUCKET_NAME gs://${BUCKET_NAME}-backup

# Set up Cloud Storage versioning
gsutil versioning set on gs://$BUCKET_NAME
```

---

## 🔧 Configuration & Management

### Update Environment Variables

```bash
# SSH to VM
gcloud compute ssh $VM_NAME --zone=$ZONE

# Edit .env
cd /opt/petroleum-rag
nano .env

# Restart services
docker-compose restart
```

### View Logs

```bash
# Cloud Logging
gcloud logging read "resource.type=gce_instance AND resource.labels.instance_id=$VM_NAME" \
    --limit 50 \
    --format json

# VM application logs
gcloud compute ssh $VM_NAME --zone=$ZONE
cd /opt/petroleum-rag
tail -f logs/*.log
```

### Scale VM

```bash
# Stop VM
gcloud compute instances stop $VM_NAME --zone=$ZONE

# Resize
gcloud compute instances set-machine-type $VM_NAME \
    --zone=$ZONE \
    --machine-type=e2-standard-8

# Start VM
gcloud compute instances start $VM_NAME --zone=$ZONE
```

### Add Documents

```bash
# Upload from local
gsutil cp your-document.pdf gs://$BUCKET_NAME/input/

# Download to VM
gcloud compute ssh $VM_NAME --zone=$ZONE
cd /opt/petroleum-rag
gsutil cp gs://$BUCKET_NAME/input/your-document.pdf data/input/

# Rerun benchmark
./start_app.sh
```

---

## 📊 Monitoring Dashboard

### Access Built-in Monitoring

```bash
# Get VM IP
VM_IP=$(gcloud compute instances describe $VM_NAME \
    --zone=$ZONE \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

# Access Grafana
open http://$VM_IP:3001
# Default login: admin/admin

# Access Prometheus
open http://$VM_IP:9091

# Access custom metrics
open http://$VM_IP:9090/metrics
```

### Create GCP Dashboard

1. Go to [Cloud Console - Monitoring](https://console.cloud.google.com/monitoring)
2. Click "Dashboards" → "Create Dashboard"
3. Add charts for:
   - VM CPU utilization
   - VM memory usage
   - Disk I/O
   - Network traffic
   - Application metrics (from Prometheus)

---

## 🔒 Security Best Practices

### 1. Restrict VM Access

```bash
# Remove external IP (use Cloud IAP for access)
gcloud compute instances delete-access-config $VM_NAME \
    --zone=$ZONE \
    --access-config-name="External NAT"

# Access via IAP tunnel
gcloud compute ssh $VM_NAME \
    --zone=$ZONE \
    --tunnel-through-iap
```

### 2. Enable VPC Firewall

```bash
# Remove allow-all rules
gcloud compute firewall-rules delete allow-streamlit
gcloud compute firewall-rules delete allow-monitoring

# Allow only specific IPs
gcloud compute firewall-rules create allow-streamlit-restricted \
    --allow=tcp:8501 \
    --source-ranges=YOUR_OFFICE_IP/32 \
    --target-tags=petroleum-rag
```

### 3. Enable Audit Logging

```bash
# Enable all audit logs
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member=serviceAccount:YOUR_SERVICE_ACCOUNT \
    --role=roles/logging.admin
```

---

## 💸 Cost Optimization

### 1. Use Preemptible VMs

```bash
# Create preemptible VM (70% cheaper)
gcloud compute instances create $VM_NAME \
    --preemptible \
    --maintenance-policy=TERMINATE \
    # ... other flags
```

### 2. Set Up Auto-Shutdown

```bash
# Create Cloud Scheduler job to stop VM at night
gcloud scheduler jobs create http stop-vm-night \
    --schedule="0 22 * * *" \
    --uri="https://compute.googleapis.com/compute/v1/projects/$PROJECT_ID/zones/$ZONE/instances/$VM_NAME/stop" \
    --http-method=POST \
    --oauth-service-account-email=YOUR_SERVICE_ACCOUNT

# Start in morning
gcloud scheduler jobs create http start-vm-morning \
    --schedule="0 8 * * 1-5" \
    --uri="https://compute.googleapis.com/compute/v1/projects/$PROJECT_ID/zones/$ZONE/instances/$VM_NAME/start" \
    --http-method=POST \
    --oauth-service-account-email=YOUR_SERVICE_ACCOUNT
```

### 3. Use Committed Use Discounts

```bash
# Purchase 1-year commitment for 30-40% savings
gcloud compute commitments create petroleum-rag-commitment \
    --region=$REGION \
    --plan=12-month \
    --resources=vcpu=4,memory=16
```

---

## 🚨 Troubleshooting

### VM Not Accessible

```bash
# Check VM status
gcloud compute instances describe $VM_NAME --zone=$ZONE

# Check firewall rules
gcloud compute firewall-rules list --filter="targetTags:petroleum-rag"

# Check serial console logs
gcloud compute instances get-serial-port-output $VM_NAME --zone=$ZONE
```

### Docker Services Not Running

```bash
gcloud compute ssh $VM_NAME --zone=$ZONE
docker-compose ps
docker-compose logs -f
docker-compose restart
```

### Out of Memory

```bash
# Check memory usage
gcloud compute ssh $VM_NAME --zone=$ZONE
free -h
docker stats

# Increase VM memory
gcloud compute instances stop $VM_NAME --zone=$ZONE
gcloud compute instances set-machine-type $VM_NAME \
    --zone=$ZONE \
    --machine-type=e2-highmem-4
gcloud compute instances start $VM_NAME --zone=$ZONE
```

### API Rate Limits

```bash
# Check rate limiter status
gcloud compute ssh $VM_NAME --zone=$ZONE
cd /opt/petroleum-rag
python3 -c "from utils.rate_limiter import rate_limiter; print(rate_limiter.get_available('openai'))"

# Adjust limits in config.py or .env
```

---

## 📚 Additional Resources

- [GCP Compute Engine Docs](https://cloud.google.com/compute/docs)
- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Memorystore Redis](https://cloud.google.com/memorystore/docs/redis)
- [Secret Manager](https://cloud.google.com/secret-manager/docs)
- [Cloud Monitoring](https://cloud.google.com/monitoring/docs)

---

## ✅ Deployment Checklist

After deployment, verify:

- [ ] VM is running and accessible
- [ ] Docker services are healthy (Chroma, Weaviate, FalkorDB)
- [ ] Secrets are properly configured
- [ ] Documents can be uploaded to GCS bucket
- [ ] Benchmark can run successfully
- [ ] Streamlit UI is accessible
- [ ] Monitoring dashboards show data
- [ ] Backups are configured
- [ ] Alerts are working
- [ ] Costs are within budget

---

## 🎉 You're Done!

Your Petroleum RAG Benchmark is now deployed on GCP!

**Access your application:**
- Web UI: `http://YOUR_VM_IP:8501`
- Monitoring: `http://YOUR_VM_IP:3001`
- API: Contact VM on port 9090

**Next steps:**
1. Upload petroleum documents to GCS bucket
2. Run benchmark via SSH or schedule with Cloud Scheduler
3. Access results in Streamlit UI
4. Set up custom alerts and dashboards
5. Optimize costs based on usage patterns
