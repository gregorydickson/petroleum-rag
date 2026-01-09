# 🚀 One-Command GCP Deployment

Deploy the entire Petroleum RAG Benchmark to Google Cloud Platform with a single command!

## ⚡ Quick Deploy

```bash
./deploy_to_gcp.sh --project YOUR_PROJECT_ID
```

That's it! The script will:
1. ✅ Validate your GCP setup
2. ✅ Enable required APIs
3. ✅ Store your API keys securely
4. ✅ Create Cloud Storage bucket
5. ✅ Create and configure VM
6. ✅ Deploy application
7. ✅ Start all services
8. ✅ Give you access URLs

**Time:** ~10-15 minutes
**Cost:** ~$128/month (~$0.17/hour)

---

## 📋 Prerequisites

### 1. GCP Account
- Active GCP account
- Billing enabled
- Project created (or script can create one)

### 2. gcloud CLI Installed

**macOS:**
```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

**Linux:**
```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

**Windows:**
Download from: https://cloud.google.com/sdk/docs/install

### 3. Authenticate

```bash
gcloud auth login
```

### 4. API Keys Ready

You'll need:
- ✅ **Anthropic API key** (sk-ant-...)
- ✅ **OpenAI API key** (sk-...)
- ✅ **LlamaParse API key** (llx-...)

Get them from:
- Anthropic: https://console.anthropic.com/
- OpenAI: https://platform.openai.com/
- LlamaParse: https://cloud.llamaindex.ai/

---

## 🎯 Deployment Options

### Option 1: Simple Deploy (Recommended)

```bash
./deploy_to_gcp.sh --project my-gcp-project
```

Creates:
- Single VM with all services
- Cloud Storage for documents
- Secret Manager for API keys
- Firewall rules for access

**Best for:** POC, testing, low traffic
**Cost:** ~$128/month

### Option 2: Production Deploy

```bash
./deploy_to_gcp.sh --project my-gcp-project --production --vm-type e2-standard-8
```

Creates:
- Larger VM with more resources
- Production-optimized configuration
- Enhanced monitoring

**Best for:** Production, high traffic
**Cost:** ~$240/month

### Option 3: Custom Configuration

```bash
./deploy_to_gcp.sh \
  --project my-gcp-project \
  --region us-west1 \
  --zone us-west1-a \
  --vm-name my-custom-vm \
  --vm-type e2-highmem-4
```

---

## 📊 What Gets Deployed

```
Google Cloud Platform
├── Compute Engine VM (e2-standard-4)
│   ├── Ubuntu 22.04 LTS
│   ├── Docker + Docker Compose
│   ├── Python 3.11
│   ├── Petroleum RAG Application
│   └── Docker Services
│       ├── Chroma (Vector DB)
│       ├── Weaviate (Hybrid Search)
│       └── FalkorDB (Graph DB)
├── Cloud Storage Bucket
│   ├── /input/  (Your documents)
│   ├── /results/ (Benchmark results)
│   └── /cache/  (Embeddings cache)
├── Secret Manager
│   ├── anthropic-api-key
│   ├── openai-api-key
│   └── llama-cloud-api-key
└── Firewall Rules
    ├── allow-streamlit (port 8501)
    └── allow-monitoring (ports 9090, 3001, 9091)
```

---

## 🖥️ After Deployment

### Access Your Application

Once deployment completes, you'll see:

```
╔═══════════════════════════════════════════════════════════════╗
║                  Deployment Successful! 🎉                    ║
╚═══════════════════════════════════════════════════════════════╝

Access Information:
─────────────────────────────────────────────────────────────

  VM Instance:     petroleum-rag-vm
  External IP:     34.123.45.67
  GCP Project:     my-gcp-project
  Region/Zone:     us-central1 / us-central1-a

Application URLs:
─────────────────────────────────────────────────────────────

  Streamlit UI:    http://34.123.45.67:8501
  Monitoring:      http://34.123.45.67:9090
  Grafana:         http://34.123.45.67:3001  (admin/admin)
  Prometheus:      http://34.123.45.67:9091
```

### Run Your First Benchmark

1. **SSH to VM:**
   ```bash
   gcloud compute ssh petroleum-rag-vm --zone=us-central1-a
   ```

2. **Navigate to app:**
   ```bash
   cd /opt/petroleum-rag
   ```

3. **Start the application:**
   ```bash
   ./start_app.sh
   ```

4. **Open UI in browser:**
   - Visit: `http://YOUR_VM_IP:8501`
   - Wait for processing to complete (~45-60 min for 11MB doc)
   - View results in the dashboard!

---

## 💰 Cost Management

### Monthly Costs Breakdown

| Service | Cost | Can Stop? |
|---------|------|-----------|
| VM (e2-standard-4) | ~$120 | ✅ Yes |
| Cloud Storage (100GB) | ~$2 | ❌ No |
| Secret Manager | ~$1 | ❌ No |
| Logging | ~$5 | ⚠️ Minimal |
| **Total** | **~$128** | |

### Save Money: Stop VM When Not in Use

```bash
# Stop VM (saves ~$120/month)
gcloud compute instances stop petroleum-rag-vm --zone=us-central1-a

# Start VM when needed
gcloud compute instances start petroleum-rag-vm --zone=us-central1-a

# Get new IP address (if it changed)
gcloud compute instances describe petroleum-rag-vm \
  --zone=us-central1-a \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
```

**💡 Pro Tip:** Use a scheduled Cloud Scheduler job to auto-stop VM at night!

### Set Up Auto-Shutdown (Save 70% on VM costs)

```bash
# Stop VM every night at 10 PM
gcloud scheduler jobs create http stop-vm-night \
  --schedule="0 22 * * *" \
  --uri="https://compute.googleapis.com/compute/v1/projects/YOUR_PROJECT/zones/us-central1-a/instances/petroleum-rag-vm/stop" \
  --http-method=POST \
  --oauth-service-account-email=YOUR_SERVICE_ACCOUNT

# Start VM every weekday at 8 AM
gcloud scheduler jobs create http start-vm-morning \
  --schedule="0 8 * * 1-5" \
  --uri="https://compute.googleapis.com/compute/v1/projects/YOUR_PROJECT/zones/us-central1-a/instances/petroleum-rag-vm/start" \
  --http-method=POST \
  --oauth-service-account-email=YOUR_SERVICE_ACCOUNT
```

---

## 🔧 Management Commands

### View Application Logs

```bash
gcloud compute ssh petroleum-rag-vm --zone=us-central1-a
cd /opt/petroleum-rag
tail -f logs/*.log
```

### Check Docker Services

```bash
gcloud compute ssh petroleum-rag-vm --zone=us-central1-a
docker-compose ps
docker-compose logs -f
```

### Upload Documents

```bash
# From your local machine
gsutil cp your-document.pdf gs://YOUR_PROJECT-petroleum-rag/input/

# Then SSH to VM and sync
gcloud compute ssh petroleum-rag-vm --zone=us-central1-a
cd /opt/petroleum-rag
gsutil -m cp gs://YOUR_PROJECT-petroleum-rag/input/* data/input/
```

### View Cache Statistics

```bash
gcloud compute ssh petroleum-rag-vm --zone=us-central1-a
cd /opt/petroleum-rag
source venv/bin/activate
python scripts/manage_cache.py stats
```

### Restart Application

```bash
gcloud compute ssh petroleum-rag-vm --zone=us-central1-a
cd /opt/petroleum-rag
docker-compose restart
```

---

## 🔍 Troubleshooting

### "VM not accessible"

```bash
# Check VM status
gcloud compute instances describe petroleum-rag-vm --zone=us-central1-a

# Check firewall rules
gcloud compute firewall-rules list --filter="targetTags:petroleum-rag"

# View serial console logs
gcloud compute instances get-serial-port-output petroleum-rag-vm --zone=us-central1-a
```

### "Services not starting"

```bash
# SSH to VM
gcloud compute ssh petroleum-rag-vm --zone=us-central1-a

# Check Docker
docker ps
docker-compose logs

# Restart Docker services
cd /opt/petroleum-rag
docker-compose down
docker-compose up -d
```

### "Can't access Streamlit UI"

1. Check firewall rules exist:
   ```bash
   gcloud compute firewall-rules describe allow-streamlit
   ```

2. Verify VM has external IP:
   ```bash
   gcloud compute instances describe petroleum-rag-vm \
     --zone=us-central1-a \
     --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
   ```

3. Check Streamlit is running:
   ```bash
   gcloud compute ssh petroleum-rag-vm --zone=us-central1-a
   ps aux | grep streamlit
   ```

---

## 🔒 Security Tips

### 1. Restrict Access by IP

```bash
# Only allow your office IP
gcloud compute firewall-rules update allow-streamlit \
  --source-ranges=YOUR_OFFICE_IP/32
```

### 2. Use VPN or Cloud IAP

```bash
# Access via Cloud Identity-Aware Proxy
gcloud compute start-iap-tunnel petroleum-rag-vm 8501 \
  --local-host-port=localhost:8501 \
  --zone=us-central1-a

# Then access: http://localhost:8501
```

### 3. Enable OS Login

```bash
gcloud compute instances add-metadata petroleum-rag-vm \
  --zone=us-central1-a \
  --metadata enable-oslogin=TRUE
```

---

## 📈 Scaling Up

### Increase VM Size

```bash
# Stop VM
gcloud compute instances stop petroleum-rag-vm --zone=us-central1-a

# Resize
gcloud compute instances set-machine-type petroleum-rag-vm \
  --zone=us-central1-a \
  --machine-type=e2-standard-8

# Start VM
gcloud compute instances start petroleum-rag-vm --zone=us-central1-a
```

### Add More Storage

```bash
# Create and attach disk
gcloud compute disks create petroleum-rag-data \
  --size=500GB \
  --zone=us-central1-a

gcloud compute instances attach-disk petroleum-rag-vm \
  --disk=petroleum-rag-data \
  --zone=us-central1-a
```

---

## 🗑️ Clean Up / Delete Everything

### Delete Entire Deployment

```bash
# Delete VM
gcloud compute instances delete petroleum-rag-vm --zone=us-central1-a --quiet

# Delete firewall rules
gcloud compute firewall-rules delete allow-streamlit --quiet
gcloud compute firewall-rules delete allow-monitoring --quiet

# Delete Cloud Storage bucket
gsutil -m rm -r gs://YOUR_PROJECT-petroleum-rag

# Delete secrets
gcloud secrets delete anthropic-api-key --quiet
gcloud secrets delete openai-api-key --quiet
gcloud secrets delete llama-cloud-api-key --quiet

# Delete project (if you want to remove everything)
# gcloud projects delete YOUR_PROJECT_ID
```

**⚠️ Warning:** This permanently deletes all data!

---

## 📚 Additional Resources

- **Full Deployment Guide:** `docs/GCP_DEPLOYMENT_GUIDE.md`
- **User Guide:** `docs/USER_GUIDE.md`
- **API Reference:** `docs/API_REFERENCE.md`
- **Architecture:** `docs/ARCHITECTURE.md`

---

## 🆘 Need Help?

1. **Check the logs:**
   ```bash
   gcloud compute ssh petroleum-rag-vm --zone=us-central1-a
   cd /opt/petroleum-rag
   tail -f logs/*.log
   ```

2. **View deployment info:**
   ```bash
   cat deployment-info.txt
   ```

3. **Consult documentation:**
   - Local: `docs/GCP_DEPLOYMENT_GUIDE.md`
   - GCP: https://cloud.google.com/docs

---

## ✅ Quick Checklist

Before deploying:
- [ ] GCP account with billing enabled
- [ ] gcloud CLI installed and authenticated
- [ ] API keys ready (Anthropic, OpenAI, LlamaParse)
- [ ] Project ID chosen or created

After deploying:
- [ ] VM is running
- [ ] Can access Streamlit UI
- [ ] Docker services are healthy
- [ ] Documents uploaded to Cloud Storage
- [ ] Benchmark completed successfully
- [ ] Monitoring dashboards accessible

---

**Ready to deploy? Run:**

```bash
./deploy_to_gcp.sh --project YOUR_PROJECT_ID
```

🎉 You'll have a production RAG system in ~15 minutes!
