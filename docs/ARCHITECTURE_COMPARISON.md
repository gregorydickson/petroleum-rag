# 🏗️ Architecture Comparison: VM vs Serverless

This document compares the two GCP deployment approaches for the Petroleum RAG Benchmark.

---

## TL;DR - Which Should I Use?

| Factor | VM Approach | Serverless Approach | Winner |
|--------|-------------|---------------------|--------|
| **Cost** | $128/month | $55/month base + usage | 🏆 **Serverless (57% cheaper)** |
| **SSH Required** | ✅ Yes | ❌ No | 🏆 **Serverless** |
| **Auto-Start** | ❌ Manual | ✅ Auto on upload | 🏆 **Serverless** |
| **Maintenance** | ⚠️ Docker + VM | ✅ Fully managed | 🏆 **Serverless** |
| **Setup Time** | 15 min | 20 min | 🏆 **VM (slightly faster)** |
| **Cold Start** | ❌ None | ⚠️ 10-15 sec | 🏆 **VM** |
| **Simplicity** | ⚠️ Manual steps | ✅ Automated | 🏆 **Serverless** |

**Recommendation:** Use **Serverless** unless you need zero cold starts or prefer direct VM access.

---

## Architecture Diagrams

### VM Approach (deploy_to_gcp.sh)

```
┌───────────────────────────────────────────────────────────────┐
│                    Google Cloud Platform                       │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Compute Engine VM (e2-standard-4)                       │ │
│  │  ┌────────────────────────────────────────────────────┐  │ │
│  │  │  Ubuntu 22.04 LTS                                  │  │ │
│  │  │  ┌──────────────────────────────────────────────┐  │  │ │
│  │  │  │  Docker Compose Stack                        │  │  │ │
│  │  │  │  ┌────────────┬────────────┬──────────────┐  │  │  │ │
│  │  │  │  │  Chroma    │  Weaviate  │  FalkorDB    │  │  │  │ │
│  │  │  │  │  (Vector)  │  (Hybrid)  │  (Graph)     │  │  │  │ │
│  │  │  │  └────────────┴────────────┴──────────────┘  │  │  │ │
│  │  │  │  ┌──────────────────────────────────────┐    │  │  │ │
│  │  │  │  │  Streamlit UI (port 8501)             │    │  │  │ │
│  │  │  │  └──────────────────────────────────────┘    │  │  │ │
│  │  │  │  ┌──────────────────────────────────────┐    │  │  │ │
│  │  │  │  │  Monitoring (Prometheus + Grafana)    │    │  │  │ │
│  │  │  │  └──────────────────────────────────────┘    │  │  │ │
│  │  │  └──────────────────────────────────────────────┘  │  │ │
│  │  │  ┌──────────────────────────────────────────────┐  │  │ │
│  │  │  │  Application (Python)                        │  │  │ │
│  │  │  │  • benchmark.py                              │  │  │ │
│  │  │  │  • analyze_results.py                        │  │  │ │
│  │  │  │  • demo_app.py                               │  │  │ │
│  │  │  └──────────────────────────────────────────────┘  │  │ │
│  │  └────────────────────────────────────────────────────┘  │ │
│  │                                                           │ │
│  │  External IP: 34.123.45.67 (always on)                   │ │
│  └──────────────────────────────────────────────────────────┘ │
│                              │                                 │
│                              │ gsutil sync                     │
│                              ▼                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Cloud Storage Bucket                                    │ │
│  │  • /input/  (documents)                                  │ │
│  │  • /results/ (benchmark results)                         │ │
│  │  • /cache/  (embeddings)                                 │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Secret Manager                                          │ │
│  │  • anthropic-api-key                                     │ │
│  │  • openai-api-key                                        │ │
│  │  • llama-cloud-api-key                                   │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
└───────────────────────────────────────────────────────────────┘

User Workflow:
1. SSH to VM: gcloud compute ssh petroleum-rag-vm
2. Sync documents: gsutil -m cp gs://bucket/input/* data/input/
3. Run processing: ./start_app.sh
4. Access UI: http://34.123.45.67:8501
```

**Key Characteristics:**
- ✅ Everything runs on one VM
- ✅ Fast (no cold starts)
- ✅ Simple architecture
- ❌ Requires SSH for management
- ❌ VM runs 24/7 (costs $120/month)
- ❌ Manual document sync and processing
- ❌ Must manage Docker containers

---

### Serverless Approach (deploy_cloudrun_serverless.sh)

```
┌────────────────────────────────────────────────────────────────┐
│                    Google Cloud Platform                        │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Cloud Run Service (petroleum-rag-ui)                     │ │
│  │  ┌─────────────────────────────────────────────────────┐  │ │
│  │  │  Container: Streamlit UI                            │  │ │
│  │  │  • Auto-scaling: 0-10 instances                     │  │ │
│  │  │  • Min instances: 0 (scales to zero!)               │  │ │
│  │  │  • Port: 8080 (HTTPS endpoint)                      │  │ │
│  │  └─────────────────────────────────────────────────────┘  │ │
│  │  URL: https://petroleum-rag-ui-xyz.a.run.app             │ │
│  └───────────────────────────────────────────────────────────┘ │
│                              ▲                                  │
│                              │ Connects to                      │
│                              │                                  │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Memorystore Redis (petroleum-rag-redis)                  │ │
│  │  • Managed Redis 7.0                                      │ │
│  │  • 5GB memory                                             │ │
│  │  • FalkorDB graph storage                                 │ │
│  │  • Auth enabled                                           │ │
│  └───────────────────────────────────────────────────────────┘ │
│                              ▲                                  │
│                              │ Also connects to                 │
│                              │                                  │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Cloud Run Job (petroleum-rag-processor)                  │ │
│  │  ┌─────────────────────────────────────────────────────┐  │ │
│  │  │  Container: Processing Job                          │  │ │
│  │  │  • Runs benchmark.py                                │  │ │
│  │  │  • Memory: 8Gi, CPU: 4                              │  │ │
│  │  │  • Timeout: 3600s (1 hour)                          │  │ │
│  │  │  • Triggered on-demand (not always running)         │  │ │
│  │  └─────────────────────────────────────────────────────┘  │ │
│  └───────────────────────────────────────────────────────────┘ │
│                              ▲                                  │
│                              │ Triggered by                     │
│                              │                                  │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Cloud Functions (petroleum-rag-trigger)                  │ │
│  │  • Gen 2 (Cloud Run-based)                                │ │
│  │  • Listens for GCS object.finalized events                │ │
│  │  • Validates PDF in input/ directory                      │ │
│  │  • Triggers Cloud Run Job via API                         │ │
│  └───────────────────────────────────────────────────────────┘ │
│                              ▲                                  │
│                              │ Eventarc trigger                 │
│                              │                                  │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Cloud Storage Bucket                                     │ │
│  │  gs://PROJECT-petroleum-rag/                              │ │
│  │  ┌─────────────────────────────────────────────────────┐  │ │
│  │  │  /input/     ← Upload documents here                │  │ │
│  │  │  /results/   ← Benchmark results                    │  │ │
│  │  │  /cache/     ← Cached embeddings                    │  │ │
│  │  └─────────────────────────────────────────────────────┘  │ │
│  │  • Versioning enabled                                     │ │
│  │  • Lifecycle rules for old versions                       │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Secret Manager                                           │ │
│  │  • anthropic-api-key                                      │ │
│  │  • openai-api-key                                         │ │
│  │  • llama-cloud-api-key                                    │ │
│  │  (Accessed by Cloud Run services via env vars)            │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Optional: Cloud Scheduler                                │ │
│  │  • Daily trigger at 2 AM                                  │ │
│  │  • Runs Cloud Run Job via HTTP POST                       │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
└────────────────────────────────────────────────────────────────┘

User Workflow:
1. Upload: gsutil cp doc.pdf gs://bucket/input/
2. Auto-processing: (Cloud Function → Cloud Run Job → Results)
3. View: https://petroleum-rag-ui-xyz.a.run.app
```

**Key Characteristics:**
- ✅ Fully serverless (no VMs)
- ✅ Auto-triggers on document upload
- ✅ Scales to zero when idle
- ✅ No SSH required ever
- ✅ Fully managed services
- ✅ 57% cost savings
- ⚠️ Cold start delay (10-15 sec)
- ⚠️ Slightly more complex architecture

---

## Detailed Comparison

### 1. Cost Analysis

#### VM Approach
| Component | Monthly Cost | Notes |
|-----------|--------------|-------|
| VM (e2-standard-4) | $120 | Runs 24/7 |
| Cloud Storage | $2 | 100GB |
| Secret Manager | $1 | 3 secrets |
| Networking | $5 | Egress |
| **Total** | **$128** | Fixed cost |

**Cost per run:** Included in monthly VM cost (already paying for it)

#### Serverless Approach
| Component | Monthly Cost | Notes |
|-----------|--------------|-------|
| Memorystore Redis | $50 | Always on |
| Cloud Storage | $2 | 100GB |
| Secret Manager | $1 | 3 secrets |
| Cloud Run UI (idle) | $1 | Scales to zero |
| Cloud Functions | $0.10 | Per trigger |
| **Base Total** | **$55** | Fixed cost |
| **Per run** | **$3** | Cloud Run job + APIs |

**Break-even:** If you process more than 24 times per month, VM becomes cheaper on a per-run basis. But serverless still saves $73/month on base costs.

### 2. Management Overhead

#### VM Approach
```bash
# Upload documents
gsutil cp doc.pdf gs://bucket/input/

# SSH to VM
gcloud compute ssh petroleum-rag-vm --zone=us-central1-a

# Sync documents
cd /opt/petroleum-rag
gsutil -m cp gs://bucket/input/* data/input/

# Run processing
./start_app.sh

# Check Docker status
docker-compose ps

# Restart services if needed
docker-compose restart

# Exit SSH
exit

# Access UI (need external IP)
# http://34.123.45.67:8501
```

**Steps required:** 7-8 manual steps
**Time:** 5-10 minutes
**SSH required:** Yes

#### Serverless Approach
```bash
# Upload documents (auto-processes)
gsutil cp doc.pdf gs://bucket/input/

# That's it! Processing starts automatically

# View UI (always accessible)
# https://petroleum-rag-ui-xyz.a.run.app
```

**Steps required:** 1 step
**Time:** 30 seconds
**SSH required:** No

### 3. Scaling Behavior

#### VM Approach
- **UI:** Can only handle concurrent requests based on VM resources
- **Processing:** One benchmark at a time (or manual parallelization)
- **Storage:** Docker containers on VM disk
- **Scaling strategy:** Vertical (bigger VM) or manual horizontal (more VMs)

#### Serverless Approach
- **UI:** Auto-scales 0-10 instances based on traffic
- **Processing:** Can run multiple jobs in parallel automatically
- **Storage:** Managed services scale automatically
- **Scaling strategy:** Automatic horizontal scaling

### 4. Cold Start Comparison

#### VM Approach
- **First access:** Instant (VM already running)
- **After restart:** 30-60 seconds (Docker containers start)
- **Subsequent requests:** Instant

#### Serverless Approach
- **First access (UI):** 10-15 seconds (cold start)
- **Subsequent requests:** Instant (warm container)
- **Processing job:** 15-20 seconds (container startup)
- **Keep warm option:** Set min-instances=1 (costs ~$30/month more)

### 5. Operational Complexity

#### VM Approach
```
Complexity: MEDIUM

Must manage:
- VM lifecycle (start/stop/restart)
- Docker containers (up/down/logs)
- SSH keys and access
- Firewall rules
- OS updates
- Docker image updates
- Container health checks
- Log rotation
```

#### Serverless Approach
```
Complexity: LOW

Google manages:
- Container orchestration
- Auto-scaling
- Health checks
- OS updates
- Network configuration
- Load balancing

You only manage:
- Container images (rebuild on code changes)
- Environment variables
- Secrets
```

### 6. Development Workflow

#### VM Approach
```bash
# 1. Make code changes locally
vim src/storage/weaviate_store.py

# 2. Commit and push
git commit -am "Fix Weaviate query"
git push

# 3. SSH to VM
gcloud compute ssh petroleum-rag-vm --zone=us-central1-a

# 4. Pull changes
cd /opt/petroleum-rag
git pull

# 5. Restart services
docker-compose restart

# 6. Test
./start_app.sh
```

#### Serverless Approach
```bash
# 1. Make code changes locally
vim src/storage/weaviate_store.py

# 2. Rebuild and redeploy
gcloud builds submit --tag gcr.io/PROJECT/petroleum-rag-processor .

gcloud run jobs update petroleum-rag-processor \
  --image gcr.io/PROJECT/petroleum-rag-processor \
  --region us-central1

# 3. Test
gcloud run jobs execute petroleum-rag-processor --region us-central1
```

**Note:** Serverless has slightly more overhead for code updates, but no SSH required.

### 7. Monitoring and Debugging

#### VM Approach
```bash
# SSH to VM
gcloud compute ssh petroleum-rag-vm --zone=us-central1-a

# View application logs
tail -f /opt/petroleum-rag/logs/*.log

# Check Docker logs
docker-compose logs -f

# Resource usage
htop

# Network connections
netstat -tulpn

# Disk usage
df -h
```

**Pros:** Full system access, can debug anything
**Cons:** Must SSH, manual log collection

#### Serverless Approach
```bash
# View all logs (no SSH)
gcloud logging read 'resource.type=cloud_run_job' --limit 50

# Follow logs in real-time
gcloud logging tail 'resource.type=cloud_run_job'

# View metrics
gcloud monitoring time-series list \
  --filter='resource.type="cloud_run_revision"'

# Structured log queries
gcloud logging read 'jsonPayload.message=~"ERROR"' --limit 100
```

**Pros:** Centralized logging, no SSH, structured logs
**Cons:** Less direct system access (but rarely needed)

---

## Migration Path

### From VM to Serverless

If you're currently using the VM approach and want to switch:

```bash
# 1. Deploy serverless alongside VM
./deploy_cloudrun_serverless.sh --project YOUR_PROJECT_ID

# 2. Test with sample document
gsutil cp test-doc.pdf gs://YOUR_PROJECT-petroleum-rag/input/

# 3. Verify results in serverless UI
open https://petroleum-rag-ui-xyz.a.run.app

# 4. If satisfied, stop VM
gcloud compute instances stop petroleum-rag-vm --zone=us-central1-a

# 5. (Optional) Delete VM after testing period
gcloud compute instances delete petroleum-rag-vm --zone=us-central1-a
```

### From Serverless to VM

If you need VM for some reason:

```bash
# 1. Deploy VM
./deploy_to_gcp.sh --project YOUR_PROJECT_ID

# 2. Sync data from Cloud Storage
gcloud compute ssh petroleum-rag-vm --zone=us-central1-a
gsutil -m cp -r gs://YOUR_PROJECT-petroleum-rag/* /opt/petroleum-rag/data/

# 3. Test
./start_app.sh

# 4. If satisfied, delete serverless resources
# (see GCP_SERVERLESS_README.md cleanup section)
```

---

## Use Case Recommendations

### Choose VM Approach When:
- ✅ You need zero cold starts (instant response always)
- ✅ You prefer SSH access for debugging
- ✅ You're comfortable managing Docker and VMs
- ✅ You process documents very frequently (20+ times/month)
- ✅ You need custom OS-level configurations
- ✅ You want full control over the runtime environment

### Choose Serverless Approach When:
- ✅ You want minimal operational overhead
- ✅ You don't want to manage infrastructure
- ✅ You're okay with 10-15 second cold starts
- ✅ You want automatic scaling
- ✅ You process documents infrequently (<20 times/month)
- ✅ You want the lowest possible cost
- ✅ You prefer event-driven architecture

---

## Summary Table

| Feature | VM Approach | Serverless Approach |
|---------|-------------|---------------------|
| **Deployment Time** | 15 min | 20 min |
| **Monthly Base Cost** | $128 | $55 |
| **Cost per Run** | Included | $3 |
| **Auto-Start** | ❌ No | ✅ Yes |
| **SSH Required** | ✅ Yes | ❌ No |
| **Cold Start** | None | 10-15 sec |
| **Auto-Scaling** | ❌ No | ✅ Yes |
| **Maintenance** | Docker + VM | Minimal |
| **Complexity** | Medium | Low |
| **Best For** | Frequent use, full control | Infrequent use, low maintenance |

---

## Conclusion

**For most users, the Serverless approach is recommended** because:
1. **57% cost savings** on base infrastructure
2. **No SSH or VM management** required
3. **Auto-starts on document upload** (event-driven)
4. **Scales to zero** when idle
5. **Fully managed** by Google

The VM approach is better only if you:
- Need absolute zero cold starts
- Prefer direct system access via SSH
- Process documents very frequently (making per-run costs irrelevant)

**Both approaches use the same application code** - you can switch between them at any time!
