# 🚀 Serverless GCP Deployment - No SSH Required!

Deploy the Petroleum RAG Benchmark to Google Cloud Platform as a **fully serverless, auto-starting** application!

## ⚡ One-Command Deploy

```bash
./deploy_cloudrun_serverless.sh --project YOUR_PROJECT_ID
```

**That's it!** No VMs to manage, no SSH required, auto-starts when you upload documents!

**Time:** ~15-20 minutes
**Cost:** ~$55/month base + usage (**57% cheaper than VM approach!**)

---

## 🎯 Why Serverless?

### Problems with VM Approach
- ❌ Requires SSH to manage
- ❌ VM runs 24/7 (costs ~$128/month)
- ❌ Manual restart after document upload
- ❌ Must manage Docker containers
- ❌ Requires VM maintenance

### Serverless Benefits
- ✅ **No SSH required ever**
- ✅ **Auto-starts on document upload**
- ✅ **Scales to zero when idle** (save money!)
- ✅ **Fully managed** (no Docker/VM to maintain)
- ✅ **Event-driven** (upload → process → results)
- ✅ **57% cost savings** ($55/month vs $128/month)

---

## 📋 Prerequisites

### 1. GCP Account
- Active GCP account with billing enabled
- Project created (or script can prompt you)

### 2. gcloud CLI Installed

**macOS/Linux:**
```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

**Authenticate:**
```bash
gcloud auth login
```

### 3. API Keys Ready
- ✅ **Anthropic API key** (sk-ant-...)
- ✅ **OpenAI API key** (sk-...)
- ✅ **LlamaParse API key** (llx-...)

---

## 🏗️ What Gets Deployed

```
Google Cloud Platform (Serverless)
├── Cloud Run Service (Streamlit UI)
│   ├── Auto-scaling (0-10 instances)
│   ├── Scales to zero when idle
│   └── Always accessible via HTTPS URL
│
├── Cloud Run Job (Processing)
│   ├── Triggered automatically on upload
│   ├── Or triggered manually
│   └── Runs only when needed
│
├── Cloud Functions (Auto-Trigger)
│   ├── Detects document uploads
│   └── Triggers processing job
│
├── Memorystore Redis (FalkorDB)
│   ├── Managed Redis instance
│   └── Graph database storage
│
├── Cloud Storage Bucket
│   ├── /input/  (Upload documents here)
│   ├── /results/ (Benchmark results)
│   └── /cache/  (Embeddings cache)
│
├── Secret Manager
│   ├── anthropic-api-key
│   ├── openai-api-key
│   └── llama-cloud-api-key
│
└── Eventarc (Event Routing)
    └── GCS upload → Cloud Function → Cloud Run Job
```

---

## 🚀 Deployment Steps

### Step 1: Deploy

```bash
./deploy_cloudrun_serverless.sh --project my-gcp-project
```

The script will:
1. ✅ Enable required APIs (Cloud Run, Cloud Functions, Eventarc, etc.)
2. ✅ Store your API keys in Secret Manager
3. ✅ Create Cloud Storage bucket
4. ✅ Create Memorystore Redis instance (~5 min)
5. ✅ Build container images for UI and processing
6. ✅ Deploy Cloud Run services
7. ✅ Set up auto-trigger function
8. ✅ Provide access URLs

### Step 2: Upload a Document

**Upload triggers processing automatically:**

```bash
# Upload document to input directory
gsutil cp your-document.pdf gs://YOUR_PROJECT-petroleum-rag/input/

# That's it! Processing starts automatically
```

### Step 3: Access the UI

```bash
# Open the URL provided after deployment
# Example: https://petroleum-rag-ui-abc123-uc.a.run.app
```

---

## 🔄 How It Works (Event-Driven Architecture)

```
┌─────────────────────────────────────────────────────────────┐
│  1. Upload Document                                          │
│     gsutil cp doc.pdf gs://bucket/input/                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Cloud Function Detects Upload (Eventarc)                │
│     • Triggered by GCS object finalized event                │
│     • Validates file is PDF in input/ directory              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Cloud Run Job Triggered                                  │
│     • Container spins up on-demand                           │
│     • Runs benchmark.py automatically                        │
│     • 4 parsers × 3 storage = 12 combinations                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Results Saved to Cloud Storage                          │
│     • Benchmark results → gs://bucket/results/               │
│     • Cached embeddings → gs://bucket/cache/                 │
│     • Job completes, container shuts down                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  5. View Results in UI                                       │
│     • Access Cloud Run UI URL                                │
│     • See benchmark results                                  │
│     • Ask questions in chat                                  │
└─────────────────────────────────────────────────────────────┘
```

**No SSH required at any step!**

---

## 💰 Cost Breakdown

### Monthly Costs (Serverless)

| Service | Always On? | Monthly Cost |
|---------|-----------|--------------|
| Memorystore Redis (5GB) | ✅ Yes | ~$50 |
| Cloud Storage (100GB) | ✅ Yes | ~$2 |
| Secret Manager | ✅ Yes | ~$1 |
| Cloud Run UI (idle most of time) | ⚠️ Scales to 0 | ~$1 |
| Cloud Run Job (per run) | ❌ No | ~$0.50/run |
| Cloud Functions (triggers) | ❌ No | ~$0.10/month |
| Processing APIs (LLM, embeddings) | ❌ No | ~$2.50/run |
| **Total Base Cost** | | **~$55/month** |
| **Per Processing Run** | | **~$3/run** |

### Compare to VM Approach

| Metric | VM Approach | Serverless | Savings |
|--------|-------------|------------|---------|
| Base monthly cost | $128 | $55 | **57%** |
| Requires SSH | ✅ Yes | ❌ No | ✅ |
| Manual start | ✅ Yes | ❌ Auto | ✅ |
| Scales to zero | ❌ No | ✅ Yes | ✅ |
| Maintenance | ⚠️ Docker/VM | ✅ None | ✅ |

**Serverless wins on cost AND convenience!**

---

## 📊 Usage Examples

### Auto-Processing Workflow

```bash
# 1. Upload document (triggers processing automatically)
gsutil cp handbook.pdf gs://my-project-petroleum-rag/input/

# 2. Check processing logs
gcloud logging read 'resource.type=cloud_run_job' --limit 50

# 3. View UI when ready
open https://petroleum-rag-ui-abc123-uc.a.run.app
```

### Manual Processing Trigger

```bash
# Trigger processing manually (without uploading document)
gcloud run jobs execute petroleum-rag-processor --region=us-central1

# View job execution history
gcloud run jobs executions list --region=us-central1
```

### Upload Multiple Documents

```bash
# Upload all PDFs from local directory
gsutil -m cp data/input/*.pdf gs://my-project-petroleum-rag/input/

# Each file triggers a separate processing job automatically
```

---

## 🔧 Management Commands

### View Processing Logs

```bash
# View recent processing logs
gcloud logging read 'resource.type=cloud_run_job' --limit 50 --format json

# Follow logs in real-time
gcloud logging tail 'resource.type=cloud_run_job'
```

### View Job Executions

```bash
# List all job executions
gcloud run jobs executions list \
  --job=petroleum-rag-processor \
  --region=us-central1

# Get details of specific execution
gcloud run jobs executions describe EXECUTION_ID \
  --region=us-central1
```

### Access UI Service

```bash
# Get UI URL
gcloud run services describe petroleum-rag-ui \
  --region=us-central1 \
  --format='value(status.url)'

# View UI logs
gcloud run services logs read petroleum-rag-ui \
  --region=us-central1 \
  --limit=50
```

### Manage Cloud Storage

```bash
# List documents
gsutil ls gs://my-project-petroleum-rag/input/

# Download results
gsutil -m cp -r gs://my-project-petroleum-rag/results/ ./local-results/

# View cache statistics
gsutil du -sh gs://my-project-petroleum-rag/cache/
```

### Check Redis Instance

```bash
# Get Redis connection info
gcloud redis instances describe petroleum-rag-redis \
  --region=us-central1

# View Redis metrics
gcloud monitoring time-series list \
  --filter='resource.type="redis_instance"'
```

---

## ⚙️ Optional: Scheduled Processing

Set up daily automatic processing:

```bash
# The deployment script asks if you want this
# Or set up manually:

# Schedule daily processing at 2 AM
gcloud scheduler jobs create http petroleum-rag-daily \
  --location=us-central1 \
  --schedule="0 2 * * *" \
  --uri="https://us-central1-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/PROJECT_ID/jobs/petroleum-rag-processor:run" \
  --http-method=POST \
  --oauth-service-account-email=cloud-run-invoker@PROJECT_ID.iam.gserviceaccount.com
```

---

## 🔍 Troubleshooting

### Processing Not Starting After Upload

**Check Cloud Function logs:**
```bash
gcloud functions logs read petroleum-rag-trigger \
  --region=us-central1 \
  --limit=50
```

**Verify Eventarc trigger:**
```bash
gcloud eventarc triggers list --location=us-central1
```

**Manual trigger to test:**
```bash
gcloud run jobs execute petroleum-rag-processor --region=us-central1
```

### UI Not Accessible

**Check service status:**
```bash
gcloud run services describe petroleum-rag-ui --region=us-central1
```

**View logs:**
```bash
gcloud run services logs read petroleum-rag-ui --region=us-central1
```

**Redeploy if needed:**
```bash
gcloud run services update petroleum-rag-ui \
  --region=us-central1 \
  --image=gcr.io/PROJECT_ID/petroleum-rag-ui
```

### Redis Connection Issues

**Check Redis status:**
```bash
gcloud redis instances describe petroleum-rag-redis --region=us-central1
```

**Verify network connectivity:**
- Cloud Run services must be in same VPC as Redis
- Check VPC connector is attached to Cloud Run services

### Secret Access Issues

**Verify secrets exist:**
```bash
gcloud secrets list
```

**Check IAM permissions:**
```bash
gcloud run services get-iam-policy petroleum-rag-ui --region=us-central1
```

**Grant secret access if needed:**
```bash
gcloud secrets add-iam-policy-binding anthropic-api-key \
  --member="serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

---

## 📈 Scaling and Performance

### UI Auto-Scaling

```bash
# Update max instances (default: 10)
gcloud run services update petroleum-rag-ui \
  --region=us-central1 \
  --max-instances=20

# Set minimum instances (avoid cold starts)
gcloud run services update petroleum-rag-ui \
  --region=us-central1 \
  --min-instances=1  # Costs more but no cold start
```

### Processing Job Resources

```bash
# Increase memory/CPU for large documents
gcloud run jobs update petroleum-rag-processor \
  --region=us-central1 \
  --memory=16Gi \
  --cpu=8
```

### Redis Scaling

```bash
# Increase Redis memory
gcloud redis instances update petroleum-rag-redis \
  --region=us-central1 \
  --size=10  # GB
```

---

## 🗑️ Clean Up / Delete Everything

```bash
# Delete Cloud Run services and jobs
gcloud run services delete petroleum-rag-ui --region=us-central1 --quiet
gcloud run jobs delete petroleum-rag-processor --region=us-central1 --quiet

# Delete Cloud Function
gcloud functions delete petroleum-rag-trigger --region=us-central1 --quiet

# Delete Redis instance
gcloud redis instances delete petroleum-rag-redis --region=us-central1 --quiet

# Delete Cloud Storage bucket
gsutil -m rm -r gs://PROJECT_ID-petroleum-rag

# Delete secrets
gcloud secrets delete anthropic-api-key --quiet
gcloud secrets delete openai-api-key --quiet
gcloud secrets delete llama-cloud-api-key --quiet

# Delete Cloud Scheduler jobs (if created)
gcloud scheduler jobs delete petroleum-rag-daily --location=us-central1 --quiet
```

**⚠️ Warning:** This permanently deletes all data and configurations!

---

## 🔒 Security Best Practices

### 1. Restrict UI Access

```bash
# Require authentication
gcloud run services update petroleum-rag-ui \
  --region=us-central1 \
  --no-allow-unauthenticated

# Access with:
gcloud run services proxy petroleum-rag-ui --port=8080
# Then open: http://localhost:8080
```

### 2. Use VPC for Redis

```bash
# Create VPC connector (already done by deploy script)
gcloud compute networks vpc-access connectors create petroleum-rag-connector \
  --region=us-central1 \
  --network=default \
  --range=10.8.0.0/28

# Attach to Cloud Run
gcloud run services update petroleum-rag-ui \
  --region=us-central1 \
  --vpc-connector=petroleum-rag-connector
```

### 3. Audit Logs

```bash
# Enable audit logging
gcloud logging read "resource.type=cloud_run_revision" --limit=100

# Set up log-based alerts
gcloud logging metrics create error-rate \
  --description="Error rate in Cloud Run" \
  --log-filter='resource.type="cloud_run_revision" AND severity="ERROR"'
```

---

## 📚 Architecture Comparison

### VM Approach (Old)
```
User → [SSH to VM] → [Start Docker] → [Run benchmark.py] → [View results]
      Manual steps required at every stage
```

### Serverless Approach (New)
```
User → [Upload to GCS] → [Auto-trigger] → [Auto-process] → [View results]
      Fully automated, no SSH required
```

---

## ✅ Quick Checklist

**Before deploying:**
- [ ] GCP account with billing enabled
- [ ] gcloud CLI installed and authenticated
- [ ] API keys ready (Anthropic, OpenAI, LlamaParse)
- [ ] Project ID chosen

**After deploying:**
- [ ] Cloud Run services are running
- [ ] Can access Streamlit UI URL
- [ ] Uploaded test document to GCS
- [ ] Processing triggered automatically
- [ ] Results visible in UI
- [ ] Monitoring logs accessible

---

## 🆘 Need Help?

1. **Check deployment info:**
   ```bash
   cat deployment-serverless-info.txt
   ```

2. **View logs:**
   ```bash
   gcloud logging read 'resource.type=cloud_run_job' --limit 50
   ```

3. **Test trigger manually:**
   ```bash
   gcloud run jobs execute petroleum-rag-processor --region=us-central1
   ```

---

## 🎉 Ready to Deploy?

```bash
./deploy_cloudrun_serverless.sh --project YOUR_PROJECT_ID
```

**In ~20 minutes you'll have:**
- ✅ Fully serverless RAG system
- ✅ Auto-processing on document upload
- ✅ No SSH or VM management
- ✅ 57% cost savings vs VM approach
- ✅ Scales to zero when idle

**Upload a document and watch it process automatically!** 🚀
