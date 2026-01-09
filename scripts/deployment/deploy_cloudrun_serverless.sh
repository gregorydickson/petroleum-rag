#!/bin/bash
# Serverless GCP Deployment - No SSH Required!
# This deploys everything to Cloud Run with automatic triggers

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_banner() {
    echo -e "${CYAN}"
    cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   🛢️  Petroleum RAG - Serverless Cloud Run Deployment 🚀     ║
║                                                               ║
║   • No SSH required                                           ║
║   • Auto-starts on document upload                            ║
║   • Scales to zero when idle                                  ║
║   • Fully managed services                                    ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

print_header() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_step() { echo -e "${BLUE}▶${NC} $1"; }
print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }

# Configuration
PROJECT_ID=""
REGION="us-central1"
BUCKET_NAME=""
REDIS_INSTANCE="petroleum-rag-redis"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --project) PROJECT_ID="$2"; shift 2 ;;
        --region) REGION="$2"; shift 2 ;;
        --help)
            echo "Usage: $0 --project PROJECT_ID [--region REGION]"
            echo ""
            echo "Deploys fully serverless architecture:"
            echo "  • Cloud Run for Streamlit UI (auto-scaling)"
            echo "  • Cloud Run Jobs for processing (triggered automatically)"
            echo "  • Memorystore Redis for storage"
            echo "  • Cloud Storage for documents"
            echo "  • Eventarc for auto-triggering on upload"
            echo ""
            echo "No SSH required!"
            exit 0
            ;;
        *) print_error "Unknown option: $1"; exit 1 ;;
    esac
done

main() {
    print_banner

    # Validate project
    if [ -z "$PROJECT_ID" ]; then
        print_step "Enter your GCP project ID:"
        read -p "Project ID: " PROJECT_ID
    fi

    gcloud config set project $PROJECT_ID
    BUCKET_NAME="${PROJECT_ID}-petroleum-rag"

    print_header "Deployment Configuration"
    echo "  Project:  $PROJECT_ID"
    echo "  Region:   $REGION"
    echo "  Bucket:   $BUCKET_NAME"
    echo ""
    read -p "Continue? (y/n): " CONFIRM
    [ "$CONFIRM" != "y" ] && exit 0

    # Enable APIs
    print_header "Step 1: Enabling APIs"
    print_step "Enabling required services..."

    gcloud services enable \
        run.googleapis.com \
        cloudbuild.googleapis.com \
        storage.googleapis.com \
        redis.googleapis.com \
        secretmanager.googleapis.com \
        eventarc.googleapis.com \
        cloudscheduler.googleapis.com \
        pubsub.googleapis.com \
        artifactregistry.googleapis.com \
        --quiet

    print_success "APIs enabled"

    # Store secrets
    print_header "Step 2: Configuring Secrets"

    if gcloud secrets describe anthropic-api-key &>/dev/null; then
        print_success "Secrets already exist"
    else
        read -p "Anthropic API key: " ANTHROPIC_KEY
        read -p "OpenAI API key: " OPENAI_KEY
        read -p "LlamaParse API key: " LLAMAPARSE_KEY

        echo -n "$ANTHROPIC_KEY" | gcloud secrets create anthropic-api-key --data-file=- --replication-policy=automatic
        echo -n "$OPENAI_KEY" | gcloud secrets create openai-api-key --data-file=- --replication-policy=automatic
        echo -n "$LLAMAPARSE_KEY" | gcloud secrets create llama-cloud-api-key --data-file=- --replication-policy=automatic

        print_success "Secrets stored"
    fi

    # Create bucket
    print_header "Step 3: Creating Storage"

    gsutil mb -l $REGION gs://$BUCKET_NAME 2>/dev/null || print_success "Bucket exists"

    # Enable versioning and lifecycle
    gsutil versioning set on gs://$BUCKET_NAME
    print_success "Cloud Storage ready"

    # Create Redis instance for FalkorDB
    print_header "Step 4: Creating Memorystore Redis"

    if gcloud redis instances describe $REDIS_INSTANCE --region=$REGION &>/dev/null; then
        print_success "Redis instance exists"
    else
        print_step "Creating Redis instance (this takes ~5 minutes)..."
        gcloud redis instances create $REDIS_INSTANCE \
            --size=5 \
            --region=$REGION \
            --redis-version=redis_7_0 \
            --tier=basic \
            --enable-auth \
            --async

        print_step "Waiting for Redis to be ready..."
        gcloud redis operations wait \
            $(gcloud redis operations list --region=$REGION --filter="targetId:$REDIS_INSTANCE" --format="value(name)" | head -1) \
            --region=$REGION

        print_success "Redis ready"
    fi

    REDIS_HOST=$(gcloud redis instances describe $REDIS_INSTANCE --region=$REGION --format='get(host)')
    REDIS_PORT=$(gcloud redis instances describe $REDIS_INSTANCE --region=$REGION --format='get(port)')

    print_success "Redis: $REDIS_HOST:$REDIS_PORT"

    # Build and deploy processing job
    print_header "Step 5: Building Processing Job"

    print_step "Creating Dockerfile for processing job..."

    cat > Dockerfile.processor << 'DOCKERFILE'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ curl git \
    && rm -rf /var/lib/apt/lists/*

# Copy application
COPY pyproject.toml .
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -e .

# Create directories
RUN mkdir -p data/input data/results data/cache logs

# Default command runs benchmark
CMD ["python", "-u", "benchmark.py"]
DOCKERFILE

    print_step "Building container image..."
    gcloud builds submit --tag gcr.io/$PROJECT_ID/petroleum-rag-processor .

    print_success "Processing image built"

    # Create Cloud Run Job for processing
    print_header "Step 6: Deploying Processing Job"

    print_step "Creating Cloud Run Job..."

    gcloud run jobs create petroleum-rag-processor \
        --image gcr.io/$PROJECT_ID/petroleum-rag-processor \
        --region $REGION \
        --memory 8Gi \
        --cpu 4 \
        --max-retries 1 \
        --task-timeout 3600 \
        --set-secrets "ANTHROPIC_API_KEY=anthropic-api-key:latest,OPENAI_API_KEY=openai-api-key:latest,LLAMA_CLOUD_API_KEY=llama-cloud-api-key:latest" \
        --set-env-vars "GCS_BUCKET=$BUCKET_NAME,FALKORDB_HOST=$REDIS_HOST,FALKORDB_PORT=$REDIS_PORT,ENABLE_CACHE=true" \
        --execute-now=false \
        2>/dev/null || \
    gcloud run jobs update petroleum-rag-processor \
        --image gcr.io/$PROJECT_ID/petroleum-rag-processor \
        --region $REGION

    print_success "Processing job created"

    # Build and deploy Streamlit UI
    print_header "Step 7: Deploying Streamlit UI"

    print_step "Creating Dockerfile for UI..."

    cat > Dockerfile.ui << 'DOCKERFILE'
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY . .

RUN pip install --no-cache-dir -e .

# Expose Streamlit port
EXPOSE 8080

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health || exit 1

# Run Streamlit on Cloud Run's expected port
CMD streamlit run demo_app.py \
    --server.port=8080 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
DOCKERFILE

    print_step "Building UI image..."
    gcloud builds submit --tag gcr.io/$PROJECT_ID/petroleum-rag-ui -f Dockerfile.ui .

    print_step "Deploying to Cloud Run..."
    gcloud run deploy petroleum-rag-ui \
        --image gcr.io/$PROJECT_ID/petroleum-rag-ui \
        --platform managed \
        --region $REGION \
        --allow-unauthenticated \
        --memory 4Gi \
        --cpu 2 \
        --timeout 3600 \
        --min-instances 0 \
        --max-instances 10 \
        --set-secrets "ANTHROPIC_API_KEY=anthropic-api-key:latest,OPENAI_API_KEY=openai-api-key:latest,LLAMA_CLOUD_API_KEY=llama-cloud-api-key:latest" \
        --set-env-vars "GCS_BUCKET=$BUCKET_NAME,FALKORDB_HOST=$REDIS_HOST,FALKORDB_PORT=$REDIS_PORT"

    UI_URL=$(gcloud run services describe petroleum-rag-ui --region=$REGION --format='value(status.url)')

    print_success "UI deployed: $UI_URL"

    # Set up auto-trigger on document upload
    print_header "Step 8: Configuring Auto-Trigger"

    print_step "Creating Cloud Function to trigger processing..."

    # Create trigger function
    mkdir -p /tmp/trigger-function
    cat > /tmp/trigger-function/main.py << 'PYTHON'
import functions_framework
from google.cloud import run_v2
import os

@functions_framework.cloud_event
def trigger_processing(cloud_event):
    """Trigger Cloud Run Job when document uploaded to GCS"""

    # Get file info
    file_name = cloud_event.data["name"]

    # Only trigger for PDF files in input/ directory
    if not file_name.startswith("input/") or not file_name.endswith(".pdf"):
        print(f"Skipping {file_name} (not a PDF in input/)")
        return

    print(f"New document uploaded: {file_name}")
    print("Triggering benchmark processing job...")

    # Trigger Cloud Run Job
    project = os.environ.get("GCP_PROJECT")
    region = os.environ.get("GCP_REGION")

    client = run_v2.JobsClient()
    job_name = f"projects/{project}/locations/{region}/jobs/petroleum-rag-processor"

    try:
        operation = client.run_job(name=job_name)
        print(f"Job triggered successfully: {operation.name}")
    except Exception as e:
        print(f"Error triggering job: {e}")
        raise
PYTHON

    cat > /tmp/trigger-function/requirements.txt << 'REQS'
functions-framework==3.*
google-cloud-run==0.10.*
REQS

    print_step "Deploying trigger function..."
    gcloud functions deploy petroleum-rag-trigger \
        --gen2 \
        --runtime python311 \
        --region $REGION \
        --source /tmp/trigger-function \
        --entry-point trigger_processing \
        --trigger-event-filters="type=google.cloud.storage.object.v1.finalized" \
        --trigger-event-filters="bucket=$BUCKET_NAME" \
        --set-env-vars "GCP_PROJECT=$PROJECT_ID,GCP_REGION=$REGION" \
        --quiet

    rm -rf /tmp/trigger-function

    print_success "Auto-trigger configured"

    # Create scheduled trigger (optional)
    print_header "Step 9: Setting Up Scheduled Processing (Optional)"

    read -p "Set up daily processing schedule? (y/n): " SETUP_SCHEDULE

    if [ "$SETUP_SCHEDULE" = "y" ]; then
        print_step "Creating Cloud Scheduler job..."

        # Create service account for scheduler
        gcloud iam service-accounts create cloud-run-invoker \
            --display-name="Cloud Run Job Invoker" 2>/dev/null || true

        # Grant permission to run jobs
        gcloud run jobs add-iam-policy-binding petroleum-rag-processor \
            --region=$REGION \
            --member="serviceAccount:cloud-run-invoker@$PROJECT_ID.iam.gserviceaccount.com" \
            --role="roles/run.invoker" \
            --quiet

        # Create scheduler job (daily at 2 AM)
        gcloud scheduler jobs create http petroleum-rag-daily \
            --location=$REGION \
            --schedule="0 2 * * *" \
            --uri="https://$REGION-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$PROJECT_ID/jobs/petroleum-rag-processor:run" \
            --http-method=POST \
            --oauth-service-account-email="cloud-run-invoker@$PROJECT_ID.iam.gserviceaccount.com" \
            2>/dev/null || \
        print_warning "Scheduler job may already exist"

        print_success "Daily processing scheduled for 2 AM"
    fi

    # Show summary
    show_summary
}

show_summary() {
    print_header "🎉 Serverless Deployment Complete!"

    echo -e "${GREEN}"
    cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║              Fully Serverless - No SSH Required! 🚀           ║
╚═══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"

    echo ""
    echo -e "${CYAN}Access Points:${NC}"
    echo "─────────────────────────────────────────────────────────"
    echo ""
    echo -e "  ${GREEN}Streamlit UI:${NC}  $UI_URL"
    echo ""
    echo -e "${CYAN}How It Works:${NC}"
    echo "─────────────────────────────────────────────────────────"
    echo ""
    echo "  1. ${BLUE}Upload Document:${NC}"
    echo "     gsutil cp your-doc.pdf gs://$BUCKET_NAME/input/"
    echo ""
    echo "  2. ${BLUE}Auto-Processing:${NC}"
    echo "     → Cloud Function detects upload"
    echo "     → Triggers Cloud Run Job automatically"
    echo "     → Processing starts (no SSH needed!)"
    echo "     → Results saved to Cloud Storage"
    echo ""
    echo "  3. ${BLUE}View Results:${NC}"
    echo "     → Open $UI_URL"
    echo "     → See benchmark results"
    echo "     → Ask questions in chat"
    echo ""
    echo -e "${CYAN}Manual Triggers:${NC}"
    echo "─────────────────────────────────────────────────────────"
    echo ""
    echo "  Run processing manually:"
    echo "    ${YELLOW}gcloud run jobs execute petroleum-rag-processor --region=$REGION${NC}"
    echo ""
    echo "  View job logs:"
    echo "    ${YELLOW}gcloud run jobs executions list --region=$REGION${NC}"
    echo ""
    echo "  Upload documents:"
    echo "    ${YELLOW}gsutil cp *.pdf gs://$BUCKET_NAME/input/${NC}"
    echo ""
    echo -e "${CYAN}Cost Savings:${NC}"
    echo "─────────────────────────────────────────────────────────"
    echo ""
    echo "  ✅ UI scales to zero when idle (free!)"
    echo "  ✅ Processing only runs when needed"
    echo "  ✅ No VM running 24/7"
    echo "  ✅ Pay only for actual usage"
    echo ""
    echo "  Estimated costs:"
    echo "    - Redis (always on):        ~\$50/month"
    echo "    - Cloud Storage:            ~\$2/month"
    echo "    - Cloud Run (per run):      ~\$0.50"
    echo "    - Processing APIs:          ~\$2.50/run (first)"
    echo "    ${GREEN}Total: ~\$55/month base + usage${NC}"
    echo ""
    echo "  ${BLUE}vs VM approach: \$128/month = 57% savings!${NC}"
    echo ""
    echo -e "${CYAN}Next Steps:${NC}"
    echo "─────────────────────────────────────────────────────────"
    echo ""
    echo "  1. Upload your first document:"
    echo "     ${YELLOW}gsutil cp data/input/*.pdf gs://$BUCKET_NAME/input/${NC}"
    echo ""
    echo "  2. Processing starts automatically (check logs):"
    echo "     ${YELLOW}gcloud logging read 'resource.type=cloud_run_job'${NC}"
    echo ""
    echo "  3. View results when ready:"
    echo "     ${YELLOW}open $UI_URL${NC}"
    echo ""

    # Save info
    cat > deployment-serverless-info.txt << EOF
Petroleum RAG - Serverless Deployment
Deployed: $(date)

Project: $PROJECT_ID
Region: $REGION
Bucket: gs://$BUCKET_NAME

Streamlit UI: $UI_URL

Upload documents:
  gsutil cp your-doc.pdf gs://$BUCKET_NAME/input/

Trigger processing manually:
  gcloud run jobs execute petroleum-rag-processor --region=$REGION

View logs:
  gcloud logging read 'resource.type=cloud_run_job' --limit 50

No SSH required!
EOF

    print_success "Info saved to: deployment-serverless-info.txt"
    echo ""
}

main
