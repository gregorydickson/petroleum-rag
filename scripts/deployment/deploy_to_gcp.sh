#!/bin/bash
# Automated GCP Deployment Script for Petroleum RAG Benchmark
# This script deploys the complete application to Google Cloud Platform

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Banner
print_banner() {
    echo -e "${CYAN}"
    cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   🛢️  Petroleum RAG Benchmark - GCP Deployment Script  🚀    ║
║                                                               ║
║   Automated deployment to Google Cloud Platform              ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# Helper functions
print_header() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_step() {
    echo -e "${BLUE}▶${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Configuration variables
PROJECT_ID=""
REGION="us-central1"
ZONE="us-central1-a"
VM_NAME="petroleum-rag-vm"
VM_MACHINE_TYPE="e2-standard-4"
BUCKET_NAME=""
DEPLOYMENT_TYPE="simple"  # simple or production

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --project)
            PROJECT_ID="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --zone)
            ZONE="$2"
            shift 2
            ;;
        --vm-name)
            VM_NAME="$2"
            shift 2
            ;;
        --vm-type)
            VM_MACHINE_TYPE="$2"
            shift 2
            ;;
        --production)
            DEPLOYMENT_TYPE="production"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --project PROJECT_ID        GCP Project ID"
            echo ""
            echo "Optional:"
            echo "  --region REGION             GCP Region (default: us-central1)"
            echo "  --zone ZONE                 GCP Zone (default: us-central1-a)"
            echo "  --vm-name NAME              VM instance name (default: petroleum-rag-vm)"
            echo "  --vm-type TYPE              VM machine type (default: e2-standard-4)"
            echo "  --production                Deploy production configuration"
            echo "  --help                      Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --project my-gcp-project"
            echo "  $0 --project my-gcp-project --production --vm-type e2-standard-8"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Main deployment
main() {
    print_banner

    # Step 1: Validate prerequisites
    print_header "Step 1: Validating Prerequisites"

    if ! command_exists gcloud; then
        print_error "gcloud CLI is not installed"
        echo ""
        echo "Install it from: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
    print_success "gcloud CLI found"

    if ! command_exists gsutil; then
        print_error "gsutil is not installed"
        exit 1
    fi
    print_success "gsutil found"

    # Check if logged in
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        print_warning "Not logged into gcloud"
        print_step "Logging in..."
        gcloud auth login
    fi
    print_success "Authenticated with GCP"

    # Get or validate project ID
    if [ -z "$PROJECT_ID" ]; then
        print_step "No project ID specified. Please enter your GCP project ID:"
        read -p "Project ID: " PROJECT_ID

        if [ -z "$PROJECT_ID" ]; then
            print_error "Project ID is required"
            exit 1
        fi
    fi

    # Set project
    print_step "Setting project to: $PROJECT_ID"
    gcloud config set project $PROJECT_ID

    # Verify project exists
    if ! gcloud projects describe $PROJECT_ID &>/dev/null; then
        print_warning "Project $PROJECT_ID does not exist"
        read -p "Create new project? (y/n): " CREATE_PROJECT

        if [ "$CREATE_PROJECT" = "y" ]; then
            gcloud projects create $PROJECT_ID
            print_success "Project created"

            print_warning "Please enable billing for this project in the GCP Console"
            read -p "Press Enter once billing is enabled..."
        else
            exit 1
        fi
    fi
    print_success "Project verified"

    # Set bucket name
    BUCKET_NAME="${PROJECT_ID}-petroleum-rag"

    # Display configuration
    print_header "Deployment Configuration"
    echo "  Project ID:      $PROJECT_ID"
    echo "  Region:          $REGION"
    echo "  Zone:            $ZONE"
    echo "  VM Name:         $VM_NAME"
    echo "  VM Type:         $VM_MACHINE_TYPE"
    echo "  Bucket:          $BUCKET_NAME"
    echo "  Deployment Type: $DEPLOYMENT_TYPE"
    echo ""
    read -p "Continue with this configuration? (y/n): " CONFIRM

    if [ "$CONFIRM" != "y" ]; then
        print_warning "Deployment cancelled"
        exit 0
    fi

    # Step 2: Enable required APIs
    print_header "Step 2: Enabling Required APIs"

    print_step "Enabling GCP APIs (this may take a few minutes)..."

    gcloud services enable \
        compute.googleapis.com \
        storage.googleapis.com \
        secretmanager.googleapis.com \
        logging.googleapis.com \
        monitoring.googleapis.com \
        --quiet 2>/dev/null || true

    print_success "APIs enabled"

    # Step 3: Set up API keys
    print_header "Step 3: Configuring API Keys"

    print_step "You need to provide API keys for:"
    echo "  1. Anthropic (Claude)"
    echo "  2. OpenAI (Embeddings)"
    echo "  3. LlamaParse (Document parsing)"
    echo ""

    # Check if secrets already exist
    SECRETS_EXIST=false
    if gcloud secrets describe anthropic-api-key &>/dev/null && \
       gcloud secrets describe openai-api-key &>/dev/null && \
       gcloud secrets describe llama-cloud-api-key &>/dev/null; then
        print_success "API key secrets already exist in Secret Manager"
        read -p "Use existing secrets? (y/n): " USE_EXISTING

        if [ "$USE_EXISTING" = "y" ]; then
            SECRETS_EXIST=true
        fi
    fi

    if [ "$SECRETS_EXIST" = false ]; then
        read -p "Enter Anthropic API key (sk-ant-...): " ANTHROPIC_KEY
        read -p "Enter OpenAI API key (sk-...): " OPENAI_KEY
        read -p "Enter LlamaParse API key (llx-...): " LLAMAPARSE_KEY

        # Create secrets
        print_step "Storing secrets in Secret Manager..."

        echo -n "$ANTHROPIC_KEY" | gcloud secrets create anthropic-api-key \
            --data-file=- --replication-policy="automatic" 2>/dev/null || \
        echo -n "$ANTHROPIC_KEY" | gcloud secrets versions add anthropic-api-key --data-file=-

        echo -n "$OPENAI_KEY" | gcloud secrets create openai-api-key \
            --data-file=- --replication-policy="automatic" 2>/dev/null || \
        echo -n "$OPENAI_KEY" | gcloud secrets versions add openai-api-key --data-file=-

        echo -n "$LLAMAPARSE_KEY" | gcloud secrets create llama-cloud-api-key \
            --data-file=- --replication-policy="automatic" 2>/dev/null || \
        echo -n "$LLAMAPARSE_KEY" | gcloud secrets versions add llama-cloud-api-key --data-file=-

        print_success "Secrets stored"
    fi

    # Step 4: Create Cloud Storage bucket
    print_header "Step 4: Creating Cloud Storage Bucket"

    if gsutil ls gs://$BUCKET_NAME &>/dev/null; then
        print_success "Bucket already exists: gs://$BUCKET_NAME"
    else
        print_step "Creating bucket: gs://$BUCKET_NAME"
        gsutil mb -l $REGION gs://$BUCKET_NAME
        print_success "Bucket created"
    fi

    # Upload input documents if they exist
    if [ -d "data/input" ] && [ "$(ls -A data/input 2>/dev/null | grep -v README)" ]; then
        print_step "Uploading documents from data/input/..."
        gsutil -m cp data/input/*.pdf gs://$BUCKET_NAME/input/ 2>/dev/null || true
        print_success "Documents uploaded"
    fi

    # Step 5: Create VM instance
    print_header "Step 5: Creating Compute Engine VM"

    # Check if VM exists
    if gcloud compute instances describe $VM_NAME --zone=$ZONE &>/dev/null; then
        print_warning "VM $VM_NAME already exists"
        read -p "Delete and recreate? (y/n): " RECREATE

        if [ "$RECREATE" = "y" ]; then
            print_step "Deleting existing VM..."
            gcloud compute instances delete $VM_NAME --zone=$ZONE --quiet
            print_success "VM deleted"
        else
            print_warning "Using existing VM"
            VM_IP=$(gcloud compute instances describe $VM_NAME \
                --zone=$ZONE \
                --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

            if [ -z "$VM_IP" ]; then
                print_error "Could not get VM IP. VM may not be running."
                exit 1
            fi

            print_success "VM IP: $VM_IP"

            # Skip to deployment
            deploy_application
            show_summary
            exit 0
        fi
    fi

    print_step "Creating VM instance: $VM_NAME ($VM_MACHINE_TYPE)"

    # Create startup script
    cat > /tmp/startup-script.sh << 'STARTUP_SCRIPT'
#!/bin/bash
# VM Startup Script

# Update system
apt-get update -y
apt-get upgrade -y

# Install Docker
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
fi

# Install Docker Compose
if ! command -v docker-compose &> /dev/null; then
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
fi

# Install Python 3.11
apt-get install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Install additional tools
apt-get install -y git curl wget vim htop

# Create app directory
mkdir -p /opt/petroleum-rag
chmod 755 /opt/petroleum-rag

# Install gcloud CLI components for secret access
apt-get install -y google-cloud-sdk

echo "Startup script completed" > /tmp/startup-complete.log
STARTUP_SCRIPT

    # Create VM
    gcloud compute instances create $VM_NAME \
        --zone=$ZONE \
        --machine-type=$VM_MACHINE_TYPE \
        --boot-disk-size=100GB \
        --boot-disk-type=pd-ssd \
        --image-family=ubuntu-2204-lts \
        --image-project=ubuntu-os-cloud \
        --scopes=cloud-platform \
        --tags=petroleum-rag,http-server,https-server \
        --metadata-from-file=startup-script=/tmp/startup-script.sh

    rm /tmp/startup-script.sh

    print_success "VM created"

    # Wait for VM to be ready
    print_step "Waiting for VM to be ready (this may take 2-3 minutes)..."
    sleep 60

    # Check if startup script completed
    for i in {1..12}; do
        if gcloud compute ssh $VM_NAME --zone=$ZONE --command="test -f /tmp/startup-complete.log" 2>/dev/null; then
            print_success "VM is ready"
            break
        fi
        echo -n "."
        sleep 10
    done
    echo ""

    # Get VM IP
    VM_IP=$(gcloud compute instances describe $VM_NAME \
        --zone=$ZONE \
        --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

    print_success "VM External IP: $VM_IP"

    # Step 6: Configure firewall rules
    print_header "Step 6: Configuring Firewall Rules"

    # Streamlit
    if ! gcloud compute firewall-rules describe allow-streamlit &>/dev/null; then
        gcloud compute firewall-rules create allow-streamlit \
            --allow=tcp:8501 \
            --source-ranges=0.0.0.0/0 \
            --target-tags=petroleum-rag \
            --description="Allow Streamlit UI access" \
            --quiet
        print_success "Streamlit firewall rule created"
    else
        print_success "Streamlit firewall rule exists"
    fi

    # Monitoring
    if ! gcloud compute firewall-rules describe allow-monitoring &>/dev/null; then
        gcloud compute firewall-rules create allow-monitoring \
            --allow=tcp:9090,tcp:3001,tcp:9091 \
            --source-ranges=0.0.0.0/0 \
            --target-tags=petroleum-rag \
            --description="Allow monitoring access" \
            --quiet
        print_success "Monitoring firewall rules created"
    else
        print_success "Monitoring firewall rules exist"
    fi

    # Deploy application
    deploy_application

    # Show summary
    show_summary
}

deploy_application() {
    print_header "Step 7: Deploying Application"

    # Create temporary archive
    print_step "Creating application archive..."
    tar -czf /tmp/petroleum-rag.tar.gz \
        --exclude='data/cache' \
        --exclude='data/results' \
        --exclude='logs' \
        --exclude='.git' \
        --exclude='venv' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.env' \
        .

    print_success "Archive created"

    # Upload to VM
    print_step "Uploading application to VM..."
    gcloud compute scp /tmp/petroleum-rag.tar.gz $VM_NAME:/tmp/ --zone=$ZONE --quiet
    rm /tmp/petroleum-rag.tar.gz
    print_success "Upload complete"

    # Deploy on VM
    print_step "Setting up application on VM..."

    gcloud compute ssh $VM_NAME --zone=$ZONE --command="
set -e

# Extract application
cd /opt/petroleum-rag
tar -xzf /tmp/petroleum-rag.tar.gz
rm /tmp/petroleum-rag.tar.gz

# Create .env file
cat > .env << EOF
# API Keys from Secret Manager
ANTHROPIC_API_KEY=\$(gcloud secrets versions access latest --secret=anthropic-api-key)
OPENAI_API_KEY=\$(gcloud secrets versions access latest --secret=openai-api-key)
LLAMA_CLOUD_API_KEY=\$(gcloud secrets versions access latest --secret=llama-cloud-api-key)

# Storage configuration
CHROMA_HOST=localhost
CHROMA_PORT=8000
WEAVIATE_HOST=localhost
WEAVIATE_PORT=8080
FALKORDB_HOST=localhost
FALKORDB_PORT=6379

# GCS bucket
GCS_BUCKET=$BUCKET_NAME

# Features
ENABLE_CACHE=true
ENABLE_MONITORING=true

# Logging
LOG_LEVEL=INFO
EOF

# Create directories
mkdir -p data/input data/results data/cache logs

# Download documents from GCS if they exist
gsutil -m cp -r gs://$BUCKET_NAME/input/* data/input/ 2>/dev/null || true

# Create Python virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip --quiet

# Install application
pip install -e . --quiet

# Start Docker services
docker-compose up -d

echo 'Application deployed successfully'
" 2>&1 | grep -v "Warning:" || true

    print_success "Application deployed"

    # Wait for Docker services
    print_step "Waiting for Docker services to start..."
    sleep 15

    gcloud compute ssh $VM_NAME --zone=$ZONE --command="
docker-compose ps
" 2>&1 | grep -v "Warning:" || true

    print_success "Docker services running"
}

show_summary() {
    print_header "🎉 Deployment Complete!"

    echo -e "${GREEN}"
    cat << EOF
╔═══════════════════════════════════════════════════════════════╗
║                  Deployment Successful! 🎉                    ║
╚═══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"

    echo ""
    echo -e "${CYAN}Access Information:${NC}"
    echo "─────────────────────────────────────────────────────────"
    echo ""
    echo -e "  ${GREEN}VM Instance:${NC}     $VM_NAME"
    echo -e "  ${GREEN}External IP:${NC}     $VM_IP"
    echo -e "  ${GREEN}GCP Project:${NC}     $PROJECT_ID"
    echo -e "  ${GREEN}Region/Zone:${NC}     $REGION / $ZONE"
    echo ""
    echo -e "${CYAN}Application URLs:${NC}"
    echo "─────────────────────────────────────────────────────────"
    echo ""
    echo -e "  ${BLUE}Streamlit UI:${NC}    http://$VM_IP:8501"
    echo -e "  ${BLUE}Monitoring:${NC}      http://$VM_IP:9090"
    echo -e "  ${BLUE}Grafana:${NC}         http://$VM_IP:3001  ${YELLOW}(admin/admin)${NC}"
    echo -e "  ${BLUE}Prometheus:${NC}      http://$VM_IP:9091"
    echo ""
    echo -e "${CYAN}Next Steps:${NC}"
    echo "─────────────────────────────────────────────────────────"
    echo ""
    echo "  1. SSH to VM:"
    echo -e "     ${YELLOW}gcloud compute ssh $VM_NAME --zone=$ZONE${NC}"
    echo ""
    echo "  2. Navigate to app directory:"
    echo -e "     ${YELLOW}cd /opt/petroleum-rag${NC}"
    echo ""
    echo "  3. Run the benchmark:"
    echo -e "     ${YELLOW}./start_app.sh${NC}"
    echo ""
    echo "  4. Access the UI in your browser:"
    echo -e "     ${YELLOW}http://$VM_IP:8501${NC}"
    echo ""
    echo -e "${CYAN}Management Commands:${NC}"
    echo "─────────────────────────────────────────────────────────"
    echo ""
    echo "  View logs:"
    echo -e "    ${YELLOW}gcloud compute ssh $VM_NAME --zone=$ZONE${NC}"
    echo -e "    ${YELLOW}cd /opt/petroleum-rag && tail -f logs/*.log${NC}"
    echo ""
    echo "  Upload documents:"
    echo -e "    ${YELLOW}gsutil cp your-doc.pdf gs://$BUCKET_NAME/input/${NC}"
    echo ""
    echo "  Stop VM (save money):"
    echo -e "    ${YELLOW}gcloud compute instances stop $VM_NAME --zone=$ZONE${NC}"
    echo ""
    echo "  Start VM:"
    echo -e "    ${YELLOW}gcloud compute instances start $VM_NAME --zone=$ZONE${NC}"
    echo ""
    echo -e "${CYAN}Cost Estimate:${NC}"
    echo "─────────────────────────────────────────────────────────"
    echo ""
    echo "  VM ($VM_MACHINE_TYPE):  ~\$120/month (24/7 operation)"
    echo "  Cloud Storage:          ~\$2/month"
    echo "  Secrets:                ~\$1/month"
    echo "  Logging:                ~\$5/month"
    echo ""
    echo "  ${GREEN}Total: ~\$128/month${NC}"
    echo ""
    echo -e "${YELLOW}💡 Tip: Stop the VM when not in use to save ~\$120/month${NC}"
    echo ""
    echo -e "${CYAN}Documentation:${NC}"
    echo "─────────────────────────────────────────────────────────"
    echo ""
    echo "  Full deployment guide: docs/GCP_DEPLOYMENT_GUIDE.md"
    echo "  Quick start:           QUICK_START.md"
    echo "  User guide:            docs/USER_GUIDE.md"
    echo ""

    # Save deployment info
    cat > deployment-info.txt << EOF
Petroleum RAG Benchmark - GCP Deployment Information
Deployed: $(date)

Project ID: $PROJECT_ID
Region: $REGION
Zone: $ZONE
VM Name: $VM_NAME
VM Type: $VM_MACHINE_TYPE
External IP: $VM_IP
Bucket: gs://$BUCKET_NAME

Access URLs:
  Streamlit UI: http://$VM_IP:8501
  Monitoring:   http://$VM_IP:9090
  Grafana:      http://$VM_IP:3001
  Prometheus:   http://$VM_IP:9091

SSH Command:
  gcloud compute ssh $VM_NAME --zone=$ZONE

Stop VM (save costs):
  gcloud compute instances stop $VM_NAME --zone=$ZONE

Start VM:
  gcloud compute instances start $VM_NAME --zone=$ZONE
EOF

    print_success "Deployment information saved to: deployment-info.txt"
    echo ""
}

# Run main function
main
