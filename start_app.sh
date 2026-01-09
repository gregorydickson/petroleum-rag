#!/bin/bash
# Petroleum RAG Benchmark - Complete Application Startup Script
#
# This script:
# 1. Validates environment and dependencies
# 2. Starts Docker services (Chroma, Weaviate, FalkorDB)
# 3. Processes documents in data/input/
# 4. Generates analysis and visualizations
# 5. Launches monitoring server
# 6. Launches Streamlit web UI

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
INPUT_DIR="data/input"
RESULTS_DIR="data/results"
MONITORING_PORT=9090
STREAMLIT_PORT=8501
SKIP_BENCHMARK=false
SKIP_ANALYSIS=false
SKIP_MONITORING=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-benchmark)
            SKIP_BENCHMARK=true
            shift
            ;;
        --skip-analysis)
            SKIP_ANALYSIS=true
            shift
            ;;
        --skip-monitoring)
            SKIP_MONITORING=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-benchmark    Skip benchmark run (use existing results)"
            echo "  --skip-analysis     Skip analysis generation"
            echo "  --skip-monitoring   Skip monitoring server"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

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

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Main script
print_header "🛢️  Petroleum RAG Benchmark - Application Startup"

# Step 1: Validate environment
print_step "Validating environment..."

if ! command_exists python3; then
    print_error "Python 3 is not installed"
    exit 1
fi

if ! command_exists docker; then
    print_error "Docker is not installed"
    exit 1
fi

if ! command_exists docker-compose; then
    print_error "Docker Compose is not installed"
    exit 1
fi

print_success "Environment validation complete"

# Step 2: Check for .env file
print_step "Checking configuration..."

if [ ! -f ".env" ]; then
    print_warning ".env file not found"
    if [ -f ".env.example" ]; then
        print_step "Copying .env.example to .env"
        cp .env.example .env
        print_warning "Please edit .env with your API keys before continuing"
        echo ""
        echo "Required keys:"
        echo "  - ANTHROPIC_API_KEY"
        echo "  - OPENAI_API_KEY"
        echo "  - LLAMA_CLOUD_API_KEY"
        echo ""
        read -p "Press Enter after updating .env file..."
    else
        print_error ".env.example not found"
        exit 1
    fi
fi

# Validate required API keys
print_step "Validating API keys..."
source .env

missing_keys=()

if [ -z "$ANTHROPIC_API_KEY" ] || [ "$ANTHROPIC_API_KEY" = "" ]; then
    missing_keys+=("ANTHROPIC_API_KEY")
fi

if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "" ]; then
    missing_keys+=("OPENAI_API_KEY")
fi

if [ -z "$LLAMA_CLOUD_API_KEY" ] || [ "$LLAMA_CLOUD_API_KEY" = "" ]; then
    missing_keys+=("LLAMA_CLOUD_API_KEY")
fi

if [ ${#missing_keys[@]} -gt 0 ]; then
    print_error "Missing required API keys: ${missing_keys[*]}"
    echo ""
    echo "Please set them in your .env file:"
    for key in "${missing_keys[@]}"; do
        echo "  $key=your_key_here"
    done
    exit 1
fi

print_success "API keys validated"

# Step 3: Check for input documents
print_step "Checking for input documents..."

if [ ! -d "$INPUT_DIR" ]; then
    print_step "Creating $INPUT_DIR directory"
    mkdir -p "$INPUT_DIR"
fi

doc_count=$(find "$INPUT_DIR" -type f \( -name "*.pdf" -o -name "*.docx" -o -name "*.txt" \) | wc -l | tr -d ' ')

if [ "$doc_count" -eq 0 ]; then
    print_warning "No documents found in $INPUT_DIR"
    echo ""
    echo "Please add PDF, DOCX, or TXT files to the $INPUT_DIR directory."
    echo "Supported formats:"
    echo "  - PDF files (.pdf)"
    echo "  - Word documents (.docx)"
    echo "  - Text files (.txt)"
    echo ""
    read -p "Press Enter after adding documents..."

    # Re-check
    doc_count=$(find "$INPUT_DIR" -type f \( -name "*.pdf" -o -name "*.docx" -o -name "*.txt" \) | wc -l | tr -d ' ')

    if [ "$doc_count" -eq 0 ]; then
        print_error "Still no documents found. Exiting."
        exit 1
    fi
fi

print_success "Found $doc_count document(s) to process"

# List documents
echo ""
echo "Documents to process:"
find "$INPUT_DIR" -type f \( -name "*.pdf" -o -name "*.docx" -o -name "*.txt" \) | while read -r file; do
    size=$(du -h "$file" | cut -f1)
    basename=$(basename "$file")
    echo "  - $basename ($size)"
done
echo ""

# Step 4: Start Docker services
print_step "Starting Docker services..."

if docker-compose ps | grep -q "Up"; then
    print_warning "Docker services already running"
else
    docker-compose up -d

    # Wait for services to be ready
    print_step "Waiting for services to be ready..."
    sleep 10

    # Check Chroma
    if curl -sf http://localhost:8000/api/v1/heartbeat > /dev/null 2>&1; then
        print_success "Chroma ready"
    else
        print_warning "Chroma not responding (may need more time)"
    fi

    # Check Weaviate
    if curl -sf http://localhost:8080/v1/.well-known/ready > /dev/null 2>&1; then
        print_success "Weaviate ready"
    else
        print_warning "Weaviate not responding (may need more time)"
    fi

    # Check FalkorDB (Redis)
    if nc -z localhost 6379 2>/dev/null; then
        print_success "FalkorDB ready"
    else
        print_warning "FalkorDB not responding (may need more time)"
    fi
fi

print_success "Docker services started"

# Step 5: Run benchmark
if [ "$SKIP_BENCHMARK" = true ]; then
    print_warning "Skipping benchmark (--skip-benchmark flag)"

    # Check if results exist
    if [ ! -f "$RESULTS_DIR/raw_results.json" ]; then
        print_error "No existing results found in $RESULTS_DIR"
        echo "Cannot skip benchmark without existing results."
        exit 1
    fi

    print_success "Using existing results from $RESULTS_DIR"
else
    print_header "🔬 Running Benchmark"

    print_step "Processing documents with all 4 parsers and 3 storage backends..."
    print_step "This will test all 12 combinations"
    echo ""
    echo "Estimated time:"
    echo "  - Small document (<5 pages): ~15-20 minutes"
    echo "  - Medium document (5-20 pages): ~30-45 minutes"
    echo "  - Large document (>20 pages): ~60-90 minutes"
    echo ""

    # Create results directory
    mkdir -p "$RESULTS_DIR"

    # Run benchmark
    python3 benchmark.py

    if [ $? -eq 0 ]; then
        print_success "Benchmark completed successfully"
    else
        print_error "Benchmark failed"
        exit 1
    fi
fi

# Step 6: Generate analysis
if [ "$SKIP_ANALYSIS" = true ]; then
    print_warning "Skipping analysis (--skip-analysis flag)"
else
    print_header "📊 Generating Analysis"

    print_step "Creating visualizations and reports..."

    python3 analyze_results.py

    if [ $? -eq 0 ]; then
        print_success "Analysis completed"

        # List generated files
        echo ""
        echo "Generated files:"
        [ -f "$RESULTS_DIR/comparison.csv" ] && echo "  ✓ comparison.csv"
        [ -f "$RESULTS_DIR/REPORT.md" ] && echo "  ✓ REPORT.md"
        [ -d "$RESULTS_DIR/charts" ] && echo "  ✓ charts/ directory"
    else
        print_error "Analysis failed"
        exit 1
    fi
fi

# Step 7: Start monitoring server
if [ "$SKIP_MONITORING" = true ]; then
    print_warning "Skipping monitoring server (--skip-monitoring flag)"
else
    print_header "📈 Starting Monitoring Server"

    # Check if monitoring server is already running
    if lsof -Pi :$MONITORING_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        print_warning "Monitoring server already running on port $MONITORING_PORT"
    else
        print_step "Starting monitoring server on port $MONITORING_PORT..."

        # Start in background
        nohup python3 monitoring_server.py > logs/monitoring.log 2>&1 &
        MONITORING_PID=$!

        # Wait for server to start
        sleep 3

        if lsof -Pi :$MONITORING_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
            print_success "Monitoring server started (PID: $MONITORING_PID)"
            echo "  - Health check: http://localhost:$MONITORING_PORT/health"
            echo "  - Metrics: http://localhost:$MONITORING_PORT/metrics"
            echo "  - Stats: http://localhost:$MONITORING_PORT/stats"
        else
            print_warning "Monitoring server may not have started correctly"
        fi
    fi
fi

# Step 8: Display results summary
print_header "📋 Results Summary"

if [ -f "$RESULTS_DIR/raw_results.json" ]; then
    # Extract winner from results
    winner=$(python3 -c "
import json
try:
    with open('$RESULTS_DIR/raw_results.json') as f:
        data = json.load(f)
        results = data.get('results', [])
        if results:
            # Sort by composite score
            sorted_results = sorted(results, key=lambda x: x.get('metrics', {}).get('composite_score', 0), reverse=True)
            if sorted_results:
                best = sorted_results[0]
                print(f\"{best['parser']} + {best['storage']}\")
                print(f\"Score: {best['metrics'].get('composite_score', 0):.4f}\")
except:
    pass
" 2>/dev/null)

    if [ ! -z "$winner" ]; then
        echo -e "${GREEN}🏆 Winner:${NC}"
        echo "$winner" | head -1
        echo ""
        echo -e "${BLUE}Score:${NC}"
        echo "$winner" | tail -1
    fi
fi

echo ""
echo "Results location: $RESULTS_DIR/"
echo ""

# Step 9: Launch Streamlit UI
print_header "🚀 Launching Web UI"

print_step "Starting Streamlit on port $STREAMLIT_PORT..."

# Check if Streamlit is already running
if lsof -Pi :$STREAMLIT_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    print_warning "Streamlit already running on port $STREAMLIT_PORT"
    print_success "Application is ready!"
else
    # Launch Streamlit
    print_success "Starting Streamlit..."
    streamlit run demo_app.py --server.port $STREAMLIT_PORT --server.headless true
fi

# This line won't be reached if Streamlit is launched successfully (it blocks)
print_success "Application startup complete"

echo ""
print_header "🌐 Access Points"
echo ""
echo -e "  ${GREEN}Web UI:${NC}          http://localhost:$STREAMLIT_PORT"
echo -e "  ${BLUE}Monitoring:${NC}      http://localhost:$MONITORING_PORT"
echo -e "  ${BLUE}Grafana:${NC}         http://localhost:3001"
echo -e "  ${BLUE}Prometheus:${NC}      http://localhost:9091"
echo ""
echo -e "${CYAN}Press Ctrl+C to stop the application${NC}"
echo ""
