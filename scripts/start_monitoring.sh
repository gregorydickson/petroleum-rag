#!/bin/bash
# Start monitoring infrastructure for RAG Benchmark System
#
# This script:
# 1. Starts Docker services (Prometheus, Grafana, storage backends)
# 2. Waits for services to be healthy
# 3. Starts the monitoring server
# 4. Opens Grafana dashboard in browser

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    print_error "docker-compose is not installed. Please install it and try again."
    exit 1
fi

print_info "Starting monitoring infrastructure..."

# Start Docker services
print_info "Starting Docker services (ChromaDB, Weaviate, FalkorDB, Prometheus, Grafana)..."
docker-compose up -d

# Wait for services to be healthy
print_info "Waiting for services to be healthy..."
sleep 5

# Check service health
print_info "Checking service health..."

services=("chroma:8000/api/v1/heartbeat" "weaviate:8080/v1/.well-known/ready" "prometheus:9090/-/healthy")
all_healthy=true

for service in "${services[@]}"; do
    IFS=':' read -r name endpoint <<< "$service"
    container_name="petroleum-rag-${name}"

    # Check if container is running
    if ! docker ps --format '{{.Names}}' | grep -q "^${container_name}$"; then
        print_warn "Container ${container_name} is not running"
        all_healthy=false
        continue
    fi

    # Check if container is healthy
    health=$(docker inspect --format='{{.State.Health.Status}}' "${container_name}" 2>/dev/null || echo "unknown")
    if [ "$health" = "healthy" ] || [ "$health" = "unknown" ]; then
        print_info "✓ ${name} is healthy"
    else
        print_warn "✗ ${name} health check: ${health}"
        all_healthy=false
    fi
done

if [ "$all_healthy" = false ]; then
    print_warn "Some services are not healthy yet. They may still be starting up."
    print_warn "You can check status with: docker-compose ps"
fi

# Check if monitoring server is already running
if lsof -Pi :9090 -sTCP:LISTEN -t >/dev/null 2>&1; then
    print_warn "Monitoring server is already running on port 9090"
else
    # Start monitoring server in background
    print_info "Starting monitoring server..."

    # Check if Python virtual environment exists
    if [ -d "venv" ]; then
        source venv/bin/activate
    elif [ -d ".venv" ]; then
        source .venv/bin/activate
    fi

    # Start monitoring server in background
    nohup python monitoring_server.py > monitoring_server.log 2>&1 &
    MONITORING_PID=$!

    print_info "Monitoring server started (PID: ${MONITORING_PID})"
    print_info "Logs: tail -f monitoring_server.log"

    # Wait for monitoring server to start
    sleep 2
fi

# Check monitoring server health
print_info "Checking monitoring server health..."
if curl -s http://localhost:9090/health > /dev/null 2>&1; then
    print_info "✓ Monitoring server is healthy"
else
    print_error "✗ Monitoring server is not responding"
    print_error "Check logs: tail -f monitoring_server.log"
fi

# Print access information
echo ""
print_info "=========================================="
print_info "Monitoring Infrastructure is Ready!"
print_info "=========================================="
echo ""
echo "Access points:"
echo "  - Grafana:           http://localhost:3001 (admin/admin)"
echo "  - Prometheus:        http://localhost:9091"
echo "  - Monitoring Server: http://localhost:9090"
echo "  - Health Checks:     http://localhost:9090/health"
echo "  - Metrics:           http://localhost:9090/metrics"
echo ""
echo "Storage backends:"
echo "  - ChromaDB:          http://localhost:8000"
echo "  - Weaviate:          http://localhost:8080"
echo "  - FalkorDB:          localhost:6379"
echo ""
echo "To stop:"
echo "  docker-compose down"
echo "  kill \$(lsof -t -i:9090)  # Stop monitoring server"
echo ""

# Open Grafana in browser (optional)
if command -v open &> /dev/null; then
    read -p "Open Grafana dashboard in browser? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Opening Grafana..."
        sleep 2  # Give Grafana time to fully start
        open http://localhost:3001
    fi
fi

print_info "Setup complete! 🎉"
