#!/bin/bash

# Start Services Script for Petroleum RAG Benchmark
# This script starts all required Docker services (ChromaDB, Weaviate, FalkorDB)

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Petroleum RAG Benchmark - Starting Services  ${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Error: Docker is not running${NC}"
    echo "Please start Docker Desktop and try again"
    exit 1
fi

echo -e "${GREEN}✓ Docker is running${NC}"
echo ""

# Check if docker-compose.yml exists
if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}❌ Error: docker-compose.yml not found${NC}"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Start services
echo -e "${BLUE}Starting Docker services...${NC}"
docker-compose up -d

# Wait for services to be healthy
echo ""
echo -e "${BLUE}Waiting for services to be ready...${NC}"
echo ""

# Function to check service health
check_service() {
    local service=$1
    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if docker-compose ps | grep -q "$service.*healthy\|$service.*Up"; then
            echo -e "${GREEN}✓ $service is ready${NC}"
            return 0
        fi

        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done

    echo -e "${RED}❌ $service failed to start${NC}"
    return 1
}

# Check each service
echo -n "ChromaDB (port 8000): "
check_service "chroma"

echo -n "Weaviate (port 8080): "
check_service "weaviate"

echo -n "FalkorDB (port 6379): "
check_service "falkordb"

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  All services are running successfully!      ${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo -e "Service endpoints:"
echo -e "  - ChromaDB:  ${BLUE}http://localhost:8000${NC}"
echo -e "  - Weaviate:  ${BLUE}http://localhost:8080${NC}"
echo -e "  - FalkorDB:  ${BLUE}localhost:6379${NC}"
echo ""
echo -e "To view service logs:"
echo -e "  ${YELLOW}docker-compose logs -f${NC}"
echo ""
echo -e "To stop services:"
echo -e "  ${YELLOW}./scripts/stop_services.sh${NC}"
echo ""
echo -e "To verify services:"
echo -e "  ${YELLOW}./scripts/verify_services.sh${NC}"
echo ""
