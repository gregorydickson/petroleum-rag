#!/bin/bash

# Verify Services Script for Petroleum RAG Benchmark
# This script checks that all Docker services are running and responding correctly

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}===================================================${NC}"
echo -e "${BLUE}  Petroleum RAG Benchmark - Verifying Services    ${NC}"
echo -e "${BLUE}===================================================${NC}"
echo ""

# Track overall status
ALL_HEALTHY=true

# Function to check if a port is open
check_port() {
    local host=$1
    local port=$2
    nc -z -w5 "$host" "$port" > /dev/null 2>&1
    return $?
}

# Function to check HTTP endpoint
check_http() {
    local url=$1
    local response=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null)
    [ "$response" = "200" ] || [ "$response" = "404" ]  # 404 is ok for some endpoints
    return $?
}

# Check Docker
echo -e "${BLUE}Checking Docker...${NC}"
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running${NC}"
    ALL_HEALTHY=false
else
    echo -e "${GREEN}✓ Docker is running${NC}"
fi
echo ""

# Check if docker-compose.yml exists
if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}❌ Error: docker-compose.yml not found${NC}"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Check Docker containers
echo -e "${BLUE}Checking Docker containers...${NC}"
CONTAINERS=$(docker-compose ps --services 2>/dev/null || echo "")

if [ -z "$CONTAINERS" ]; then
    echo -e "${RED}❌ No containers are running${NC}"
    echo -e "${YELLOW}Run './scripts/start_services.sh' to start services${NC}"
    exit 1
fi

# Check each service
for service in chroma weaviate falkordb; do
    echo -n "  - $service: "
    if docker-compose ps | grep -q "$service.*Up"; then
        echo -e "${GREEN}✓ Running${NC}"
    else
        echo -e "${RED}❌ Not running${NC}"
        ALL_HEALTHY=false
    fi
done
echo ""

# Check ChromaDB (port 8000)
echo -e "${BLUE}Checking ChromaDB (port 8000)...${NC}"
if check_port localhost 8000; then
    echo -e "${GREEN}✓ Port 8000 is open${NC}"

    if check_http "http://localhost:8000/api/v1/heartbeat"; then
        echo -e "${GREEN}✓ ChromaDB API is responding${NC}"
    else
        echo -e "${YELLOW}⚠️  ChromaDB port is open but API not responding${NC}"
        ALL_HEALTHY=false
    fi
else
    echo -e "${RED}❌ ChromaDB port 8000 is not accessible${NC}"
    ALL_HEALTHY=false
fi
echo ""

# Check Weaviate (port 8080)
echo -e "${BLUE}Checking Weaviate (port 8080)...${NC}"
if check_port localhost 8080; then
    echo -e "${GREEN}✓ Port 8080 is open${NC}"

    if check_http "http://localhost:8080/v1/.well-known/ready"; then
        echo -e "${GREEN}✓ Weaviate API is responding${NC}"
    else
        echo -e "${YELLOW}⚠️  Weaviate port is open but API not fully ready${NC}"
        ALL_HEALTHY=false
    fi
else
    echo -e "${RED}❌ Weaviate port 8080 is not accessible${NC}"
    ALL_HEALTHY=false
fi
echo ""

# Check FalkorDB (port 6379)
echo -e "${BLUE}Checking FalkorDB (port 6379)...${NC}"
if check_port localhost 6379; then
    echo -e "${GREEN}✓ Port 6379 is open${NC}"

    # Try to ping Redis
    if docker exec petroleum-rag-falkordb redis-cli ping > /dev/null 2>&1; then
        echo -e "${GREEN}✓ FalkorDB is responding to Redis commands${NC}"

        # Check if FalkorDB module is loaded
        if docker exec petroleum-rag-falkordb redis-cli MODULE LIST | grep -q "graph"; then
            echo -e "${GREEN}✓ FalkorDB graph module is loaded${NC}"
        else
            echo -e "${YELLOW}⚠️  FalkorDB graph module not detected${NC}"
            ALL_HEALTHY=false
        fi
    else
        echo -e "${YELLOW}⚠️  FalkorDB port is open but not responding${NC}"
        ALL_HEALTHY=false
    fi
else
    echo -e "${RED}❌ FalkorDB port 6379 is not accessible${NC}"
    ALL_HEALTHY=false
fi
echo ""

# Final status
echo -e "${BLUE}===================================================${NC}"
if [ "$ALL_HEALTHY" = true ]; then
    echo -e "${GREEN}  ✓ All services are healthy and operational!     ${NC}"
    echo -e "${BLUE}===================================================${NC}"
    echo ""
    echo -e "Service endpoints:"
    echo -e "  - ChromaDB:  ${BLUE}http://localhost:8000${NC}"
    echo -e "  - Weaviate:  ${BLUE}http://localhost:8080${NC}"
    echo -e "  - FalkorDB:  ${BLUE}localhost:6379${NC}"
    echo ""
    echo -e "Next steps:"
    echo -e "  1. Verify Python setup: ${YELLOW}python verify_setup.py${NC}"
    echo -e "  2. Run the benchmark:   ${YELLOW}python benchmark.py${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}  ❌ Some services are not healthy               ${NC}"
    echo -e "${BLUE}===================================================${NC}"
    echo ""
    echo -e "Troubleshooting:"
    echo -e "  1. Check service logs:   ${YELLOW}docker-compose logs${NC}"
    echo -e "  2. Restart services:     ${YELLOW}./scripts/stop_services.sh && ./scripts/start_services.sh${NC}"
    echo -e "  3. Check Docker status:  ${YELLOW}docker-compose ps${NC}"
    echo ""
    exit 1
fi
