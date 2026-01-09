#!/bin/bash

# Stop Services Script for Petroleum RAG Benchmark
# This script stops all Docker services and optionally removes volumes

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Petroleum RAG Benchmark - Stopping Services  ${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Error: Docker is not running${NC}"
    echo "Please start Docker Desktop and try again"
    exit 1
fi

# Check if docker-compose.yml exists
if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}❌ Error: docker-compose.yml not found${NC}"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Parse command line arguments
REMOVE_VOLUMES=false
if [ "$1" == "--remove-volumes" ] || [ "$1" == "-v" ]; then
    REMOVE_VOLUMES=true
    echo -e "${YELLOW}⚠️  Warning: This will remove all data volumes${NC}"
    echo -e "${YELLOW}⚠️  All stored data will be permanently deleted${NC}"
    echo ""
    read -p "Are you sure you want to continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Operation cancelled"
        exit 0
    fi
fi

# Stop services
echo -e "${BLUE}Stopping Docker services...${NC}"
docker-compose down

if [ "$REMOVE_VOLUMES" = true ]; then
    echo ""
    echo -e "${BLUE}Removing data volumes...${NC}"
    docker-compose down -v
    echo -e "${GREEN}✓ Volumes removed${NC}"
fi

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  All services stopped successfully!          ${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""

if [ "$REMOVE_VOLUMES" = false ]; then
    echo -e "To remove data volumes (delete all stored data):"
    echo -e "  ${YELLOW}./scripts/stop_services.sh --remove-volumes${NC}"
    echo ""
fi

echo -e "To start services again:"
echo -e "  ${YELLOW}./scripts/start_services.sh${NC}"
echo ""
