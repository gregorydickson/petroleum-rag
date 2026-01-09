#!/bin/bash

# End-to-End Test Runner for Petroleum RAG Benchmark
# This script orchestrates the complete E2E test suite:
# 1. Checks for Docker
# 2. Starts required services
# 3. Generates sample PDF (if needed)
# 4. Runs pytest E2E tests
# 5. Shows summary
# 6. Cleans up (optional)

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
E2E_DIR="$PROJECT_ROOT/tests/e2e"

# Configuration
CLEANUP_AFTER=false
SKIP_DOCKER=false
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cleanup)
            CLEANUP_AFTER=true
            shift
            ;;
        --skip-docker)
            SKIP_DOCKER=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --cleanup        Stop Docker services after tests"
            echo "  --skip-docker    Skip Docker service checks/startup"
            echo "  --verbose, -v    Show verbose output"
            echo "  --help, -h       Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

# Print header
echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}  PETROLEUM RAG BENCHMARK - E2E TEST SUITE${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

# Check prerequisites
echo -e "${YELLOW}[1/7]${NC} Checking prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Python: $(python3 --version)"

# Check pytest
if ! python3 -c "import pytest" &> /dev/null; then
    echo -e "${RED}✗ pytest not installed${NC}"
    echo "Install with: pip install pytest pytest-asyncio"
    exit 1
fi
echo -e "${GREEN}✓${NC} pytest installed"

# Check Docker (unless skipped)
if [ "$SKIP_DOCKER" = false ]; then
    if ! command -v docker &> /dev/null; then
        echo -e "${YELLOW}⚠${NC}  Docker not found - some tests may be skipped"
        SKIP_DOCKER=true
    else
        echo -e "${GREEN}✓${NC} Docker: $(docker --version | cut -d' ' -f3 | cut -d',' -f1)"
    fi

    if ! command -v docker-compose &> /dev/null; then
        echo -e "${YELLOW}⚠${NC}  docker-compose not found - some tests may be skipped"
        SKIP_DOCKER=true
    else
        echo -e "${GREEN}✓${NC} docker-compose: $(docker-compose --version | cut -d' ' -f4 | cut -d',' -f1)"
    fi
fi

echo ""

# Generate sample PDF if needed
echo -e "${YELLOW}[2/7]${NC} Checking sample PDF..."
SAMPLE_PDF="$E2E_DIR/sample_petroleum_doc.pdf"

if [ ! -f "$SAMPLE_PDF" ]; then
    echo -e "${BLUE}→${NC} Generating sample petroleum engineering document..."
    cd "$PROJECT_ROOT"
    python3 "$E2E_DIR/generate_sample_doc.py"

    if [ ! -f "$SAMPLE_PDF" ]; then
        echo -e "${RED}✗ Failed to generate sample PDF${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓${NC} Sample PDF generated"
else
    PDF_SIZE=$(ls -lh "$SAMPLE_PDF" | awk '{print $5}')
    echo -e "${GREEN}✓${NC} Sample PDF exists ($PDF_SIZE)"
fi

echo ""

# Start Docker services (unless skipped)
if [ "$SKIP_DOCKER" = false ]; then
    echo -e "${YELLOW}[3/7]${NC} Starting Docker services..."
    cd "$PROJECT_ROOT"

    # Check if services are already running
    if docker-compose ps | grep -q "Up"; then
        echo -e "${GREEN}✓${NC} Docker services already running"
    else
        echo -e "${BLUE}→${NC} Starting ChromaDB, Weaviate, and FalkorDB..."
        docker-compose up -d

        echo -e "${BLUE}→${NC} Waiting for services to be healthy (15 seconds)..."
        sleep 15

        # Verify services
        if docker-compose ps | grep -q "Up"; then
            echo -e "${GREEN}✓${NC} Docker services started successfully"
        else
            echo -e "${YELLOW}⚠${NC}  Some services may not be running - tests will continue"
        fi
    fi
else
    echo -e "${YELLOW}[3/7]${NC} Skipping Docker service management"
fi

echo ""

# Check API keys
echo -e "${YELLOW}[4/7]${NC} Checking configuration..."
cd "$PROJECT_ROOT"

if [ -f ".env" ]; then
    echo -e "${GREEN}✓${NC} .env file found"

    # Check for critical API keys
    MISSING_KEYS=()

    if ! grep -q "OPENAI_API_KEY=sk-" .env 2>/dev/null; then
        MISSING_KEYS+=("OPENAI_API_KEY")
    fi

    if ! grep -q "ANTHROPIC_API_KEY=sk-" .env 2>/dev/null; then
        MISSING_KEYS+=("ANTHROPIC_API_KEY")
    fi

    if [ ${#MISSING_KEYS[@]} -gt 0 ]; then
        echo -e "${YELLOW}⚠${NC}  Missing API keys: ${MISSING_KEYS[*]}"
        echo "    Some tests may be skipped"
    else
        echo -e "${GREEN}✓${NC} API keys configured"
    fi
else
    echo -e "${YELLOW}⚠${NC}  .env file not found - using environment variables"
fi

echo ""

# Run E2E tests
echo -e "${YELLOW}[5/7]${NC} Running E2E tests..."
cd "$PROJECT_ROOT"

# Build pytest command
PYTEST_CMD="python3 -m pytest tests/e2e/test_full_pipeline.py"

if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -v -s"
else
    PYTEST_CMD="$PYTEST_CMD -v"
fi

PYTEST_CMD="$PYTEST_CMD --tb=short --color=yes"

echo -e "${BLUE}→${NC} Running: $PYTEST_CMD"
echo ""

# Run tests and capture exit code
set +e
START_TIME=$(date +%s)
$PYTEST_CMD
TEST_EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
set -e

echo ""

# Show results
echo -e "${YELLOW}[6/7]${NC} Test Results Summary"
echo ""
echo "Duration: ${DURATION}s"

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED${NC}"
else
    echo -e "${RED}✗ SOME TESTS FAILED (exit code: $TEST_EXIT_CODE)${NC}"
fi

echo ""

# Cleanup (optional)
if [ "$CLEANUP_AFTER" = true ]; then
    echo -e "${YELLOW}[7/7]${NC} Cleaning up..."

    if [ "$SKIP_DOCKER" = false ]; then
        echo -e "${BLUE}→${NC} Stopping Docker services..."
        cd "$PROJECT_ROOT"
        docker-compose down
        echo -e "${GREEN}✓${NC} Docker services stopped"
    fi

    echo -e "${GREEN}✓${NC} Cleanup complete"
else
    echo -e "${YELLOW}[7/7]${NC} Skipping cleanup (use --cleanup to stop services)"

    if [ "$SKIP_DOCKER" = false ]; then
        echo ""
        echo "To stop services manually, run:"
        echo "  cd $PROJECT_ROOT && docker-compose down"
    fi
fi

echo ""
echo -e "${BLUE}================================================================${NC}"

# Exit with test exit code
exit $TEST_EXIT_CODE
