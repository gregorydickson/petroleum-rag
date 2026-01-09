# Service Management Scripts

This directory contains convenience scripts for managing the Docker services required by the Petroleum RAG Benchmark.

## Scripts

### start_services.sh

Starts all required Docker services (ChromaDB, Weaviate, FalkorDB).

**Usage:**
```bash
./scripts/start_services.sh
```

**Features:**
- Checks Docker is running
- Starts all services using docker-compose
- Waits for services to become healthy
- Displays service endpoints
- Provides helpful next steps

**Exit codes:**
- 0: All services started successfully
- 1: Docker not running or services failed to start

### stop_services.sh

Stops all Docker services.

**Usage:**
```bash
# Stop services (keep data)
./scripts/stop_services.sh

# Stop services and remove all data volumes
./scripts/stop_services.sh --remove-volumes
./scripts/stop_services.sh -v
```

**Features:**
- Gracefully stops all containers
- Optional data volume removal with confirmation
- Provides helpful next steps

**Options:**
- `--remove-volumes` or `-v`: Remove all data volumes (requires confirmation)

**Exit codes:**
- 0: Services stopped successfully
- 1: Docker not running or docker-compose.yml not found

### verify_services.sh

Verifies all Docker services are running and responding correctly.

**Usage:**
```bash
./scripts/verify_services.sh
```

**Checks:**
- Docker daemon is running
- All containers are up
- Ports are accessible (8000, 8080, 6379)
- Service-specific health checks:
  - ChromaDB: Heartbeat API endpoint
  - Weaviate: Ready API endpoint
  - FalkorDB: Redis PING + graph module loaded

**Features:**
- Comprehensive health checks
- Clear success/failure indicators
- Helpful troubleshooting suggestions
- Color-coded output

**Exit codes:**
- 0: All services are healthy
- 1: One or more services are unhealthy

## Service Details

### ChromaDB (port 8000)

Vector database for pure similarity search.

- **Container:** petroleum-rag-chroma
- **Image:** chromadb/chroma:latest
- **Health check:** `http://localhost:8000/api/v1/heartbeat`
- **Data volume:** chroma_data

### Weaviate (port 8080)

Hybrid search engine combining vector similarity and BM25 keyword search.

- **Container:** petroleum-rag-weaviate
- **Image:** semitechnologies/weaviate:1.24.4
- **Health check:** `http://localhost:8080/v1/.well-known/ready`
- **Data volume:** weaviate_data

### FalkorDB (port 6379)

Graph database for graph-based retrieval with vector embeddings.

- **Container:** petroleum-rag-falkordb
- **Image:** falkordb/falkordb:latest
- **Protocol:** Redis
- **Health check:** `redis-cli ping` + graph module check
- **Data volume:** falkordb_data

## Common Workflows

### First-time Setup

```bash
# Start services
./scripts/start_services.sh

# Verify everything is working
./scripts/verify_services.sh

# Run Python setup verification
python verify_setup.py
```

### Daily Usage

```bash
# Start services
./scripts/start_services.sh

# Run your benchmark
python benchmark.py

# Stop services when done
./scripts/stop_services.sh
```

### Troubleshooting

```bash
# Check service status
./scripts/verify_services.sh

# View logs for all services
docker-compose logs -f

# View logs for specific service
docker-compose logs -f chroma
docker-compose logs -f weaviate
docker-compose logs -f falkordb

# Restart everything
./scripts/stop_services.sh
./scripts/start_services.sh

# Clean slate (removes all data)
./scripts/stop_services.sh --remove-volumes
./scripts/start_services.sh
```

### Data Management

```bash
# Check volume sizes
docker volume ls
docker system df -v

# Backup data volumes
docker run --rm -v petroleum-rag_chroma_data:/data -v $(pwd)/backups:/backup alpine tar czf /backup/chroma_data.tar.gz -C /data .

# Remove old data and start fresh
./scripts/stop_services.sh --remove-volumes
./scripts/start_services.sh
```

## Requirements

- Docker Desktop or Docker Engine
- docker-compose
- bash shell
- curl (for health checks)
- nc (netcat) for port checking

## Troubleshooting

### Services won't start

1. Check Docker is running: `docker info`
2. Check ports are available: `lsof -i :8000,8080,6379`
3. Check disk space: `docker system df`
4. View service logs: `docker-compose logs`

### Services start but fail health checks

1. Wait longer - services may need more time to initialize
2. Check service logs: `docker-compose logs [service-name]`
3. Check system resources (CPU, memory)
4. Restart services: `./scripts/stop_services.sh && ./scripts/start_services.sh`

### Port conflicts

If ports 8000, 8080, or 6379 are already in use:

1. Find the conflicting process: `lsof -i :8000`
2. Stop the conflicting service or change ports in `docker-compose.yml`
3. Update `config.py` if you change ports

### Data corruption

If you suspect data corruption:

```bash
# Remove all data and start fresh
./scripts/stop_services.sh --remove-volumes
./scripts/start_services.sh
```

## Notes

- All scripts must be run from the project root directory
- Scripts use color output for better readability
- Health checks include 30-second timeouts
- Data persists across restarts unless explicitly removed
- Network is named `petroleum-rag-network` for isolation
