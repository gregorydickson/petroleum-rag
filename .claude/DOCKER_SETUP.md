# Docker Setup Quick Reference

Agent 11 completed Docker and deployment configuration.

## Files Created

```
petroleum-rag/
├── docker-compose.yml                    # Docker services configuration
└── scripts/
    ├── start_services.sh                 # Start all services
    ├── stop_services.sh                  # Stop services
    ├── verify_services.sh                # Verify service health
    └── README.md                         # Detailed documentation
```

## Quick Commands

```bash
# Start all services
./scripts/start_services.sh

# Verify services are healthy
./scripts/verify_services.sh

# Stop services (keep data)
./scripts/stop_services.sh

# Stop services and remove all data
./scripts/stop_services.sh --remove-volumes
```

## Services

| Service   | Port | URL                       | Purpose                    |
|-----------|------|---------------------------|----------------------------|
| ChromaDB  | 8000 | http://localhost:8000     | Pure vector search         |
| Weaviate  | 8080 | http://localhost:8080     | Hybrid vector + BM25       |
| FalkorDB  | 6379 | localhost:6379            | Graph + vector embeddings  |

## Data Volumes

- `chroma_data` - ChromaDB persistent storage
- `weaviate_data` - Weaviate indexes and data
- `falkordb_data` - FalkorDB graph data

All volumes persist across container restarts.

## Troubleshooting

```bash
# View logs
docker-compose logs -f

# Check container status
docker-compose ps

# Restart everything
./scripts/stop_services.sh && ./scripts/start_services.sh

# Check ports
lsof -i :8000,8080,6379
```

## Next Steps

1. Start services: `./scripts/start_services.sh`
2. Verify health: `./scripts/verify_services.sh`
3. Run setup verification: `python verify_setup.py`
4. Implement storage backends (Agents 6, 7, 8)
