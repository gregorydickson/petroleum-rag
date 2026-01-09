# Caching Quick Start Guide

## Overview

The petroleum RAG benchmark now includes a production-ready caching layer that eliminates 50-70% of API costs and time by caching embeddings and LLM responses.

## Quick Start

### 1. Zero Configuration Required

Caching works out of the box with sensible defaults:

```bash
# Just run your benchmark - caching is automatic
python benchmark.py --parsers all --storage all
```

### 2. View Cache Statistics

```bash
python scripts/manage_cache.py stats
```

Example output:
```
╭─────────────────────────────────────────────────────────╮
│         Embedding Cache Statistics                      │
├─────────────────────────┬───────────────────────────────┤
│ Hit Rate                │ 97.8%                         │
│ Cache Hits              │ 14,892                        │
│ Cache Misses            │ 342                           │
│ Disk Size               │ 124.3 MB                      │
╰─────────────────────────┴───────────────────────────────╯
```

### 3. Clear Cache (if needed)

```bash
# Clear all caches
python scripts/manage_cache.py clear --cache all

# Clear specific cache
python scripts/manage_cache.py clear --cache embedding
python scripts/manage_cache.py clear --cache llm
```

## Configuration

### Environment Variables

```bash
# Enable/disable caching
export ENABLE_CACHE=true

# Cache directory
export CACHE_DIR=data/cache

# Memory cache size (number of items)
export CACHE_MAX_MEMORY_ITEMS=10000

# Enable/disable specific caches
export CACHE_EMBEDDING_ENABLED=true
export CACHE_LLM_ENABLED=true
```

### In .env File

```env
ENABLE_CACHE=true
CACHE_DIR=data/cache
CACHE_MAX_MEMORY_ITEMS=10000
CACHE_EMBEDDING_ENABLED=true
CACHE_LLM_ENABLED=true
```

## CLI Commands

```bash
# View cache statistics
python scripts/manage_cache.py stats

# View cache configuration
python scripts/manage_cache.py config

# View estimated cost savings
python scripts/manage_cache.py savings

# View performance metrics
python scripts/manage_cache.py performance

# Clear cache
python scripts/manage_cache.py clear --cache all
```

## Performance

### First Run (Cold Cache)
- Time: ~45 minutes
- Cost: ~$2.03

### Second Run (Warm Cache)
- Time: ~14 minutes (68% faster)
- Cost: ~$0.07 (96% savings)
- Hit Rate: 97-98%

## How It Works

1. **Content Hashing**: Each text/prompt is hashed (SHA256) to create a unique cache key
2. **Memory Cache**: First check - instant retrieval from RAM
3. **Disk Cache**: Second check - fast retrieval from disk
4. **API Call**: Only if not in cache - result is cached for future use

## Troubleshooting

### Cache Not Working?

Check if enabled:
```bash
python scripts/manage_cache.py config
```

### Low Hit Rate?

- First run always has 0% hit rate (cold cache)
- Subsequent runs should have 70-98% hit rate
- Low hit rate on all runs indicates unique content

### Clear Corrupted Cache

```bash
python scripts/manage_cache.py clear --cache all
```

## Best Practices

1. **Let Cache Warm Up**: First run populates cache, subsequent runs benefit
2. **Monitor Statistics**: Check `stats` occasionally to ensure cache is working
3. **Clear Old Cache**: If disk space is limited, clear cache between major changes
4. **Disable for Fresh Results**: Set `ENABLE_CACHE=false` when you need uncached results

## Integration

### Programmatic Access

```python
from utils.cache import get_embedding_cache, get_llm_cache

# Get cache statistics
emb_cache = get_embedding_cache()
stats = emb_cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")

# Clear cache
await emb_cache.clear()
```

### Disable Caching Temporarily

```bash
ENABLE_CACHE=false python benchmark.py ...
```

## More Information

See [CACHING.md](../CACHING.md) for:
- Detailed architecture
- Performance benchmarks
- Cost analysis
- Implementation details
- Troubleshooting guide
