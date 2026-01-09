# User Guide

Complete step-by-step guide for using the Petroleum RAG Benchmark system.

## Table of Contents

- [Getting Started](#getting-started)
- [Basic Usage](#basic-usage)
- [Adding Your Documents](#adding-your-documents)
- [Customizing Queries](#customizing-queries)
- [Interpreting Results](#interpreting-results)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)

## Getting Started

### Prerequisites Check

Before starting, ensure you have:

1. **Python 3.11+**
   ```bash
   python --version
   # Should show: Python 3.11.x or higher
   ```

2. **Docker Desktop**
   ```bash
   docker --version
   # Should show: Docker version 20.x or higher
   ```

3. **API Keys**
   - Anthropic API key (for Claude)
   - OpenAI API key (for embeddings)
   - LlamaParse API key
   - Optional: Google Cloud credentials for Vertex Document AI

### Installation Steps

#### 1. Clone and Setup Environment

```bash
# Navigate to your projects directory
cd ~/projects

# Clone the repository (if not already cloned)
# git clone <repository-url>
cd petroleum-rag

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate

# Verify activation (should show path to venv)
which python
```

#### 2. Install Dependencies

```bash
# Install the package
pip install -e .

# Verify installation
python -c "import parsers, storage, evaluation; print('All imports successful!')"
```

#### 3. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Open .env in your editor
nano .env  # or vim, code, etc.
```

Add your API keys to `.env`:
```bash
ANTHROPIC_API_KEY=sk-ant-your-key-here
OPENAI_API_KEY=sk-your-key-here
LLAMA_CLOUD_API_KEY=llx-your-key-here
```

**Getting API Keys**:
- **Anthropic**: https://console.anthropic.com/
- **OpenAI**: https://platform.openai.com/api-keys
- **LlamaParse**: https://cloud.llamaindex.ai/

#### 4. Start Docker Services

```bash
# Start all services (ChromaDB, Weaviate, FalkorDB)
docker-compose up -d

# Wait 10-15 seconds for services to initialize

# Verify all services are running
docker-compose ps

# Should show all 3 services as "Up"
```

**Verify Service Health**:
```bash
# ChromaDB
curl http://localhost:8000/api/v1/heartbeat
# Should return: {"nanosecond heartbeat": ...}

# Weaviate
curl http://localhost:8080/v1/.well-known/ready
# Should return: {"status": "ok"}

# FalkorDB
redis-cli -p 6379 ping
# Should return: PONG
```

#### 5. Run Setup Verification

```bash
python verify_setup.py
```

Expected output:
```
✓ Python 3.11+
✓ All dependencies installed
✓ ChromaDB accessible at http://localhost:8000
✓ Weaviate accessible at http://localhost:8080
✓ FalkorDB accessible at localhost:6379
✓ Anthropic API key configured
✓ OpenAI API key configured
✓ LlamaParse API key configured

Setup complete! Ready to run benchmarks.
```

## Basic Usage

### Quick Start Example

This is the fastest way to run a complete benchmark:

```bash
# 1. Add a sample PDF (replace with your document)
cp ~/Documents/petroleum-report.pdf data/input/

# 2. Run the benchmark (takes 60-100 minutes)
python benchmark.py

# 3. Generate analysis and charts
python analyze_results.py

# 4. View results interactively
streamlit run demo_app.py
```

### Understanding the Benchmark Process

The benchmark runs through these stages:

1. **Parsing** (2-5 min): Extracts content from PDFs using 4 different parsers
2. **Storage** (1-2 min): Stores parsed content in 3 different databases
3. **Querying** (45-90 min): Tests retrieval and answer generation on 12 combinations
4. **Analysis** (1-2 min): Calculates metrics and generates visualizations

**Progress Indicators**: The benchmark shows progress bars for each stage:
```
Parsing documents: 100%|████████████| 4/4 [02:30<00:00]
Storing in backends: 100%|██████████| 12/12 [01:45<00:00]
Running queries: 100%|█████████████| 180/180 [87:23<00:00]
```

## Adding Your Documents

### Document Requirements

**Supported Formats**:
- PDF (primary format)
- DOCX (experimental)
- TXT (plain text)

**Document Characteristics**:
- **Size**: Up to 50 MB per file
- **Pages**: 1-500 pages (optimal: 10-100)
- **Content**: Technical documents with tables, figures, text
- **Language**: English (primary), other languages may work

**Best Results**:
- High-quality PDFs (not scanned images without OCR)
- Clear table structures
- Standard fonts and layouts
- Properly formatted sections

### Adding Documents

#### Single Document

```bash
# Copy to input directory
cp /path/to/your/document.pdf data/input/

# Verify it's there
ls -lh data/input/
```

#### Multiple Documents

```bash
# Copy all PDFs from a directory
cp ~/Documents/petroleum-reports/*.pdf data/input/

# Or use find for recursive copy
find ~/Documents/petroleum-reports -name "*.pdf" -exec cp {} data/input/ \;
```

#### Document Organization

The benchmark processes all PDFs in `data/input/`. To organize:

```bash
data/input/
├── drilling-report-2024.pdf
├── production-analysis.pdf
└── reservoir-study.pdf
```

**Tip**: Start with one document to verify setup, then add more.

### Document Naming

Use descriptive names for easier result interpretation:
- Good: `drilling-report-q4-2024.pdf`
- Bad: `doc1.pdf`, `final_final_v2.pdf`

## Customizing Queries

### Query Structure

Queries are defined in `evaluation/queries.json`. Each query has:

```json
{
  "query_id": "unique_identifier",
  "query": "What is the optimal drilling fluid density?",
  "ground_truth_answer": "The expected answer...",
  "relevant_element_ids": ["doc1_table_3", "doc1_para_45"],
  "query_type": "numerical",
  "difficulty": "medium",
  "notes": "Optional context about this query"
}
```

**Fields Explained**:
- `query_id`: Unique identifier (e.g., "query_001", "table_extraction_1")
- `query`: The actual question to ask
- `ground_truth_answer`: Expected correct answer (for evaluation)
- `relevant_element_ids`: Which document elements contain the answer
- `query_type`: Category (see types below)
- `difficulty`: "easy", "medium", or "hard"
- `notes`: Optional notes for documentation

### Query Types

**Available Types**:
- `table`: Questions requiring table data
- `keyword`: Exact keyword or term lookup
- `semantic`: Conceptual understanding needed
- `multi_hop`: Requires connecting multiple sections
- `numerical`: Calculations or numerical comparisons
- `general`: General information retrieval

**Examples**:

**Table Query**:
```json
{
  "query_id": "table_001",
  "query": "What are the production rates for Well A in Q3 2024?",
  "query_type": "table",
  "difficulty": "easy"
}
```

**Semantic Query**:
```json
{
  "query_id": "semantic_001",
  "query": "What factors affect drilling efficiency?",
  "query_type": "semantic",
  "difficulty": "medium"
}
```

**Multi-hop Query**:
```json
{
  "query_id": "multihop_001",
  "query": "How does pressure affect flow rate and what are the economic implications?",
  "query_type": "multi_hop",
  "difficulty": "hard"
}
```

### Creating Custom Queries

#### Step 1: Read Your Document

First, understand what information your document contains.

#### Step 2: Create Queries File

```bash
# Copy the template
cp evaluation/queries.json evaluation/custom-queries.json

# Edit with your queries
code evaluation/custom-queries.json  # or nano, vim
```

#### Step 3: Add Your Queries

```json
{
  "queries": [
    {
      "query_id": "custom_001",
      "query": "Your question here?",
      "ground_truth_answer": "Expected answer",
      "query_type": "general",
      "difficulty": "medium"
    },
    {
      "query_id": "custom_002",
      "query": "Another question?",
      "ground_truth_answer": "Another answer",
      "query_type": "semantic",
      "difficulty": "easy"
    }
  ]
}
```

#### Step 4: Run with Custom Queries

```bash
python benchmark.py --queries evaluation/custom-queries.json
```

### Query Best Practices

**Good Queries**:
- Specific and clear
- Have a definite answer in the document
- Test different aspects (tables, concepts, facts)
- Vary in difficulty

**Examples**:
```
✓ "What is the API gravity of the crude oil in Well 5?"
✓ "What safety measures are recommended for H2S exposure?"
✓ "How does temperature affect viscosity according to Table 3?"
```

**Bad Queries**:
```
✗ "Tell me about oil" (too vague)
✗ "What is the meaning of life?" (not in document)
✗ "Is the report good?" (subjective)
```

### Query Templates

**Factual Extraction**:
```json
{
  "query": "What is the [metric] for [entity] in [time period]?",
  "query_type": "table",
  "difficulty": "easy"
}
```

**Comparison**:
```json
{
  "query": "Compare [A] and [B] in terms of [metric]",
  "query_type": "semantic",
  "difficulty": "medium"
}
```

**Causal**:
```json
{
  "query": "What causes [effect] according to the document?",
  "query_type": "semantic",
  "difficulty": "medium"
}
```

**Calculation**:
```json
{
  "query": "Calculate [derived metric] based on [given data]",
  "query_type": "numerical",
  "difficulty": "hard"
}
```

## Interpreting Results

### Output Files

After running the benchmark, check `data/results/`:

```
data/results/
├── raw_results.json          # Complete data (all queries, metrics)
├── comparison.csv            # Summary table (easy to open in Excel)
├── REPORT.md                 # Human-readable analysis
└── charts/
    ├── heatmap_performance.png      # Parser × Storage matrix
    ├── metric_bars.png              # Metric comparison bars
    ├── timing_comparison.png        # Speed analysis
    ├── radar_top3.png               # Top 3 combinations
    └── precision_recall.png         # Precision-Recall curves
```

### Understanding the Winner

The `REPORT.md` starts with the winner:

```markdown
# Benchmark Results Report

## Winner: docling_weaviate
**Composite Score**: 0.847

The winning combination demonstrates:
- Excellent hybrid search capabilities
- Strong table extraction
- Fast retrieval times (avg 45ms)
- High answer quality (0.89)
```

**What the Score Means**:
- **0.9-1.0**: Excellent (production-ready)
- **0.8-0.9**: Very Good (recommended)
- **0.7-0.8**: Good (acceptable)
- **0.6-0.7**: Fair (needs improvement)
- **<0.6**: Poor (not recommended)

### Reading the Comparison Table

Open `comparison.csv` in Excel or view in terminal:

```bash
column -t -s, data/results/comparison.csv | less -S
```

**Key Columns**:
- `combination`: Parser_Storage combination
- `composite_score`: Overall performance (0-1)
- `precision@5`: Accuracy of top 5 results
- `ndcg@5`: Ranking quality
- `answer_correctness`: How good are the answers
- `avg_retrieval_time`: Speed in seconds

**Example**:
```
combination          composite  precision@5  answer_correctness  speed_ms
docling_weaviate     0.847      0.92         0.89               45
llamaparse_falkordb  0.831      0.88         0.91               62
vertex_chroma        0.816      0.90         0.84               38
```

**Interpretation**:
- **docling_weaviate**: Best overall (balanced accuracy and speed)
- **llamaparse_falkordb**: Best answer quality, slightly slower
- **vertex_chroma**: Fastest, but lower answer quality

### Visualizations

#### 1. Heatmap (heatmap_performance.png)

Shows composite scores for all 12 combinations:

```
                ChromaDB   Weaviate   FalkorDB
LlamaParse        0.78      0.82       0.83
Docling           0.80      0.85       0.81
PageIndex         0.75      0.79       0.77
Vertex            0.82      0.81       0.78
```

**How to Read**:
- Darker = better performance
- Look for darkest cell = best combination
- Compare rows = parser performance
- Compare columns = storage performance

#### 2. Metric Bars (metric_bars.png)

Bar chart comparing top 3 combinations across key metrics.

**What to Look For**:
- Consistent performance across metrics
- Trade-offs (high accuracy but slow?)
- Strengths and weaknesses

#### 3. Timing Comparison (timing_comparison.png)

Shows retrieval and generation times.

**Analysis**:
- Which combinations are fastest?
- Is speed consistent?
- Speed vs accuracy trade-offs

#### 4. Radar Chart (radar_top3.png)

Multi-dimensional view of top 3 combinations.

**Dimensions**:
- Precision
- Recall
- NDCG
- Answer Quality
- Speed

**How to Read**:
- Larger area = better overall
- Look for balance vs spikes
- Identify strengths/weaknesses

#### 5. Precision-Recall (precision_recall.png)

Performance at different top-K values.

**Analysis**:
- How does performance change with K?
- Which combination is most stable?
- Trade-offs between precision and recall

### Choosing Your Configuration

**Decision Guide**:

**Prioritize Accuracy**:
- Choose highest composite score
- Example: `docling_weaviate` (0.847)

**Prioritize Speed**:
- Choose fastest with acceptable accuracy
- Example: `vertex_chroma` (38ms, 0.816 score)

**Prioritize Cost**:
- Avoid cloud parsers (LlamaParse, Vertex)
- Choose: `docling_*` or `pageindex_*`

**Prioritize Answer Quality**:
- Sort by `answer_correctness`
- Example: `llamaparse_falkordb` (0.91)

**Balanced Approach**:
- Use composite score (already balanced)
- Top 3 are usually good choices

## Advanced Usage

### Running Specific Combinations

Test only certain parser-storage combinations:

```bash
# Only test Docling parser with all storage
python benchmark.py --parsers docling

# Only test ChromaDB with all parsers
python benchmark.py --storage chroma

# Specific combination
python benchmark.py --parsers docling llamaparse --storage weaviate falkordb
```

### Batch Processing Multiple Documents

```bash
# Process all PDFs in a directory
for pdf in data/input/*.pdf; do
    python benchmark.py --input-dir "$(dirname "$pdf")"
done
```

### Custom Configuration

Create a custom `.env` file:

```bash
# Copy base config
cp .env .env.experiment1

# Edit settings
nano .env.experiment1

# Run with custom config
export ENV_FILE=.env.experiment1
python benchmark.py
```

### Adjusting Chunking Strategy

Edit `.env`:

```bash
# Smaller chunks (more granular)
CHUNK_SIZE=500
CHUNK_OVERLAP=100

# Larger chunks (more context)
CHUNK_SIZE=2000
CHUNK_OVERLAP=400
```

**Impact**:
- Smaller chunks: Better precision, potentially lower recall
- Larger chunks: More context, potentially more noise

### Adjusting Retrieval Parameters

```bash
# Retrieve more chunks
RETRIEVAL_TOP_K=10

# Lower similarity threshold
RETRIEVAL_MIN_SCORE=0.3
```

**Impact**:
- Higher top-K: Better recall, more processing time
- Lower min score: More results, potentially less relevant

### Parallel vs Sequential

```bash
# Disable parallel processing (for debugging)
BENCHMARK_PARALLEL_PARSERS=false
BENCHMARK_PARALLEL_STORAGE=false

# Enable (default, faster)
BENCHMARK_PARALLEL_PARSERS=true
BENCHMARK_PARALLEL_STORAGE=true
```

### Saving Costs

**Reduce API Calls**:
```bash
# Use only free parsers
python benchmark.py --parsers docling pageindex

# Reduce query count
# Edit evaluation/queries.json to include fewer queries

# Disable intermediate saves
BENCHMARK_SAVE_INTERMEDIATE_RESULTS=false
```

**Estimate Costs**:
- OpenAI embeddings: ~$0.10 per 1M tokens
- LlamaParse: $0.003 per page
- Vertex Document AI: $1.50 per 1000 pages
- Claude (evaluation): ~$3 per 1M tokens

### Export Results

```bash
# Export to Excel-compatible CSV
python analyze_results.py --output-dir exports/experiment1/

# Export visualizations only
python analyze_results.py --charts-only

# Export with custom format
python -c "
import json
import pandas as pd
with open('data/results/raw_results.json') as f:
    data = json.load(f)
df = pd.DataFrame(data['results'])
df.to_excel('results.xlsx', index=False)
"
```

## Troubleshooting

### Common Issues

#### 1. Docker Services Not Starting

**Symptom**: `docker-compose ps` shows services as "Exited" or "Restarting"

**Solutions**:

```bash
# Check Docker is running
docker info

# Check for port conflicts
lsof -i :8000,8080,6379

# Stop and restart
docker-compose down
docker-compose up -d

# Check logs
docker-compose logs chromadb
docker-compose logs weaviate
docker-compose logs falkordb

# Remove volumes and restart (WARNING: deletes data)
docker-compose down -v
docker-compose up -d
```

#### 2. API Key Errors

**Symptom**: "Authentication failed" or "Invalid API key"

**Solutions**:

```bash
# Check .env file exists
cat .env | grep API_KEY

# Verify keys are not empty
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "ANTHROPIC_API_KEY is not set"
fi

# Test API connectivity
python -c "
import anthropic
client = anthropic.Anthropic(api_key='your-key')
print('Anthropic OK')
"
```

#### 3. Out of Memory

**Symptom**: Process killed or "MemoryError"

**Solutions**:

```bash
# Reduce batch size
EMBEDDING_BATCH_SIZE=50

# Disable parallel processing
BENCHMARK_PARALLEL_PARSERS=false

# Process one document at a time
python benchmark.py --parsers docling  # One parser at a time

# Increase swap space (Linux)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 4. Slow Performance

**Symptom**: Benchmark taking much longer than expected

**Solutions**:

```bash
# Enable parallel processing (if disabled)
BENCHMARK_PARALLEL_PARSERS=true
BENCHMARK_PARALLEL_STORAGE=true

# Reduce query count
# Edit evaluation/queries.json

# Use faster parsers
python benchmark.py --parsers docling pageindex

# Check network connectivity
ping api.openai.com
ping api.anthropic.com
```

#### 5. Parser Failures

**Symptom**: "Parser failed" errors in logs

**Solutions**:

**LlamaParse**:
```bash
# Check API key
echo $LLAMA_CLOUD_API_KEY

# Check document size (<50 MB)
ls -lh data/input/

# Try different document
```

**Vertex Document AI**:
```bash
# Check credentials
echo $GOOGLE_APPLICATION_CREDENTIALS
cat $GOOGLE_APPLICATION_CREDENTIALS | jq .

# Check project ID
gcloud config get-value project

# Verify API is enabled
gcloud services list --enabled | grep documentai
```

#### 6. Storage Connection Failures

**Symptom**: "Failed to connect to storage backend"

**Solutions**:

**ChromaDB**:
```bash
curl http://localhost:8000/api/v1/heartbeat
docker logs petroleum-rag-chroma
```

**Weaviate**:
```bash
curl http://localhost:8080/v1/.well-known/ready
docker logs petroleum-rag-weaviate
```

**FalkorDB**:
```bash
redis-cli -p 6379 ping
docker logs petroleum-rag-falkordb
```

### Debug Mode

Enable verbose logging:

```bash
# Set debug mode in .env
LOG_LEVEL=DEBUG

# Run with verbose output
python benchmark.py 2>&1 | tee benchmark-debug.log
```

### Getting Help

1. **Check logs**: `tail -f benchmark.log`
2. **Search issues**: Check GitHub issues for similar problems
3. **Enable debug mode**: Set `LOG_LEVEL=DEBUG` in `.env`
4. **Check documentation**: Review relevant docs in `docs/`
5. **Run verification**: `python verify_setup.py`

## FAQ

### General Questions

**Q: How long does a benchmark take?**
A: For 1 document with 15 queries: 60-100 minutes. Scales linearly with document count and query count.

**Q: Can I run benchmarks in parallel?**
A: Parsers and storage run in parallel within a benchmark. Multiple benchmark runs should be sequential to avoid resource conflicts.

**Q: Do I need all 4 parsers?**
A: No, you can run with any subset: `python benchmark.py --parsers docling pageindex`

**Q: What if my document isn't in English?**
A: The system primarily supports English. Other languages may work but aren't officially tested.

**Q: Can I use my own LLM for evaluation?**
A: Currently Claude Sonnet 4 is used for evaluation. Modify `evaluation/evaluator.py` to use a different LLM.

### Cost Questions

**Q: How much does it cost to run a benchmark?**
A: Approximate costs for 1 document, 15 queries:
- OpenAI embeddings: $0.50-1.00
- Claude evaluation: $5-10
- LlamaParse (if used): $0.10-0.50
- Vertex AI (if used): $0.50-2.00
- **Total**: $6-14 per full benchmark

**Q: How can I reduce costs?**
A:
- Use only free parsers (Docling, PageIndex)
- Reduce query count
- Disable intermediate saves
- Use cached embeddings

**Q: Are there any free tiers?**
A: OpenAI and Anthropic offer free credits for new accounts. Check their websites for current offers.

### Technical Questions

**Q: Can I add my own parser?**
A: Yes! See [ARCHITECTURE.md](ARCHITECTURE.md#adding-a-new-parser) for detailed instructions.

**Q: Can I use a different embedding model?**
A: Yes, modify `EMBEDDING_MODEL` in `.env` and update `embeddings/embedder.py`.

**Q: How are embeddings stored?**
A: Embeddings are generated once and stored in each backend. They're not cached across benchmark runs.

**Q: Can I export results to Excel?**
A: Yes, `comparison.csv` opens in Excel. For custom exports, see [Advanced Usage](#export-results).

**Q: What happens if a query fails?**
A: The benchmark continues with other queries. Failed queries are logged and marked in results.

### Results Questions

**Q: What if all scores are low?**
A: Check that:
- Queries match document content
- Ground truth answers are correct
- Document quality is good (not scanned images)

**Q: Why do scores vary between runs?**
A: LLM-based evaluation has some variance. Run multiple times and average for more stable results.

**Q: Which metric is most important?**
A: Use composite score for overall comparison. Focus on specific metrics (precision, answer correctness) based on your priorities.

**Q: Can I customize metric weights?**
A: Yes, modify `evaluation/metrics.py` to adjust composite score calculation.

### Deployment Questions

**Q: Can I deploy this in production?**
A: Yes! See [DEPLOYMENT.md](DEPLOYMENT.md) for GCP Cloud Run deployment guide.

**Q: Does it support distributed execution?**
A: Not currently. All processing happens on one machine. Storage backends can be remote.

**Q: Can I use managed vector databases?**
A: Yes, modify connection settings in `.env` to point to managed instances (e.g., Weaviate Cloud).

**Q: How do I monitor production deployments?**
A: The system logs to files and stdout. Integrate with your monitoring stack (e.g., GCP Cloud Logging).

---

**Need more help?**
- Check [ARCHITECTURE.md](ARCHITECTURE.md) for system design details
- See [API_REFERENCE.md](API_REFERENCE.md) for API documentation
- Review [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment
- Open an issue on GitHub for additional support
