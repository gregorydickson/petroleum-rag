"""Example script to test Weaviate hybrid search functionality.

This script demonstrates:
1. Connecting to Weaviate
2. Storing document chunks with embeddings
3. Performing hybrid search (vector + keyword/BM25)
4. Comparing semantic vs keyword matching

Prerequisites:
- Weaviate running on localhost:8080
- Run: docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:latest
"""

import asyncio

from models import DocumentChunk, RetrievalResult
from storage.weaviate_store import WeaviateStore


async def test_hybrid_search() -> None:
    """Test Weaviate hybrid search with petroleum engineering content."""
    print("=" * 80)
    print("Weaviate Hybrid Search Test")
    print("=" * 80)

    # Initialize Weaviate store
    store = WeaviateStore()
    print("\n1. Initializing Weaviate connection...")
    try:
        await store.initialize()
        print("   ✓ Connected to Weaviate")
    except Exception as e:
        print(f"   ✗ Failed to connect: {e}")
        print("\n   Make sure Weaviate is running:")
        print("   docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:latest")
        return

    # Health check
    is_healthy = await store.health_check()
    print(f"   ✓ Health check: {'Healthy' if is_healthy else 'Unhealthy'}")

    # Create sample petroleum engineering chunks
    chunks = [
        DocumentChunk(
            chunk_id="chunk_1",
            document_id="drilling_guide",
            content=(
                "Drilling operations require careful monitoring of mud weight and "
                "circulation rates. The drilling fluid density must be maintained "
                "between 9.0 and 12.0 ppg to prevent formation damage."
            ),
            metadata={"topic": "drilling", "page": "15"},
            chunk_index=0,
        ),
        DocumentChunk(
            chunk_id="chunk_2",
            document_id="drilling_guide",
            content=(
                "Hydraulic fracturing, also known as fracking, involves pumping "
                "high-pressure fluid into the formation to create fractures. "
                "Proppant materials like sand keep fractures open."
            ),
            metadata={"topic": "completion", "page": "42"},
            chunk_index=1,
        ),
        DocumentChunk(
            chunk_id="chunk_3",
            document_id="reservoir_engineering",
            content=(
                "Reservoir pressure maintenance is critical for optimal recovery. "
                "Water flooding and gas injection are common secondary recovery methods "
                "used to maintain pressure and sweep hydrocarbons."
            ),
            metadata={"topic": "reservoir", "page": "88"},
            chunk_index=0,
        ),
        DocumentChunk(
            chunk_id="chunk_4",
            document_id="production_ops",
            content=(
                "Production logging tools measure flow rates, fluid density, and "
                "temperature profiles. These measurements help identify producing zones "
                "and diagnose production problems."
            ),
            metadata={"topic": "production", "page": "123"},
            chunk_index=0,
        ),
        DocumentChunk(
            chunk_id="chunk_5",
            document_id="safety_manual",
            content=(
                "Blowout preventers (BOPs) are critical safety equipment that seal "
                "the wellbore in emergency situations. Regular BOP testing and "
                "maintenance is required by regulations."
            ),
            metadata={"topic": "safety", "page": "5"},
            chunk_index=0,
        ),
    ]

    # Generate simple mock embeddings (in real use, use OpenAI/etc)
    # Each chunk gets a unique embedding vector
    embeddings = [
        [0.1, 0.2, 0.3, 0.4, 0.5],  # drilling operations
        [0.2, 0.3, 0.4, 0.5, 0.6],  # hydraulic fracturing
        [0.3, 0.4, 0.5, 0.6, 0.7],  # reservoir pressure
        [0.4, 0.5, 0.6, 0.7, 0.8],  # production logging
        [0.5, 0.6, 0.7, 0.8, 0.9],  # blowout preventers
    ]

    # Store chunks
    print("\n2. Storing document chunks...")
    try:
        await store.store_chunks(chunks, embeddings)
        print(f"   ✓ Stored {len(chunks)} chunks")
    except Exception as e:
        print(f"   ✗ Failed to store chunks: {e}")
        return

    # Test 1: Semantic search (should find based on meaning)
    print("\n3. Test 1: Semantic Search")
    print("   Query: 'How do you prevent well blowouts?'")
    query1_embedding = [0.5, 0.6, 0.7, 0.8, 0.9]  # Similar to BOP chunk
    results1 = await store.retrieve(
        query="How do you prevent well blowouts?",
        query_embedding=query1_embedding,
        top_k=3,
    )
    print(f"   Found {len(results1)} results:")
    for result in results1:
        print(f"   - [{result.score:.3f}] {result.chunk_id}: {result.content[:80]}...")

    # Test 2: Keyword search (should find exact term match)
    print("\n4. Test 2: Keyword Search")
    print("   Query: 'fracturing proppant'")
    query2_embedding = [0.1, 0.1, 0.1, 0.1, 0.1]  # Generic embedding
    results2 = await store.retrieve(
        query="fracturing proppant",
        query_embedding=query2_embedding,
        top_k=3,
    )
    print(f"   Found {len(results2)} results:")
    for result in results2:
        print(f"   - [{result.score:.3f}] {result.chunk_id}: {result.content[:80]}...")

    # Test 3: Hybrid search with filter
    print("\n5. Test 3: Filtered Search")
    print("   Query: 'drilling' (filtered to drilling_guide document)")
    query3_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    results3 = await store.retrieve(
        query="drilling",
        query_embedding=query3_embedding,
        top_k=3,
        filters={"document_id": "drilling_guide"},
    )
    print(f"   Found {len(results3)} results:")
    for result in results3:
        print(
            f"   - [{result.score:.3f}] {result.chunk_id} (doc: {result.document_id}): "
            f"{result.content[:60]}..."
        )

    # Test 4: Show hybrid alpha effect
    print("\n6. Test 4: Alpha Parameter Effect")
    print("   Comparing different alpha values (vector vs keyword weight)")

    # Alpha = 1.0 (100% vector, 0% keyword)
    store.alpha = 1.0
    results_vector = await store.retrieve(
        query="mud weight circulation",
        query_embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
        top_k=2,
    )
    print(f"   Alpha=1.0 (pure vector): {len(results_vector)} results")

    # Alpha = 0.0 (0% vector, 100% keyword)
    store.alpha = 0.0
    results_keyword = await store.retrieve(
        query="mud weight circulation",
        query_embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
        top_k=2,
    )
    print(f"   Alpha=0.0 (pure keyword): {len(results_keyword)} results")

    # Alpha = 0.7 (70% vector, 30% keyword) - default
    store.alpha = 0.7
    results_hybrid = await store.retrieve(
        query="mud weight circulation",
        query_embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
        top_k=2,
    )
    print(f"   Alpha=0.7 (hybrid): {len(results_hybrid)} results")

    # Clean up
    print("\n7. Cleaning up...")
    await store.clear()
    print("   ✓ Cleared all data")

    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_hybrid_search())
