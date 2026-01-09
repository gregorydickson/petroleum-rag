"""Manual test script for FalkorDB storage.

This script can be run independently to verify FalkorDB functionality
without needing the full test suite.

Requirements:
- FalkorDB running on localhost:6379
- Run: docker run -p 6379:6379 falkordb/falkordb:latest

Usage:
    python test_falkordb_manual.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models import DocumentChunk
from storage.falkordb_store import FalkorDBStore


async def test_basic_operations():
    """Test basic FalkorDB operations."""
    print("=== Testing FalkorDB Storage ===\n")

    # Initialize store
    print("1. Initializing FalkorDB store...")
    config = {
        "host": "localhost",
        "port": 6379,
        "graph_name": "manual_test_graph",
        "top_k": 5,
        "min_score": 0.0,
    }
    store = FalkorDBStore(config)

    try:
        await store.initialize()
        print(f"   ✓ Connected to FalkorDB: {store}")
    except Exception as e:
        print(f"   ✗ Failed to connect: {e}")
        print("\nMake sure FalkorDB is running:")
        print("   docker run -p 6379:6379 falkordb/falkordb:latest")
        return False

    # Clear any existing data
    print("\n2. Clearing existing data...")
    await store.clear()
    print("   ✓ Graph cleared")

    # Create test chunks
    print("\n3. Creating test chunks...")
    chunks = [
        DocumentChunk(
            chunk_id="chunk_1",
            document_id="petroleum_doc_1",
            content="Drilling operations require proper mud weight control. The mud density "
            "must be balanced between hydrostatic pressure and formation pressure.",
            chunk_index=0,
            start_page=1,
            end_page=1,
            token_count=50,
            parent_section="Drilling Operations",
            element_ids=["elem_1", "elem_2"],
            metadata={"topic": "drilling", "difficulty": "medium"},
        ),
        DocumentChunk(
            chunk_id="chunk_2",
            document_id="petroleum_doc_1",
            content="Reservoir characterization involves analyzing porosity, permeability, "
            "and fluid saturation. Core analysis provides critical data.",
            chunk_index=1,
            start_page=2,
            end_page=2,
            token_count=45,
            parent_section="Reservoir Engineering",
            element_ids=["elem_3"],
            metadata={"topic": "reservoir", "difficulty": "hard", "references": "chunk_1"},
        ),
        DocumentChunk(
            chunk_id="chunk_3",
            document_id="petroleum_doc_1",
            content="Production optimization requires understanding well performance curves "
            "and decline analysis methods.",
            chunk_index=2,
            start_page=3,
            end_page=3,
            token_count=40,
            parent_section="Production Engineering",
            element_ids=["elem_4", "elem_5"],
            metadata={"topic": "production", "difficulty": "medium"},
        ),
    ]

    # Create simple embeddings (3D for testing)
    embeddings = [
        [1.0, 0.2, 0.1],  # chunk_1 - drilling focus
        [0.3, 1.0, 0.2],  # chunk_2 - reservoir focus
        [0.1, 0.3, 1.0],  # chunk_3 - production focus
    ]

    print(f"   ✓ Created {len(chunks)} test chunks")

    # Store chunks
    print("\n4. Storing chunks in graph database...")
    await store.store_chunks(chunks, embeddings)
    print("   ✓ Chunks stored successfully")

    # Verify graph structure
    print("\n5. Verifying graph structure...")

    # Check Document node
    query = "MATCH (d:Document) RETURN COUNT(d) AS count"
    result = store.graph.query(query)
    doc_count = result.result_set[0][0]
    print(f"   • Document nodes: {doc_count}")

    # Check Section nodes
    query = "MATCH (s:Section) RETURN COUNT(s) AS count"
    result = store.graph.query(query)
    section_count = result.result_set[0][0]
    print(f"   • Section nodes: {section_count}")

    # Check Chunk nodes
    query = "MATCH (c:Chunk) RETURN COUNT(c) AS count"
    result = store.graph.query(query)
    chunk_count = result.result_set[0][0]
    print(f"   • Chunk nodes: {chunk_count}")

    # Check FOLLOWS relationships
    query = "MATCH ()-[r:FOLLOWS]->() RETURN COUNT(r) AS count"
    result = store.graph.query(query)
    follows_count = result.result_set[0][0]
    print(f"   • FOLLOWS relationships: {follows_count}")

    # Check CONTAINS relationships
    query = "MATCH ()-[r:CONTAINS]->() RETURN COUNT(r) AS count"
    result = store.graph.query(query)
    contains_count = result.result_set[0][0]
    print(f"   • CONTAINS relationships: {contains_count}")

    # Test retrieval
    print("\n6. Testing retrieval...")

    # Query about drilling (similar to chunk_1)
    print("\n   Query 1: 'drilling operations mud weight'")
    query_embedding = [0.9, 0.2, 0.1]  # Similar to chunk_1
    results = await store.retrieve(
        query="drilling operations mud weight",
        query_embedding=query_embedding,
        top_k=3,
    )

    print(f"   • Retrieved {len(results)} results")
    for i, result in enumerate(results[:3], 1):
        print(f"     {i}. {result.chunk_id} (score: {result.score:.3f})")
        print(f"        Preview: {result.content[:80]}...")

    # Query about reservoir (similar to chunk_2)
    print("\n   Query 2: 'reservoir characterization porosity'")
    query_embedding = [0.2, 0.9, 0.2]  # Similar to chunk_2
    results = await store.retrieve(
        query="reservoir characterization porosity",
        query_embedding=query_embedding,
        top_k=3,
    )

    print(f"   • Retrieved {len(results)} results")
    for i, result in enumerate(results[:3], 1):
        print(f"     {i}. {result.chunk_id} (score: {result.score:.3f})")

    # Test graph traversal (multi-hop)
    print("\n7. Testing graph traversal (multi-hop)...")
    query = """
    MATCH path = (c1:Chunk {chunk_id: 'chunk_1'})-[:FOLLOWS*1..2]->(c2:Chunk)
    RETURN c2.chunk_id AS chunk_id, length(path) AS hops
    """
    result = store.graph.query(query)
    if result.result_set:
        print("   • Found connected chunks via FOLLOWS:")
        for record in result.result_set:
            print(f"     - {record[0]} (distance: {record[1]} hops)")
    else:
        print("   • No multi-hop connections found")

    # Test REFERENCES relationship
    print("\n8. Testing REFERENCES relationships...")
    query = """
    MATCH (c1:Chunk)-[:REFERENCES]->(c2:Chunk)
    RETURN c1.chunk_id AS from_chunk, c2.chunk_id AS to_chunk
    """
    result = store.graph.query(query)
    if result.result_set:
        print("   • Found REFERENCES relationships:")
        for record in result.result_set:
            print(f"     - {record[0]} → {record[1]}")
    else:
        print("   • No REFERENCES relationships found")

    # Health check
    print("\n9. Running health check...")
    is_healthy = await store.health_check()
    print(f"   • Health status: {'✓ Healthy' if is_healthy else '✗ Unhealthy'}")

    # Cleanup
    print("\n10. Cleaning up...")
    await store.clear()
    print("   ✓ Test graph cleared")

    print("\n=== All tests completed successfully! ===")
    return True


async def main():
    """Main entry point."""
    try:
        success = await test_basic_operations()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
