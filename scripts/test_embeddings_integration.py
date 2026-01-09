"""Integration test for UnifiedEmbedder with real OpenAI API calls.

This script tests the embedder with actual API calls to verify:
- Single text embedding works
- Batch embedding works
- Dimensions are correct
- Connection validation works

Usage:
    python scripts/test_embeddings_integration.py

Requirements:
    - OPENAI_API_KEY must be set in environment or .env file
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from embeddings import UnifiedEmbedder
from utils.logging import setup_logging

# Sample petroleum engineering texts
SAMPLE_TEXTS = [
    "The porosity of the reservoir rock is a critical parameter in petroleum engineering.",
    "Enhanced oil recovery techniques include water flooding and CO2 injection.",
    "The American Petroleum Institute (API) gravity is a measure of how heavy or light petroleum liquids are.",
    "Hydraulic fracturing involves injecting high-pressure fluid to create fractures in rock formations.",
    "Reservoir simulation models predict the behavior of oil and gas reservoirs over time.",
]


async def test_single_embedding(embedder: UnifiedEmbedder) -> None:
    """Test single text embedding."""
    print("\n=== Testing Single Text Embedding ===")
    text = SAMPLE_TEXTS[0]
    print(f"Text: {text}")

    embedding = await embedder.embed_text(text)

    print(f"✓ Embedding generated successfully")
    print(f"  - Dimensions: {len(embedding)}")
    print(f"  - First 5 values: {embedding[:5]}")
    print(f"  - Last 5 values: {embedding[-5:]}")
    print(f"  - Min value: {min(embedding):.4f}")
    print(f"  - Max value: {max(embedding):.4f}")

    # Verify dimensions match settings
    assert len(embedding) == embedder.dimensions, (
        f"Dimension mismatch: {len(embedding)} vs {embedder.dimensions}"
    )
    print(f"✓ Dimensions verified: {embedder.dimensions}")


async def test_batch_embedding(embedder: UnifiedEmbedder) -> None:
    """Test batch embedding."""
    print("\n=== Testing Batch Embedding ===")
    print(f"Number of texts: {len(SAMPLE_TEXTS)}")

    embeddings = await embedder.embed_batch(SAMPLE_TEXTS)

    print(f"✓ Batch embeddings generated successfully")
    print(f"  - Number of embeddings: {len(embeddings)}")
    print(f"  - All dimensions: {[len(e) for e in embeddings]}")

    # Verify all dimensions
    assert len(embeddings) == len(SAMPLE_TEXTS), (
        f"Count mismatch: {len(embeddings)} vs {len(SAMPLE_TEXTS)}"
    )
    assert all(len(e) == embedder.dimensions for e in embeddings), (
        "Some embeddings have wrong dimensions"
    )
    print(f"✓ All embeddings have correct dimensions: {embedder.dimensions}")

    # Test similarity (first two embeddings should be somewhat similar)
    # as they're both about petroleum engineering concepts
    dot_product = sum(a * b for a, b in zip(embeddings[0], embeddings[1]))
    print(f"  - Dot product of first two embeddings: {dot_product:.4f}")


async def test_large_batch(embedder: UnifiedEmbedder) -> None:
    """Test large batch that exceeds batch size."""
    print("\n=== Testing Large Batch (Multiple API Calls) ===")

    # Create a large batch
    large_batch = [f"Sample text number {i} about petroleum engineering." for i in range(25)]
    print(f"Number of texts: {len(large_batch)}")
    print(f"Batch size: {embedder.batch_size}")

    embeddings = await embedder.embed_batch(large_batch)

    print(f"✓ Large batch embeddings generated successfully")
    print(f"  - Number of embeddings: {len(embeddings)}")

    assert len(embeddings) == len(large_batch), (
        f"Count mismatch: {len(embeddings)} vs {len(large_batch)}"
    )
    assert all(len(e) == embedder.dimensions for e in embeddings), (
        "Some embeddings have wrong dimensions"
    )
    print(f"✓ All embeddings have correct dimensions")


async def test_connection_validation(embedder: UnifiedEmbedder) -> None:
    """Test connection validation."""
    print("\n=== Testing Connection Validation ===")

    is_valid = await embedder.validate_connection()

    if is_valid:
        print("✓ Connection validation successful")
    else:
        print("✗ Connection validation failed")
        raise RuntimeError("Connection validation failed")


async def main() -> None:
    """Run all integration tests."""
    # Setup logging
    setup_logging(log_level="INFO")

    print("=" * 60)
    print("UnifiedEmbedder Integration Tests")
    print("=" * 60)

    try:
        # Initialize embedder
        print("\n=== Initializing UnifiedEmbedder ===")
        embedder = UnifiedEmbedder(batch_size=10)
        print(f"✓ Embedder initialized")
        print(f"  - Model: {embedder.model}")
        print(f"  - Dimensions: {embedder.dimensions}")
        print(f"  - Batch size: {embedder.batch_size}")

        # Run tests
        await test_connection_validation(embedder)
        await test_single_embedding(embedder)
        await test_batch_embedding(embedder)
        await test_large_batch(embedder)

        # Get stats
        print("\n=== Embedder Statistics ===")
        stats = embedder.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Cleanup
        await embedder.close()

        print("\n" + "=" * 60)
        print("✓ All integration tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
