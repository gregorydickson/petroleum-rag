"""FalkorDB storage backend with graph-based retrieval.

FalkorDB combines vector similarity search with graph traversal for multi-hop
reasoning. Documents are stored as graph nodes with embeddings, and relationships
(CONTAINS, FOLLOWS, REFERENCES) enable context expansion beyond simple vector search.

Key features:
- Vector embeddings on graph nodes
- Graph relationships for structural context
- Multi-hop traversal for expanded retrieval
- Section and cross-reference linking
"""

import logging
from typing import Any

from falkordb import FalkorDB

from models import DocumentChunk, RetrievalResult
from storage.base import BaseStorage

logger = logging.getLogger(__name__)


class FalkorDBStore(BaseStorage):
    """FalkorDB storage backend using graph structure and vector embeddings.

    Implements graph-based storage where:
    - Documents, Sections, and Chunks are nodes
    - CONTAINS, FOLLOWS, REFERENCES are relationships
    - Vector embeddings enable similarity search
    - Graph traversal expands context beyond vector matches

    Attributes:
        db: FalkorDB database connection
        graph: FalkorDB graph instance
        graph_name: Name of the graph to use
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize FalkorDB storage backend.

        Args:
            config: Configuration dictionary with host, port, graph_name, etc.
        """
        super().__init__("FalkorDB", config)
        self.graph_name = self.config.get("graph_name", "petroleum_rag")
        self.host = self.config.get("host", "localhost")
        self.port = self.config.get("port", 6379)
        self.db: FalkorDB | None = None
        self.graph: Any = None

    async def initialize(self) -> None:
        """Initialize FalkorDB connection and create graph schema.

        Creates graph schema with:
        - Nodes: Document, Section, Chunk
        - Edges: CONTAINS (hierarchical), FOLLOWS (sequential), REFERENCES (semantic)
        - Vector index on Chunk nodes for similarity search

        Raises:
            ConnectionError: If unable to connect to FalkorDB
            RuntimeError: If schema creation fails
        """
        try:
            # Connect to FalkorDB (Redis-based)
            self.db = FalkorDB(host=self.host, port=self.port)
            self.graph = self.db.select_graph(self.graph_name)

            logger.info(f"Connected to FalkorDB at {self.host}:{self.port}")

            # Create indices for faster lookups
            # Vector index on Chunk nodes
            try:
                # Create vector index for embeddings (if supported)
                # FalkorDB supports vector indices through Redis Vector Similarity
                self.graph.query(
                    """
                    CALL db.idx.vector.create(
                        'Chunk',
                        'embedding',
                        'FLAT',
                        {},
                        'COSINE'
                    )
                    """
                )
                logger.info("Created vector index on Chunk.embedding")
            except Exception as e:
                # Index might already exist or vector extension not enabled
                logger.warning(f"Vector index creation skipped: {e}")

            # Create property indices for fast lookups
            try:
                self.graph.query("CREATE INDEX ON :Chunk(chunk_id)")
                self.graph.query("CREATE INDEX ON :Document(document_id)")
                self.graph.query("CREATE INDEX ON :Section(section_name)")
            except Exception as e:
                # Indices might already exist
                logger.debug(f"Index creation skipped: {e}")

            self._initialized = True
            logger.info(f"FalkorDB graph '{self.graph_name}' initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize FalkorDB: {e}")
            raise ConnectionError(f"Could not connect to FalkorDB: {e}") from e

    async def store_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Store document chunks as graph nodes with embeddings.

        Creates graph structure:
        - Chunk nodes with embeddings and metadata
        - Document and Section nodes
        - CONTAINS relationships (Document->Section->Chunk)
        - FOLLOWS relationships (Chunk->Chunk in sequence)
        - REFERENCES relationships (based on metadata cross-refs)

        Args:
            chunks: List of DocumentChunk objects to store
            embeddings: List of embedding vectors (one per chunk)

        Raises:
            ValueError: If chunks and embeddings length mismatch
            RuntimeError: If storage operation fails
        """
        self.validate_chunks_embeddings(chunks, embeddings)

        if not self._initialized:
            raise RuntimeError("Storage backend not initialized. Call initialize() first.")

        try:
            # Group chunks by document
            chunks_by_doc: dict[str, list[tuple[DocumentChunk, list[float]]]] = {}
            for chunk, embedding in zip(chunks, embeddings):
                if chunk.document_id not in chunks_by_doc:
                    chunks_by_doc[chunk.document_id] = []
                chunks_by_doc[chunk.document_id].append((chunk, embedding))

            # Store each document's chunks with relationships
            for doc_id, doc_chunks in chunks_by_doc.items():
                await self._store_document_graph(doc_id, doc_chunks)

            logger.info(f"Stored {len(chunks)} chunks across {len(chunks_by_doc)} documents")

        except Exception as e:
            logger.error(f"Failed to store chunks in FalkorDB: {e}")
            raise RuntimeError(f"Storage operation failed: {e}") from e

    async def _store_document_graph(
        self,
        document_id: str,
        chunks_with_embeddings: list[tuple[DocumentChunk, list[float]]],
    ) -> None:
        """Store a single document as a graph structure.

        Args:
            document_id: Document identifier
            chunks_with_embeddings: List of (chunk, embedding) tuples
        """
        # Create Document node
        query = """
        MERGE (d:Document {document_id: $doc_id})
        RETURN d
        """
        self.graph.query(query, {"doc_id": document_id})

        # Group chunks by section
        sections: dict[str, list[tuple[DocumentChunk, list[float]]]] = {}
        for chunk, embedding in chunks_with_embeddings:
            section = chunk.parent_section or "default"
            if section not in sections:
                sections[section] = []
            sections[section].append((chunk, embedding))

        # Create Section nodes and store chunks
        for section_name, section_chunks in sections.items():
            # Create Section node
            query = """
            MATCH (d:Document {document_id: $doc_id})
            MERGE (s:Section {section_name: $section, document_id: $doc_id})
            MERGE (d)-[:CONTAINS]->(s)
            RETURN s
            """
            self.graph.query(query, {"doc_id": document_id, "section": section_name})

            # Sort chunks by index for FOLLOWS relationship
            section_chunks.sort(key=lambda x: x[0].chunk_index)

            # Store chunks and create relationships
            prev_chunk_id = None
            for chunk, embedding in section_chunks:
                # Create Chunk node with embedding
                # FalkorDB uses array format for vector embeddings
                query = """
                MATCH (s:Section {section_name: $section, document_id: $doc_id})
                CREATE (c:Chunk {
                    chunk_id: $chunk_id,
                    document_id: $doc_id,
                    content: $content,
                    chunk_index: $chunk_index,
                    start_page: $start_page,
                    end_page: $end_page,
                    token_count: $token_count,
                    parent_section: $parent_section,
                    metadata: $metadata,
                    embedding: $embedding
                })
                CREATE (s)-[:CONTAINS]->(c)
                RETURN c
                """
                self.graph.query(
                    query,
                    {
                        "doc_id": document_id,
                        "section": section_name,
                        "chunk_id": chunk.chunk_id,
                        "content": chunk.content,
                        "chunk_index": chunk.chunk_index,
                        "start_page": chunk.start_page,
                        "end_page": chunk.end_page,
                        "token_count": chunk.token_count,
                        "parent_section": chunk.parent_section,
                        "metadata": str(chunk.metadata),  # Convert dict to string
                        "embedding": embedding,
                    },
                )

                # Create FOLLOWS relationship with previous chunk
                if prev_chunk_id is not None:
                    query = """
                    MATCH (c1:Chunk {chunk_id: $prev_id})
                    MATCH (c2:Chunk {chunk_id: $curr_id})
                    CREATE (c1)-[:FOLLOWS]->(c2)
                    """
                    self.graph.query(
                        query,
                        {"prev_id": prev_chunk_id, "curr_id": chunk.chunk_id},
                    )

                prev_chunk_id = chunk.chunk_id

                # Create REFERENCES relationships if mentioned in metadata
                if "references" in chunk.metadata:
                    ref_ids = chunk.metadata["references"].split(",")
                    for ref_id in ref_ids:
                        ref_id = ref_id.strip()
                        if ref_id:
                            query = """
                            MATCH (c1:Chunk {chunk_id: $chunk_id})
                            MATCH (c2:Chunk {chunk_id: $ref_id})
                            MERGE (c1)-[:REFERENCES]->(c2)
                            """
                            try:
                                self.graph.query(
                                    query,
                                    {"chunk_id": chunk.chunk_id, "ref_id": ref_id},
                                )
                            except Exception:
                                # Reference might not exist yet
                                pass

    async def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve relevant chunks using vector similarity and graph traversal.

        Retrieval strategy:
        1. Vector similarity search finds initial top_k chunks
        2. Graph traversal follows REFERENCES edges for related context
        3. Results are ranked by combined similarity and graph distance
        4. Returns expanded context for better RAG performance

        Args:
            query: Query text (for logging/debugging)
            query_embedding: Query vector embedding
            top_k: Number of initial results to retrieve
            filters: Optional metadata filters (e.g., document_id, section)

        Returns:
            List of RetrievalResult objects, ranked by relevance

        Raises:
            RuntimeError: If retrieval fails
        """
        if not self._initialized:
            raise RuntimeError("Storage backend not initialized. Call initialize() first.")

        try:
            results: list[RetrievalResult] = []

            # Step 1: Vector similarity search
            # FalkorDB supports vector similarity through Redis Vector Similarity
            vector_query = """
            MATCH (c:Chunk)
            WHERE vecf.euclideandistance(c.embedding, $query_embedding) IS NOT NULL
            WITH c, vecf.euclideandistance(c.embedding, $query_embedding) AS distance
            ORDER BY distance ASC
            LIMIT $limit
            RETURN c.chunk_id AS chunk_id,
                   c.document_id AS document_id,
                   c.content AS content,
                   c.metadata AS metadata,
                   (1.0 / (1.0 + distance)) AS score
            """

            # Apply filters if provided
            if filters:
                filter_clauses = []
                if "document_id" in filters:
                    filter_clauses.append("c.document_id = $doc_id")
                if "parent_section" in filters:
                    filter_clauses.append("c.parent_section = $section")

                if filter_clauses:
                    where_clause = " AND ".join(filter_clauses)
                    vector_query = vector_query.replace(
                        "WHERE vecf.euclideandistance",
                        f"WHERE {where_clause} AND vecf.euclideandistance",
                    )

            params = {
                "query_embedding": query_embedding,
                "limit": top_k,
            }
            if filters:
                if "document_id" in filters:
                    params["doc_id"] = filters["document_id"]
                if "parent_section" in filters:
                    params["section"] = filters["parent_section"]

            result_set = self.graph.query(vector_query, params)

            # Step 2: Extract initial matches
            initial_chunks = set()
            rank = 1
            for record in result_set.result_set:
                chunk_id = record[0]
                document_id = record[1]
                content = record[2]
                metadata = record[3]
                score = float(record[4])

                results.append(
                    RetrievalResult(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        content=content,
                        score=score,
                        metadata={"raw_metadata": metadata},
                        rank=rank,
                        retrieval_method="graph",
                    )
                )
                initial_chunks.add(chunk_id)
                rank += 1

            # Step 3: Graph expansion - follow REFERENCES for multi-hop
            # For each initial result, find referenced chunks
            expanded_chunks = []
            for result in results[:top_k]:  # Only expand top results
                expansion_query = """
                MATCH (c1:Chunk {chunk_id: $chunk_id})-[:REFERENCES]->(c2:Chunk)
                WHERE NOT c2.chunk_id IN $already_retrieved
                RETURN c2.chunk_id AS chunk_id,
                       c2.document_id AS document_id,
                       c2.content AS content,
                       c2.metadata AS metadata
                LIMIT 2
                """
                expansion_result = self.graph.query(
                    expansion_query,
                    {
                        "chunk_id": result.chunk_id,
                        "already_retrieved": list(initial_chunks),
                    },
                )

                # Add expanded results with lower scores
                for record in expansion_result.result_set:
                    chunk_id = record[0]
                    if chunk_id not in initial_chunks:
                        expanded_chunks.append(
                            RetrievalResult(
                                chunk_id=chunk_id,
                                document_id=record[1],
                                content=record[2],
                                score=result.score * 0.7,  # Discount for expanded results
                                metadata={"raw_metadata": record[3], "expanded_from": result.chunk_id},
                                rank=rank,
                                retrieval_method="graph",
                            )
                        )
                        initial_chunks.add(chunk_id)
                        rank += 1

            # Combine initial and expanded results
            results.extend(expanded_chunks)

            # Apply minimum score threshold
            min_score = self.get_min_score()
            results = [r for r in results if r.score >= min_score]

            logger.info(
                f"Retrieved {len(results)} chunks (initial: {top_k}, expanded: {len(expanded_chunks)})"
            )

            return results

        except Exception as e:
            logger.error(f"Failed to retrieve from FalkorDB: {e}")
            raise RuntimeError(f"Retrieval operation failed: {e}") from e

    async def clear(self) -> None:
        """Clear all data from the graph.

        Removes all nodes and relationships from the graph.
        Use with caution - this is destructive!

        Raises:
            RuntimeError: If clear operation fails
        """
        if not self._initialized:
            raise RuntimeError("Storage backend not initialized. Call initialize() first.")

        try:
            # Delete all nodes and relationships
            query = """
            MATCH (n)
            DETACH DELETE n
            """
            self.graph.query(query)
            logger.info(f"Cleared all data from graph '{self.graph_name}'")

        except Exception as e:
            logger.error(f"Failed to clear FalkorDB: {e}")
            raise RuntimeError(f"Clear operation failed: {e}") from e

    async def health_check(self) -> bool:
        """Check if FalkorDB is healthy and responsive.

        Returns:
            True if backend is healthy, False otherwise
        """
        if not self._initialized:
            return False

        try:
            # Simple query to check connection
            result = self.graph.query("RETURN 1")
            return result is not None
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.close()

    async def close(self) -> None:
        """Cleanup Redis/FalkorDB connection."""
        if self.db is not None:
            try:
                # FalkorDB is Redis-based, close the connection
                if hasattr(self.db, 'close'):
                    self.db.close()
                self.graph = None
                self.db = None
                self._initialized = False
                logger.info("FalkorDB connection closed")
            except Exception as e:
                logger.warning(f"Error closing FalkorDB connection: {e}")

    def __repr__(self) -> str:
        """String representation of the storage backend."""
        return f"FalkorDBStore(graph='{self.graph_name}', host='{self.host}', port={self.port})"
