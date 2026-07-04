"""
Qdrant Vector Store — Concrete VectorDBClientProtocol Implementation.

HNSW-indexed vector space optimized for rapid payload retrieval with
structural metadata filtering. Supports both a remote Qdrant server
and an in-memory mode for local development.

Embedding generation uses google-genai embeddings by default, with a
configurable callable for alternative embedding providers.
"""
import logging
from typing import List, Dict, Any, Optional, Callable

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchAny,
)

from src.database.models import PDFContext

logger = logging.getLogger(__name__)

# Type alias for embedding functions: text -> vector
EmbeddingFn = Callable[[str], List[float]]


def _default_embedding_fn(text: str) -> List[float]:
    """
    Default embedding function using google-genai.

    Uses the text-embedding-004 model which outputs 768-dimensional
    vectors suitable for cosine similarity search.
    """
    import os
    from google import genai

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=text,
    )
    return response.embeddings[0].values


class QdrantStore:
    """
    Qdrant-backed vector store implementing the VectorDBClientProtocol.

    Stores PDF context chunks with their embeddings and structural
    metadata (entity tags, document title, page number). Supports
    filtered semantic search for cross-source context fusion.
    """

    def __init__(
        self,
        collection_name: str = "pdf_contexts",
        embedding_dim: int = 768,
        host: str = "localhost",
        port: int = 6333,
        use_in_memory: bool = True,
        embedding_fn: Optional[EmbeddingFn] = None,
    ):
        """
        Args:
            collection_name: Qdrant collection name.
            embedding_dim: Dimensionality of embedding vectors.
            host: Qdrant server host (ignored if use_in_memory=True).
            port: Qdrant server port (ignored if use_in_memory=True).
            use_in_memory: If True, uses an ephemeral in-memory store.
            embedding_fn: Callable that converts text to embedding vector.
                         Defaults to google-genai text-embedding-004.
        """
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.embedding_fn = embedding_fn or _default_embedding_fn

        if use_in_memory:
            self.client = QdrantClient(location=":memory:")
            logger.info("Qdrant initialized in in-memory mode.")
        else:
            self.client = QdrantClient(host=host, port=port)
            logger.info(f"Qdrant connected to {host}:{port}")

        self._ensure_collection()

    def _ensure_collection(self):
        """Creates the collection if it doesn't already exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                f"Created Qdrant collection '{self.collection_name}' "
                f"(dim={self.embedding_dim}, distance=COSINE)."
            )
        else:
            logger.info(
                f"Qdrant collection '{self.collection_name}' already exists."
            )

    def embed_text(self, text: str) -> List[float]:
        """
        Generates an embedding vector for the given text.

        Public method so the orchestrator can reuse the same embedding
        function for cache key generation.
        """
        return self.embedding_fn(text)

    def upsert_pdf_contexts(self, contexts: List[PDFContext]) -> int:
        """
        Stores PDF context chunks with their embeddings in Qdrant.

        Args:
            contexts: List of PDFContext objects to store.

        Returns:
            Number of points successfully upserted.
        """
        if not contexts:
            return 0

        points = []
        for ctx in contexts:
            try:
                embedding = self.embed_text(ctx.extracted_text)

                point = PointStruct(
                    id=hash(ctx.id) & 0x7FFFFFFFFFFFFFFF,  # Positive int64
                    vector=embedding,
                    payload={
                        "chunk_id": ctx.id,
                        "document_title": ctx.document_title,
                        "extracted_text": ctx.extracted_text,
                        "published_at": ctx.published_at.isoformat(),
                        "page_number": ctx.page_number,
                        "chunk_index": ctx.chunk_index,
                        "base_sentiment_score": ctx.base_sentiment_score,
                        "entity_tags": ctx.entity_tags,
                    },
                )
                points.append(point)
            except Exception as e:
                logger.error(
                    f"Failed to embed chunk {ctx.id}: {e}. Skipping."
                )

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            logger.info(
                f"Upserted {len(points)} points into "
                f"'{self.collection_name}'."
            )

        return len(points)

    def search_similar_pdfs(
        self,
        query: str,
        limit: int = 3,
        entity_tags: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic vector search for PDF contexts matching the query.

        This method satisfies the VectorDBClientProtocol interface.

        Args:
            query: Search query text (will be embedded).
            limit: Maximum number of results to return.
            entity_tags: Optional entity tag filter — only return contexts
                        whose entity_tags overlap with this list.

        Returns:
            List of dicts compatible with PDFContext(**result).
        """
        try:
            query_vector = self.embed_text(query)
        except Exception as e:
            logger.error(f"Failed to embed search query: {e}")
            return []

        # Build optional metadata filter
        search_filter = None
        if entity_tags:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="entity_tags",
                        match=MatchAny(any=entity_tags),
                    )
                ]
            )

        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=search_filter,
                limit=limit,
                with_payload=True,
            )

            contexts = []
            for point in results.points:
                payload = point.payload
                contexts.append({
                    "id": payload.get("chunk_id", ""),
                    "document_title": payload.get("document_title", ""),
                    "extracted_text": payload.get("extracted_text", ""),
                    "published_at": payload.get("published_at", ""),
                    "page_number": payload.get("page_number"),
                    "chunk_index": payload.get("chunk_index"),
                    "base_sentiment_score": payload.get("base_sentiment_score"),
                    "entity_tags": payload.get("entity_tags", []),
                })

            logger.info(
                f"Vector search returned {len(contexts)} results "
                f"for query (limit={limit})."
            )
            return contexts

        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return []
