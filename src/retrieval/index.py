"""
Veritas RAG — Hybrid Vector Index

Combines ChromaDB dense vector search with BM25 sparse retrieval
using Reciprocal Rank Fusion (RRF) for hybrid search.

Architecture:
    1. ChromaDB stores embeddings (all-MiniLM-L6-v2) + metadata
    2. BM25 index maintained in-memory over the same corpus
    3. retrieve() queries both → RRF merges → top-K results

The BM25 index is rebuilt on add_chunks() and persisted alongside
the ChromaDB data for consistency.

Usage:
    from src.retrieval.index import VectorIndex
    idx = VectorIndex()
    idx.add_chunks(chunks)
    results = idx.retrieve("How do I create an agent?", top_k=20)
"""

import json
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rank_bm25 import BM25Okapi

from src.config import settings
from src.observability.logger import get_logger
from src.schemas import ChunkMetadata, DocumentChunk, RetrievalResult

log = get_logger("retrieval.index", stage="retrieval")


class VectorIndex:
    """
    Hybrid dense + sparse vector index.

    Dense: ChromaDB with sentence-transformer embeddings
    Sparse: BM25Okapi over tokenized chunk texts
    Fusion: Reciprocal Rank Fusion (RRF) with configurable weights
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
    ):
        self._persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self._collection_name = collection_name or settings.CHROMA_COLLECTION_NAME

        # Ensure persist directory exists
        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB with persistent storage
        self._client = chromadb.PersistentClient(path=self._persist_dir)

        # Embedding function — runs locally, no API key needed
        self._embed_fn = SentenceTransformerEmbeddingFunction(
            model_name=settings.EMBEDDING_MODEL
        )

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

        # BM25 index — maintained in memory
        self._bm25: BM25Okapi | None = None
        self._bm25_chunk_ids: list[str] = []
        self._bm25_texts: list[str] = []

        # Rebuild BM25 from existing ChromaDB data if any
        self._rebuild_bm25_from_chroma()

        log.info(
            "index_initialized",
            persist_dir=self._persist_dir,
            collection=self._collection_name,
            existing_docs=self._collection.count(),
            embedding_model=settings.EMBEDDING_MODEL,
        )

    def _rebuild_bm25_from_chroma(self) -> None:
        """Rebuild BM25 index from existing ChromaDB documents."""
        count = self._collection.count()
        if count == 0:
            self._bm25 = None
            self._bm25_chunk_ids = []
            self._bm25_texts = []
            return

        # Fetch all documents from ChromaDB
        all_docs = self._collection.get(
            include=["documents"],
        )

        self._bm25_chunk_ids = all_docs["ids"]
        self._bm25_texts = all_docs["documents"]

        # Tokenize for BM25
        tokenized = [doc.lower().split() for doc in self._bm25_texts]
        self._bm25 = BM25Okapi(tokenized)

        log.debug(
            "bm25_rebuilt",
            doc_count=count,
        )

    def _metadata_to_dict(self, meta: ChunkMetadata) -> dict:
        """Convert ChunkMetadata to a flat dict for ChromaDB storage."""
        return {
            "source": meta.source,
            "page": meta.page,
            "doc_version": meta.doc_version,
            "ingest_confidence": meta.ingest_confidence,
            "timestamp": meta.timestamp.isoformat(),
            "source_tier": meta.source_tier,
            "chunk_index": meta.chunk_index,
        }

    def _dict_to_metadata(self, d: dict) -> ChunkMetadata:
        """Convert ChromaDB metadata dict back to ChunkMetadata."""
        from datetime import datetime

        return ChunkMetadata(
            source=d["source"],
            page=d["page"],
            doc_version=d["doc_version"],
            ingest_confidence=d["ingest_confidence"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            source_tier=d["source_tier"],
            chunk_index=d["chunk_index"],
        )

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """
        Add document chunks to the index (both ChromaDB and BM25).

        Uses upsert for idempotency — re-ingesting the same doc
        with the same chunk_ids will update rather than duplicate.

        Args:
            chunks: List of DocumentChunk objects to index.

        Returns:
            Number of chunks indexed.
        """
        if not chunks:
            log.warning("add_chunks_empty", message="No chunks to index")
            return 0

        log.info(
            "indexing_start",
            chunk_count=len(chunks),
            source=chunks[0].metadata.source,
        )

        # Prepare data for ChromaDB
        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [self._metadata_to_dict(c.metadata) for c in chunks]

        # Upsert into ChromaDB (handles dedup via chunk_id)
        # ChromaDB has a batch size limit, so chunk in batches
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            batch_end = min(i + batch_size, len(ids))
            self._collection.upsert(
                ids=ids[i:batch_end],
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end],
            )

        # Rebuild BM25 index
        self._rebuild_bm25_from_chroma()

        log.info(
            "indexing_complete",
            chunks_indexed=len(chunks),
            total_in_collection=self._collection.count(),
        )

        return len(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Hybrid search: dense (ChromaDB) + sparse (BM25) with RRF fusion.

        Args:
            query: Search query string.
            top_k: Number of results to return (default: settings.RETRIEVAL_TOP_K).

        Returns:
            List of RetrievalResult sorted by RRF score (descending).
        """
        top_k = top_k or settings.RETRIEVAL_TOP_K
        collection_count = self._collection.count()

        if collection_count == 0:
            log.warning("retrieve_empty_index", query=query[:50])
            return []

        # Limit query to available documents
        query_n = min(top_k * 2, collection_count)  # Over-fetch for fusion

        log.info(
            "retrieval_start",
            query=query[:80],
            top_k=top_k,
            index_size=collection_count,
        )

        # 1. Dense retrieval via ChromaDB
        dense_results = self._collection.query(
            query_texts=[query],
            n_results=query_n,
            include=["documents", "metadatas", "distances"],
        )

        # Build dense ranking: chunk_id → (rank, distance)
        dense_ranks: dict[str, tuple[int, float]] = {}
        for rank, (chunk_id, distance) in enumerate(
            zip(dense_results["ids"][0], dense_results["distances"][0])
        ):
            # ChromaDB returns cosine distance, convert to similarity
            similarity = 1.0 - distance
            dense_ranks[chunk_id] = (rank + 1, similarity)

        # 2. Sparse retrieval via BM25
        sparse_ranks: dict[str, tuple[int, float]] = {}
        if self._bm25 is not None:
            tokenized_query = query.lower().split()
            bm25_scores = self._bm25.get_scores(tokenized_query)

            # Get top-k by BM25 score
            scored_indices = sorted(
                enumerate(bm25_scores),
                key=lambda x: x[1],
                reverse=True,
            )[:query_n]

            for rank, (idx, score) in enumerate(scored_indices):
                if score > 0:
                    chunk_id = self._bm25_chunk_ids[idx]
                    sparse_ranks[chunk_id] = (rank + 1, score)

        # 3. Reciprocal Rank Fusion (RRF)
        k = 60  # RRF constant
        all_chunk_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        rrf_scores: dict[str, tuple[float, float, float]] = {}

        for chunk_id in all_chunk_ids:
            dense_rank, dense_score = dense_ranks.get(chunk_id, (query_n + 1, 0.0))
            sparse_rank, sparse_score = sparse_ranks.get(chunk_id, (query_n + 1, 0.0))

            rrf = (
                settings.DENSE_WEIGHT * (1.0 / (k + dense_rank))
                + settings.BM25_WEIGHT * (1.0 / (k + sparse_rank))
            )

            rrf_scores[chunk_id] = (rrf, dense_score, sparse_score)

        # Sort by RRF score and take top-k
        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x][0], reverse=True)[
            :top_k
        ]

        # 4. Build result objects with full metadata
        results: list[RetrievalResult] = []

        # Fetch metadata for all result IDs from ChromaDB
        if sorted_ids:
            fetched = self._collection.get(
                ids=sorted_ids,
                include=["documents", "metadatas"],
            )

            # Build lookup
            id_to_data: dict[str, tuple[str, dict]] = {}
            for i, chunk_id in enumerate(fetched["ids"]):
                id_to_data[chunk_id] = (
                    fetched["documents"][i],
                    fetched["metadatas"][i],
                )

            for chunk_id in sorted_ids:
                if chunk_id in id_to_data:
                    text, meta_dict = id_to_data[chunk_id]
                    rrf, dense_s, sparse_s = rrf_scores[chunk_id]

                    results.append(
                        RetrievalResult(
                            chunk_id=chunk_id,
                            text=text,
                            metadata=self._dict_to_metadata(meta_dict),
                            dense_score=dense_s,
                            sparse_score=sparse_s,
                            rrf_score=rrf,
                        )
                    )

        log.info(
            "retrieval_complete",
            query=query[:80],
            dense_hits=len(dense_ranks),
            sparse_hits=len(sparse_ranks),
            fused_hits=len(results),
        )

        return results

    def clear(self) -> None:
        """Delete all documents from the collection. For testing."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25 = None
        self._bm25_chunk_ids = []
        self._bm25_texts = []

        log.info("index_cleared", collection=self._collection_name)

    @property
    def count(self) -> int:
        """Number of chunks in the index."""
        return self._collection.count()
