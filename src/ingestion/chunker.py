"""
Veritas RAG — Document Chunker

Recursive character splitting with full PRD-mandated metadata.

Each chunk gets:
    source, page, doc_version, ingest_confidence, timestamp,
    source_tier, chunk_index, chunk_id (deterministic hash)

The chunk_id is a deterministic SHA-256 hash of (source + page + chunk_index)
so re-ingesting the same document produces the same IDs — enabling
idempotent upserts into ChromaDB.

Usage:
    from src.ingestion.chunker import chunk_document
    from src.schemas import DocumentMetadata
    chunks = chunk_document(
        raw_pages=pages,
        doc_metadata=DocumentMetadata(
            filename="agents_guide_v1.pdf",
            doc_version="v1",
            source_tier="official",
        ),
    )
"""

import hashlib
from datetime import datetime, timezone

from src.config import settings
from src.observability.logger import get_logger
from src.schemas import ChunkMetadata, DocumentChunk, DocumentMetadata, RawPage

log = get_logger("ingestion.chunker", stage="ingestion")


def _make_chunk_id(source: str, page: int, chunk_index: int) -> str:
    """
    Deterministic chunk ID from source + page + index.

    Uses SHA-256 truncated to 16 hex chars for readability
    while maintaining collision resistance for our corpus size.
    """
    raw = f"{source}::page{page}::chunk{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
    Split text into overlapping chunks by character count.

    Tries to split on paragraph boundaries (double newline) first,
    then sentence boundaries (. ! ?), then word boundaries (space),
    falling back to hard character split.

    Args:
        text: The text to split.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    separators = ["\n\n", "\n", ". ", "! ", "? ", " "]
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            # Last chunk — take everything remaining
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Try to find the best split point near the end
        best_split = end
        for sep in separators:
            # Look for separator in the last 20% of the chunk
            search_start = start + int(chunk_size * 0.8)
            pos = text.rfind(sep, search_start, end)
            if pos != -1:
                best_split = pos + len(sep)
                break

        chunk = text[start:best_split].strip()
        if chunk:
            chunks.append(chunk)

        # Move start forward with overlap
        start = best_split - chunk_overlap

        # Safety: ensure we always make forward progress
        if start <= (best_split - chunk_size):
            start = best_split

    return chunks


def chunk_document(
    raw_pages: list[RawPage],
    doc_metadata: DocumentMetadata,
) -> list[DocumentChunk]:
    """
    Split parsed pages into chunks with full PRD-mandated metadata.

    Args:
        raw_pages: List of RawPage objects from the parser.
        doc_metadata: Document-level metadata (filename, version, tier).

    Returns:
        List of DocumentChunk objects ready for indexing.
    """
    all_chunks: list[DocumentChunk] = []
    now = datetime.now(timezone.utc)

    log.info(
        "chunking_start",
        source=doc_metadata.filename,
        version=doc_metadata.doc_version,
        tier=doc_metadata.source_tier,
        total_pages=len(raw_pages),
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    for page in raw_pages:
        # Skip empty pages
        if not page.text.strip():
            log.debug(
                "empty_page_skipped",
                source=doc_metadata.filename,
                page=page.page_number,
            )
            continue

        # Split page text into chunks
        text_chunks = _split_text(
            text=page.text,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

        for idx, chunk_text in enumerate(text_chunks):
            chunk_id = _make_chunk_id(
                source=doc_metadata.filename,
                page=page.page_number,
                chunk_index=idx,
            )

            metadata = ChunkMetadata(
                source=doc_metadata.filename,
                page=page.page_number,
                doc_version=doc_metadata.doc_version,
                ingest_confidence=page.ingest_confidence,
                timestamp=now,
                source_tier=doc_metadata.source_tier,
                chunk_index=idx,
            )

            all_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    metadata=metadata,
                )
            )

    # Log summary statistics
    chunk_sizes = [len(c.text) for c in all_chunks]
    avg_size = sum(chunk_sizes) / max(len(chunk_sizes), 1)

    log.info(
        "chunking_complete",
        source=doc_metadata.filename,
        total_chunks=len(all_chunks),
        avg_chunk_size=round(avg_size, 1),
        min_chunk_size=min(chunk_sizes) if chunk_sizes else 0,
        max_chunk_size=max(chunk_sizes) if chunk_sizes else 0,
        pages_with_chunks=len(set(c.metadata.page for c in all_chunks)),
    )

    return all_chunks
