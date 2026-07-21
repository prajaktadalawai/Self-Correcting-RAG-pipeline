"""
Veritas RAG — Pydantic Data Models

All data structures that flow through the pipeline are defined here.
The PRD's required metadata fields (source, page, doc_version,
ingest_confidence, timestamp, source_tier) are enforced as
**non-optional required fields** on ChunkMetadata.

Usage:
    from src.schemas import DocumentChunk, ChunkMetadata
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ── Ingestion Models ────────────────────────────────────────────


class RawPage(BaseModel):
    """Single page extracted from a document by the parser."""

    text: str = Field(description="Extracted text content")
    page_number: int = Field(ge=1, description="1-indexed page number")
    extraction_method: Literal["pymupdf", "tesseract"] = Field(
        description="Which extraction method produced this text"
    )
    char_count: int = Field(ge=0, description="Character count of extracted text")
    ingest_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence in extraction quality. "
            "1.0 = clean PyMuPDF, 0.3-0.8 = Tesseract (based on char density)"
        ),
    )


class ChunkMetadata(BaseModel):
    """
    PRD-mandated metadata fields — ALL required, none optional.

    From PRD Section 7: "Chunks must include source, page, version,
    ingest_confidence, timestamp, and source_tier"
    """

    source: str = Field(description="Source filename (e.g. 'agents_guide_v1.pdf')")
    page: int = Field(ge=1, description="Page number this chunk came from")
    doc_version: str = Field(
        description="Document version identifier (e.g. 'v1', 'v2')"
    )
    ingest_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Extraction confidence from parser (1.0=clean, <0.5=noisy OCR)",
    )
    timestamp: datetime = Field(
        description="ISO 8601 timestamp of when this chunk was ingested"
    )
    source_tier: Literal["official", "draft", "archived"] = Field(
        description=(
            "Authority level of the source document. "
            "Used for future authority-weighted contradiction resolution."
        ),
    )
    chunk_index: int = Field(ge=0, description="Position of this chunk within the page")


class DocumentChunk(BaseModel):
    """A single chunk ready for indexing."""

    chunk_id: str = Field(description="Deterministic hash: source + page + chunk_index")
    text: str = Field(min_length=1, description="Chunk text content")
    metadata: ChunkMetadata


# ── Retrieval Models ────────────────────────────────────────────


class RetrievalResult(BaseModel):
    """Single result from hybrid search (before reranking)."""

    chunk_id: str
    text: str
    metadata: ChunkMetadata
    dense_score: float = Field(default=0.0, description="ChromaDB cosine similarity")
    sparse_score: float = Field(default=0.0, description="BM25 score")
    rrf_score: float = Field(default=0.0, description="Reciprocal Rank Fusion score")


class RerankedResult(BaseModel):
    """Single result after cross-encoder reranking."""

    chunk_id: str
    text: str
    metadata: ChunkMetadata
    rerank_score: float = Field(description="Cross-encoder relevance score")
    original_rrf_score: float = Field(description="Pre-reranking RRF score")


# ── Output Models (used by all tiers) ──────────────────────────


class Citation(BaseModel):
    """Single citation in the final output — PRD Section 12."""

    source: str
    page: int
    source_tier: Literal["official", "draft", "archived"]


class PipelineOutput(BaseModel):
    """
    Final output schema — exactly matches PRD Section 12.

    This is the contract for the FastAPI endpoint response.
    """

    answer: str
    confidence_label: Literal["high", "medium", "low"]
    citations: list[Citation]
    retry_count: int = Field(ge=0, description="Number of retries performed")
    verification_status: Literal["verified", "flagged", "regenerated"]
    status: Literal["success", "clarification_needed", "contradiction_found", "low_confidence"]
    
    # ── Transparency Metadata (for Glass-Box UI) ──
    original_query: str = Field(default="", description="The query before rewriting")
    retrieved_chunks: list[dict] = Field(default_factory=list, description="Raw chunk text and scores for UI transparency")
    critic_reasoning: str = Field(default="", description="The internal reasoning log from the critic layer")


# ── Document Metadata (for ingestion input) ─────────────────────


class DocumentMetadata(BaseModel):
    """Metadata passed to the chunker for each document being ingested."""

    filename: str
    doc_version: str = "v1"
    source_tier: Literal["official", "draft", "archived"] = "official"
