"""
Veritas RAG — Centralized Configuration

All settings are loaded from .env via Pydantic Settings.
PRD constraints (max retries, chunk sizes, top-k values) are enforced
here as typed defaults with validation — not comments.

Usage:
    from src.config import settings
    print(settings.CHUNK_SIZE)  # 512
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root = veritas-rag/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Type-safe settings with PRD constraints enforced as validators."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",  # Don't crash on unknown env vars
    )

    # ── LLM Provider ────────────────────────────────────────────
    LLM_PROVIDER: Literal["google", "openai"] = "google"
    GOOGLE_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    CRITIC_MODEL: str = "gemini-2.0-flash"
    GENERATOR_MODEL: str = "gemini-2.0-flash"

    # ── Embeddings (local) ──────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = Field(default=384, ge=1)

    # ── Reranker (local) ────────────────────────────────────────
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Verifier (local) ────────────────────────────────────────
    VERIFIER_MODEL: str = "cross-encoder/nli-deberta-v3-small"

    # ── ChromaDB ────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = str(PROJECT_ROOT / "data" / "chroma_db")
    CHROMA_COLLECTION_NAME: str = "veritas_rag"

    # ── Retrieval ───────────────────────────────────────────────
    RETRIEVAL_TOP_K: int = Field(default=20, ge=1, le=100)
    RERANK_TOP_K: int = Field(default=6, ge=1, le=50)
    BM25_WEIGHT: float = Field(default=0.3, ge=0.0, le=1.0)
    DENSE_WEIGHT: float = Field(default=0.7, ge=0.0, le=1.0)

    # ── Ingestion ───────────────────────────────────────────────
    OCR_FALLBACK_CHAR_THRESHOLD: int = Field(default=50, ge=0)
    CHUNK_SIZE: int = Field(default=512, ge=100, le=4096)
    CHUNK_OVERLAP: int = Field(default=64, ge=0)

    # ── Orchestration (PRD: bounded retries) ────────────────────
    MAX_REWRITE_RETRIES: int = Field(default=1, ge=0, le=3)
    MAX_REGEN_RETRIES: int = Field(default=1, ge=0, le=3)

    # ── Logging ─────────────────────────────────────────────────
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    LOG_FORMAT: Literal["json", "console"] = "json"

    # ── API Server ──────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = Field(default=8000, ge=1024, le=65535)

    # ── Validators ──────────────────────────────────────────────
    @field_validator("CHUNK_OVERLAP")
    @classmethod
    def overlap_must_be_less_than_chunk(cls, v: int, info) -> int:
        """Overlap must be strictly less than chunk size."""
        chunk_size = info.data.get("CHUNK_SIZE", 512)
        if v >= chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({v}) must be < CHUNK_SIZE ({chunk_size})"
            )
        return v

    @field_validator("BM25_WEIGHT", "DENSE_WEIGHT")
    @classmethod
    def weights_are_positive(cls, v: float) -> float:
        """Fusion weights must be non-negative."""
        if v < 0:
            raise ValueError(f"Weight must be >= 0, got {v}")
        return v


# Singleton — import this everywhere
settings = Settings()
