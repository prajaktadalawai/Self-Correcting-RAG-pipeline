"""
Veritas RAG — Cross-Encoder Reranker

Rescores retrieval candidates using a cross-encoder model
(ms-marco-MiniLM-L-6-v2) and returns the top-K results.

The cross-encoder takes (query, passage) pairs and produces
a relevance score — this is much more accurate than embedding
similarity alone because it considers both texts jointly.

PRD constraint: Reranks top-20 → top-6 (configurable via settings).

Usage:
    from src.retrieval.reranker import rerank
    top6 = rerank(query="How do tools work?", candidates=top20_results, top_k=6)
"""

from sentence_transformers import CrossEncoder

from src.config import settings
from src.observability.logger import get_logger
from src.schemas import RerankedResult, RetrievalResult

log = get_logger("retrieval.reranker", stage="reranking")

# Lazy-load the model on first use (heavy import)
_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    """Load the cross-encoder model (cached after first call)."""
    global _model
    if _model is None:
        log.info(
            "reranker_loading",
            model=settings.RERANKER_MODEL,
        )
        _model = CrossEncoder(settings.RERANKER_MODEL)
        log.info("reranker_loaded", model=settings.RERANKER_MODEL)
    return _model


def rerank(
    query: str,
    candidates: list[RetrievalResult],
    top_k: int | None = None,
) -> list[RerankedResult]:
    """
    Rerank retrieval candidates using the cross-encoder.

    Args:
        query: The user's search query.
        candidates: List of RetrievalResult from hybrid search.
        top_k: Number of results to return (default: settings.RERANK_TOP_K).

    Returns:
        List of RerankedResult sorted by cross-encoder score (descending),
        truncated to top_k.
    """
    top_k = top_k or settings.RERANK_TOP_K

    if not candidates:
        log.warning("rerank_empty_input", query=query[:50])
        return []

    log.info(
        "reranking_start",
        query=query[:80],
        input_count=len(candidates),
        target_top_k=top_k,
    )

    model = _get_model()

    # Create (query, passage) pairs for the cross-encoder
    pairs = [(query, c.text) for c in candidates]

    # Score all pairs in a single batch
    scores = model.predict(pairs)

    # Pair scores with candidates and sort
    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    # Take top-k
    top_results = scored[:top_k]

    # Build RerankedResult objects
    results: list[RerankedResult] = []
    for candidate, score in top_results:
        results.append(
            RerankedResult(
                chunk_id=candidate.chunk_id,
                text=candidate.text,
                metadata=candidate.metadata,
                rerank_score=float(score),
                original_rrf_score=candidate.rrf_score,
            )
        )

    # Log score distribution
    all_scores = [float(s) for _, s in scored]
    top_scores = [float(s) for _, s in top_results]

    log.info(
        "reranking_complete",
        query=query[:80],
        input_count=len(candidates),
        output_count=len(results),
        top_score=round(max(top_scores), 4) if top_scores else 0,
        bottom_score=round(min(top_scores), 4) if top_scores else 0,
        median_score=round(
            sorted(all_scores)[len(all_scores) // 2], 4
        )
        if all_scores
        else 0,
        cutoff_score=round(min(top_scores), 4) if top_scores else 0,
    )

    return results
