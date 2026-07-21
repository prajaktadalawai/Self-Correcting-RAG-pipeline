import pytest
import os
from src.schemas import DocumentChunk, ChunkMetadata
from src.agent.verifier import verify_hallucination

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), 
    reason="Requires OPENAI_API_KEY"
)

def test_verifier_catches_hallucination():
    from datetime import datetime
    
    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            text="OneInbox is a smart agent platform.",
            metadata=ChunkMetadata(
                source="guide.pdf",
                page=1,
                doc_version="v1",
                ingest_confidence=1.0,
                timestamp=datetime.now(),
                source_tier="official",
                chunk_index=0
            )
        )
    ]
    
    # Hallucinated answer with facts not in chunks
    bad_answer = "OneInbox is a smart agent platform founded in 2024 by Google."
    
    verdict = verify_hallucination(bad_answer, chunks)
    assert verdict.is_entailed is False

def test_verifier_passes_clean_answer():
    from datetime import datetime
    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            text="OneInbox is a smart agent platform.",
            metadata=ChunkMetadata(
                source="guide.pdf",
                page=1,
                doc_version="v1",
                ingest_confidence=1.0,
                timestamp=datetime.now(),
                source_tier="official",
                chunk_index=0
            )
        )
    ]
    
    good_answer = "OneInbox is a smart agent platform."
    
    verdict = verify_hallucination(good_answer, chunks)
    assert verdict.is_entailed is True
