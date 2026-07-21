import pytest
import os
from src.schemas import DocumentChunk, ChunkMetadata
from src.agent.generator import generate_final_answer

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), 
    reason="Requires OPENAI_API_KEY"
)

def test_generator_citations():
    from datetime import datetime
    
    query = "What is the capital of France?"
    chunks = [
        DocumentChunk(
            chunk_id="chunk-abc",
            text="Paris is the capital and most populous city of France.",
            metadata=ChunkMetadata(
                source="geography_v1.pdf",
                page=1,
                doc_version="v1",
                ingest_confidence=1.0,
                timestamp=datetime.now(),
                source_tier="official",
                chunk_index=0
            )
        )
    ]
    
    result = generate_final_answer(query, chunks)
    
    # Check that it generated an answer
    assert len(result.answer) > 5
    # Check that it cited the chunk we provided
    assert "chunk-abc" in result.used_chunk_ids
