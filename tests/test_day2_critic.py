import pytest
import os
from src.schemas import DocumentChunk, ChunkMetadata, DocumentMetadata
from src.agent.critic import evaluate_context

# Skip tests if no API key is provided
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), 
    reason="Requires OPENAI_API_KEY"
)

def test_critic_contradiction():
    query = "How many tools does OneInbox support?"
    
    # Mock chunks with the exact contradiction
    from datetime import datetime
    
    chunks = [
        DocumentChunk(
            chunk_id="1",
            text="OneInbox supports 6 tool types.",
            metadata=ChunkMetadata(
                source="oneinbox_tools_guide_v1.pdf",
                page=1,
                doc_version="v1",
                ingest_confidence=1.0,
                timestamp=datetime.now(),
                source_tier="official",
                chunk_index=0
            )
        ),
        DocumentChunk(
            chunk_id="2",
            text="OneInbox supports 8 tool types including schedule_calendar_event.",
            metadata=ChunkMetadata(
                source="oneinbox_tools_guide_v2.pdf",
                page=1,
                doc_version="v2",
                ingest_confidence=1.0,
                timestamp=datetime.now(),
                source_tier="official",
                chunk_index=0
            )
        )
    ]
    
    verdict = evaluate_context(query, chunks)
    
    # The context is sufficient to answer, BUT it contains a contradiction
    assert verdict.has_contradiction is True
    assert verdict.needs_clarification is False
