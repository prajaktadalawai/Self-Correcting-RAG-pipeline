"""
Veritas RAG — Day 1 End-to-End Integration Test

Tests the full ingestion and retrieval pipeline:
1. Parse 5 test docs
2. Chunk them
3. Index them (Hybrid search)
4. Retrieve top-20
5. Rerank to top-6
"""

from pathlib import Path

from src.ingestion.parser import parse_document
from src.ingestion.chunker import chunk_document
from src.retrieval.index import VectorIndex
from src.retrieval.reranker import rerank
from src.schemas import DocumentMetadata

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_CORPUS_DIR = PROJECT_ROOT / "data" / "test_corpus"


def test_day1_pipeline():
    # Setup fresh index
    idx = VectorIndex(collection_name="test_veritas_rag")
    idx.clear()

    # 1 & 2. Ingest Corpus
    docs = {
        "oneinbox_agents_guide_v1.pdf": DocumentMetadata(
            filename="oneinbox_agents_guide_v1.pdf", doc_version="v1", source_tier="official"
        ),
        "oneinbox_kb_guide_scan.png": DocumentMetadata(
            filename="oneinbox_kb_guide_scan.png", doc_version="v1", source_tier="official"
        ),
        "oneinbox_tools_guide_v1.pdf": DocumentMetadata(
            filename="oneinbox_tools_guide_v1.pdf", doc_version="v1", source_tier="official"
        ),
        "oneinbox_tools_guide_v2.pdf": DocumentMetadata(
            filename="oneinbox_tools_guide_v2.pdf", doc_version="v2", source_tier="official"
        ),
        "oneinbox_quickstart_truncated.pdf": DocumentMetadata(
            filename="oneinbox_quickstart_truncated.pdf", doc_version="v1", source_tier="official"
        ),
    }

    all_chunks = []
    for filename, metadata in docs.items():
        path = TEST_CORPUS_DIR / filename
        assert path.exists(), f"Test file missing: {path}"
        
        pages = parse_document(path)
        assert len(pages) > 0
        
        chunks = chunk_document(pages, metadata)
        assert len(chunks) > 0
        all_chunks.extend(chunks)

    # 3. Index
    assert len(all_chunks) > 0
    num_indexed = idx.add_chunks(all_chunks)
    assert num_indexed == len(all_chunks)
    assert idx.count == num_indexed

    # 4. Retrieve
    query = "How many tool types does OneInbox support?"
    retrieved = idx.retrieve(query, top_k=20)
    assert len(retrieved) <= 20
    assert len(retrieved) > 0

    # 5. Rerank
    reranked = rerank(query, retrieved, top_k=6)
    assert len(reranked) <= 6
    assert len(reranked) > 0
    
    # Verify reranking sorted correctly
    for i in range(len(reranked) - 1):
        assert reranked[i].rerank_score >= reranked[i + 1].rerank_score

    print("End-to-End Test Passed!")
