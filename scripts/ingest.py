import os
from pathlib import Path
from src.ingestion.parser import parse_document
from src.ingestion.chunker import chunk_document
from src.retrieval.index import VectorIndex
from src.schemas import DocumentMetadata

TEST_CORPUS_DIR = Path("data/test_corpus")

def ingest_all():
    idx = VectorIndex() # Default collection
    idx.clear()

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
        if not path.exists():
            print(f"Skipping {filename} (not found)")
            continue
            
        print(f"Parsing {filename}...")
        pages = parse_document(path)
        chunks = chunk_document(pages, metadata)
        all_chunks.extend(chunks)

    print(f"Indexing {len(all_chunks)} chunks...")
    idx.add_chunks(all_chunks)
    print("Done!")

if __name__ == "__main__":
    ingest_all()
