import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.index import VectorIndex
from src.retrieval.reranker import rerank

def check_retrieval():
    # Load existing index (already built from the e2e test)
    idx = VectorIndex(collection_name="test_veritas_rag")
    
    query = "How many tools does OneInbox support?"
    print(f"\nQuery: '{query}'\n")
    
    # 1. Retrieve top-20
    retrieved = idx.retrieve(query, top_k=20)
    print(f"Retrieved {len(retrieved)} chunks via Hybrid Search.")
    
    # 2. Rerank to top-6
    reranked = rerank(query, retrieved, top_k=6)
    print(f"\nTop {len(reranked)} Reranked Results:\n")
    
    for i, r in enumerate(reranked):
        source = r.metadata.source
        version = r.metadata.doc_version
        text = r.text[:80].replace("\n", " ")
        print(f"[{i+1}] Score: {r.rerank_score:.4f} | {source} (v{version}) | {text}...")

if __name__ == "__main__":
    check_retrieval()
