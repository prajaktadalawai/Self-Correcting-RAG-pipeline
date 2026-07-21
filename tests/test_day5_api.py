from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_ask_endpoint_schema():
    # We pass a simple query. Since our orchestrator has a mock retrieve node,
    # it won't actually hit the vector db, but it will run the critic logic.
    # Without an API key, this might fail or hit the critic node logic.
    # To test just the schema, we can mock the orchestrator invoke if we want,
    # but the test instructions require testing the API response structure.
    
    # We will patch the orchestrator invoke to guarantee a controlled state output
    # so we don't need a live LLM key just to test the API layer's JSON wrapping.
    
    from unittest.mock import patch
    
    mock_final_state = {
        "query": "test query",
        "original_query": "test query",
        "retrieved_chunks": [],
        "critic_verdict": None,
        "generation": "This is a mock answer.",
        "citations": [{"source": "guide.pdf", "page": 1, "source_tier": "official"}],
        "retry_count": 0,
        "generation_retry_count": 0,
        "verification_status": "verified",
        "status": "success"
    }
    
    with patch("src.api.app.agent_app.invoke", return_value=mock_final_state):
        response = client.post("/ask", json={"query": "test query"})
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert data["answer"] == "This is a mock answer."
        assert data["confidence_label"] == "high"
        assert len(data["citations"]) == 1
        assert data["citations"][0]["source"] == "guide.pdf"
        assert data["verification_status"] == "verified"
        assert data["status"] == "success"
