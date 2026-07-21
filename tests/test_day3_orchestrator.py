from src.agent.orchestrator import app
from src.agent.state import CriticVerdict

def test_orchestrator_contradiction_route():
    # Mocking the state to bypass real retrieval and critic LLM calls
    # for the purpose of testing the LangGraph routing logic
    initial_state = {
        "query": "How many tools does OneInbox support?",
        "original_query": "How many tools does OneInbox support?",
        "retrieved_chunks": [],
        "critic_verdict": CriticVerdict(
            is_sufficient=True,
            has_contradiction=True,
            needs_clarification=False,
            reasoning="Found contradiction."
        ),
        "generation": None,
        "retry_count": 0,
        "status": ""
    }

    # Instead of running the full app, we just test the routing logic directly
    from src.agent.orchestrator import route_after_critic
    route = route_after_critic(initial_state)
    assert route == "surface_contradiction"

def test_orchestrator_rewrite_loop():
    from src.agent.orchestrator import route_after_critic
    
    # State 1: Insufficient, 0 retries -> should rewrite
    state_0_retries = {
        "query": "query",
        "original_query": "query",
        "retrieved_chunks": [],
        "critic_verdict": CriticVerdict(
            is_sufficient=False,
            has_contradiction=False,
            needs_clarification=False,
            reasoning="Not enough info."
        ),
        "generation": None,
        "retry_count": 0,
        "status": ""
    }
    assert route_after_critic(state_0_retries) == "rewrite_query"

    # State 2: Insufficient, 1 retry -> should give up (low confidence)
    state_1_retry = dict(state_0_retries)
    state_1_retry["retry_count"] = 1
    assert route_after_critic(state_1_retry) == "low_confidence_flag"
