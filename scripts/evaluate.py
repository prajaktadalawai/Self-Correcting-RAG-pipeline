import json
import os
from datetime import datetime
from src.agent.orchestrator import app as agent_app
from src.agent.state import AgentState

# 15 Targeted Queries for Evaluation
TEST_QUERIES = [
    # 5 Standard Factual Queries
    "What is OneInbox?",
    "How do I create a new knowledge base?",
    "What types of agents does OneInbox support?",
    "Explain how webhooks work in OneInbox.",
    "Where can I find the Calls API?",

    # 5 Contradiction Queries (Testing Critic)
    "How many tools does OneInbox support?", # 6 vs 8
    "Can schedule_calendar_event be used by standard agents?", # Contradicted by v2
    "Is the quickstart guide 5 pages long?", # Truncated doc
    "Does OneInbox support direct database integration?", # Might be conflicting
    "Are voice agents currently in beta?", # Mocking contradiction

    # 5 Out-of-bounds/Ambiguous Queries (Testing Clarification/Low Confidence)
    "How do I do it?", # Ambiguous
    "What is the capital of France?", # Out of bounds
    "Who is the CEO of Google?", # Out of bounds
    "Explain the thing about the stuff.", # Ambiguous
    "Can you reset my password?" # Out of bounds
]

def create_initial_state(query: str) -> AgentState:
    return {
        "query": query,
        "original_query": query,
        "retrieved_chunks": [], # In a real test, mock retrieve_node would populate this from ChromaDB
        "critic_verdict": None,
        "generation": None,
        "citations": [],
        "retry_count": 0,
        "generation_retry_count": 0,
        "verification_status": "flagged",
        "status": "processing"
    }

def evaluate_baseline(query: str) -> dict:
    """Baseline: Retrieve -> Generate (No Critic, No Verifier)"""
    from src.agent.orchestrator import retrieve_node, generate_answer_node
    state = create_initial_state(query)
    try:
        state = retrieve_node(state)
        state = generate_answer_node(state)
        return {"variant": "baseline", "query": query, "status": "success", "answer": state.get("generation", "")}
    except Exception as e:
        return {"variant": "baseline", "query": query, "error": str(e)}

def evaluate_plus_critic(query: str) -> dict:
    """+Critic: Retrieve -> Rerank -> Critic -> Generate"""
    from src.agent.orchestrator import retrieve_node, critic_node, generate_answer_node, route_after_critic
    state = create_initial_state(query)
    try:
        state = retrieve_node(state)
        state = critic_node(state)
        next_step = route_after_critic(state)
        if next_step == "generate_answer":
            state = generate_answer_node(state)
            return {"variant": "plus_critic", "query": query, "status": "success", "answer": state.get("generation", "")}
        else:
            return {"variant": "plus_critic", "query": query, "status": next_step, "answer": f"Stopped by critic: {next_step}"}
    except Exception as e:
        return {"variant": "plus_critic", "query": query, "error": str(e)}

def evaluate_full_pipeline(query: str) -> dict:
    """Full Pipeline with Verification"""
    state = create_initial_state(query)
    try:
        final_state = agent_app.invoke(state)
        return {
            "variant": "full_pipeline",
            "query": query,
            "status": final_state["status"],
            "verification_status": final_state["verification_status"],
            "answer": final_state.get("generation", "")
        }
    except Exception as e:
        return {"variant": "full_pipeline", "query": query, "error": str(e)}

def run_evaluation():
    print(f"Starting Evaluation Harness across {len(TEST_QUERIES)} queries...")
    results = []
    
    for idx, query in enumerate(TEST_QUERIES):
        print(f"[{idx+1}/{len(TEST_QUERIES)}] Evaluating: {query}")
        
        # [STUB] EEOC Compliance: In a production hiring scenario, track demographic metadata 
        # for each query (e.g., candidate group) to calculate Adverse Impact Ratio (AIR)
        # def calculate_adverse_impact_ratio(protected_pass_rate, majority_pass_rate):
        #     air = protected_pass_rate / majority_pass_rate
        #     if air < 0.80:
        #         logger.warning("EEOC ALERT: Adverse Impact Ratio below 80% threshold (4/5ths Rule).")
        #     return air

        # In a real execution with API key, this would run live against the LLMs.
        # If no key is present, it will fail gracefully or return mock data.
        if os.getenv("OPENAI_API_KEY"):
            base_res = evaluate_baseline(query)
            crit_res = evaluate_plus_critic(query)
            full_res = evaluate_full_pipeline(query)
        else:
            base_res = {"query": query, "variant": "baseline", "status": "mocked", "answer": "Mocked baseline"}
            crit_res = {"query": query, "variant": "plus_critic", "status": "mocked", "answer": "Mocked critic"}
            full_res = {"query": query, "variant": "full_pipeline", "status": "mocked", "answer": "Mocked full"}
            
        results.append({
            "query": query,
            "baseline": base_res,
            "plus_critic": crit_res,
            "full_pipeline": full_res
        })
        
    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "evaluation_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Evaluation complete. Results saved to {output_path}")

if __name__ == "__main__":
    run_evaluation()
