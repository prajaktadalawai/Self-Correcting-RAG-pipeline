import structlog
from typing import Literal
from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.schemas import DocumentChunk
from src.agent.critic import evaluate_context
from src.agent.generator import generate_final_answer
from src.agent.verifier import verify_hallucination
from src.retrieval.index import VectorIndex
from src.retrieval.reranker import rerank
from src.agent.critic import evaluate_context, client, LLM_MODEL
from pydantic import BaseModel, Field

logger = structlog.get_logger("agent.orchestrator")

# Constants
MAX_RETRIES = 1
MAX_GENERATION_RETRIES = 1

# Lazy load index to avoid initialization during imports
_index = None
def get_index():
    global _index
    if _index is None:
        _index = VectorIndex()
    return _index

# -- Node Functions --

def retrieve_node(state: AgentState) -> AgentState:
    """Executes real hybrid retrieval and reranking."""
    logger.info("node_retrieve", query=state["query"])
    idx = get_index()
    candidates = idx.retrieve(state["query"], top_k=20)
    reranked = rerank(state["query"], candidates, top_k=6)
    # Convert RerankedResult back to DocumentChunk for state consistency
    chunks = [DocumentChunk(chunk_id=r.chunk_id, text=r.text, metadata=r.metadata) for r in reranked]
    state["retrieved_chunks"] = chunks
    return state

def critic_node(state: AgentState) -> AgentState:
    """Calls the batched Critic LLM."""
    logger.info("node_critic", query=state["query"])
    if not state.get("retrieved_chunks"):
        from src.agent.state import CriticVerdict
        state["critic_verdict"] = CriticVerdict(
            is_sufficient=False, has_contradiction=False, needs_clarification=False, reasoning="No context."
        )
        return state
        
    verdict = evaluate_context(state["query"], state["retrieved_chunks"])
    state["critic_verdict"] = verdict
    return state

def generate_answer_node(state: AgentState) -> AgentState:
    """Generates the final answer with citations."""
    logger.info("node_generate", query=state["query"])
    result = generate_final_answer(state["query"], state["retrieved_chunks"])
    
    state["generation"] = result.answer
    
    # Map chunk IDs to citations
    used_ids = set(result.used_chunk_ids)
    citations = []
    for chunk in state["retrieved_chunks"]:
        if chunk.chunk_id in used_ids:
            citations.append({
                "source": chunk.metadata.source,
                "page": chunk.metadata.page,
                "source_tier": chunk.metadata.source_tier
            })
    
    state["citations"] = citations
    return state

def verify_node(state: AgentState) -> AgentState:
    """Verifies the generated answer for hallucinations."""
    logger.info("node_verify", query=state["query"])
    verdict = verify_hallucination(state["generation"], state["retrieved_chunks"])
    
    if verdict.is_entailed:
        state["verification_status"] = "verified"
        state["status"] = "success"
    else:
        state["verification_status"] = "flagged"
        state["generation_retry_count"] = state.get("generation_retry_count", 0) + 1
        
    return state

class RewriteOutput(BaseModel):
    query: str = Field(description="The rewritten search query")

def rewrite_query_node(state: AgentState) -> AgentState:
    """Rewrites the query using an LLM and increments retry count."""
    logger.info("node_rewrite", query=state["query"], retry_count=state.get("retry_count", 0))
    state["retry_count"] = state.get("retry_count", 0) + 1
    
    system_prompt = (
        "You are an expert search-query optimization agent.\n"
        "Rewrite the user's original query into a better, more detailed search query for a vector database.\n"
        "Expand abbreviations and add necessary domain context."
    )
    try:
        response = client.beta.chat.completions.parse(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Original query: {state['original_query']}"}
            ],
            temperature=0.2,
            response_format=RewriteOutput,
        )
        state["query"] = response.choices[0].message.parsed.query
    except Exception as e:
        logger.error("rewrite_error", error=str(e))
        state["query"] = f"Rewritten: {state['original_query']}"
        
    return state

def low_confidence_flag_node(state: AgentState) -> AgentState:
    """Terminal state for low confidence (insufficient context or unfixable hallucination)."""
    logger.info("node_low_confidence", query=state["query"])
    state["status"] = "low_confidence"
    state["generation"] = "I don't have enough verified information to confidently answer."
    state["verification_status"] = "flagged"
    state["citations"] = []
    return state

def ask_clarification_node(state: AgentState) -> AgentState:
    """Terminal state for ambiguous queries."""
    logger.info("node_clarification", query=state["query"])
    state["status"] = "clarification_needed"
    state["generation"] = "Could you please clarify your question?"
    state["verification_status"] = "verified"
    state["citations"] = []
    return state

def surface_contradiction_node(state: AgentState) -> AgentState:
    """Terminal state for contradictions."""
    logger.info("node_contradiction", query=state["query"])
    state["status"] = "contradiction_found"
    state["generation"] = "I found conflicting information in the sources."
    state["verification_status"] = "verified"
    state["citations"] = []
    return state

# -- Edge Routing --

def route_after_critic(state: AgentState) -> Literal[
    "generate_answer",
    "surface_contradiction",
    "ask_clarification",
    "rewrite_query",
    "low_confidence_flag"
]:
    """Routes based on the Critic's verdict."""
    verdict = state["critic_verdict"]
    if verdict.needs_clarification: return "ask_clarification"
    if verdict.has_contradiction: return "surface_contradiction"
    if verdict.is_sufficient: return "generate_answer"
    
    if state.get("retry_count", 0) < MAX_RETRIES: return "rewrite_query"
    return "low_confidence_flag"

def route_after_verify(state: AgentState) -> Literal[
    "generate_answer",
    "low_confidence_flag",
    "__end__"
]:
    """Routes based on verification entailment and retry count."""
    if state["verification_status"] == "verified":
        return END
        
    if state.get("generation_retry_count", 0) < MAX_GENERATION_RETRIES:
        # Loop back to generate again
        logger.info("verification_failed_retrying")
        state["verification_status"] = "regenerated"
        return "generate_answer"
        
    # If we hit max retries on generation and still hallucinating, fallback to low confidence
    logger.info("verification_failed_max_retries")
    return "low_confidence_flag"


# -- Graph Construction --

workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("critic", critic_node)
workflow.add_node("generate_answer", generate_answer_node)
workflow.add_node("verify", verify_node)
workflow.add_node("rewrite_query", rewrite_query_node)
workflow.add_node("low_confidence_flag", low_confidence_flag_node)
workflow.add_node("ask_clarification", ask_clarification_node)
workflow.add_node("surface_contradiction", surface_contradiction_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "critic")

workflow.add_conditional_edges(
    "critic",
    route_after_critic,
    {
        "generate_answer": "generate_answer",
        "surface_contradiction": "surface_contradiction",
        "ask_clarification": "ask_clarification",
        "rewrite_query": "rewrite_query",
        "low_confidence_flag": "low_confidence_flag"
    }
)

workflow.add_edge("rewrite_query", "retrieve")
workflow.add_edge("generate_answer", "verify")

workflow.add_conditional_edges(
    "verify",
    route_after_verify,
    {
        "generate_answer": "generate_answer",
        "low_confidence_flag": "low_confidence_flag",
        END: END
    }
)

workflow.add_edge("surface_contradiction", END)
workflow.add_edge("ask_clarification", END)
workflow.add_edge("low_confidence_flag", END)

app = workflow.compile()
