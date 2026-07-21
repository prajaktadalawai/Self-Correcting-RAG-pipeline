import os
import json
from tenacity import retry, stop_after_attempt, wait_fixed
from openai import OpenAI
import structlog

from src.schemas import DocumentChunk
from src.agent.state import CriticVerdict

logger = structlog.get_logger("agent.critic")

# Initialize OpenAI client
# We use env variables to allow pointing to free providers (like Groq) instead of paid OpenAI
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-tests"),
    base_url=os.getenv("OPENAI_BASE_URL"), # Allows overriding for free APIs
    max_retries=0 # Disable internal exponential backoff so it fails fast on rate limits
)

# Use env var for model to allow free alternatives (e.g., llama-3.1-8b-instant on Groq)
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

@retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
def evaluate_context(query: str, chunks: list[DocumentChunk]) -> CriticVerdict:
    """
    Batched Critic LLM call to evaluate the retrieved context against the query.
    Performs 3 checks in one call: sufficiency, contradiction, clarity.
    
    Wrapped with a timeout (implicitly via OpenAI client) and 1 retry via tenacity.
    """
    logger.info("critic_evaluation_start", query=query, chunk_count=len(chunks), stage="critic")
    
    # Format chunks for the LLM
    context_text = "\n\n".join([
        f"--- Source: {c.metadata.source} (Version: {c.metadata.doc_version}) ---\n{c.text}" 
        for c in chunks
    ])
    
    system_prompt = (
        "You are a strict, objective Critic evaluating retrieved context for a RAG system.\n"
        "Your job is to analyze the provided context against the user's query and make three binary decisions:\n"
        "1. is_sufficient: True if the context contains enough factual information to answer the query.\n"
        "2. has_contradiction: True if different sources in the context provide directly conflicting facts regarding the query.\n"
        "3. needs_clarification: True if the user's query is fundamentally ambiguous or nonsensical, making it impossible to answer even with perfect context.\n"
        "Provide a brief 'reasoning' explaining your verdicts.\n"
        "You MUST output valid JSON with the exact keys: 'is_sufficient', 'has_contradiction', 'needs_clarification', and 'reasoning'."
    )
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}\n\nContext:\n{context_text}"}
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            timeout=15.0 # Timeout of 15 seconds
        )
        
        parsed_dict = json.loads(response.choices[0].message.content)
        verdict = CriticVerdict(**parsed_dict)
        
        logger.info(
            "critic_evaluation_complete",
            is_sufficient=verdict.is_sufficient,
            has_contradiction=verdict.has_contradiction,
            needs_clarification=verdict.needs_clarification,
            stage="critic"
        )
        
        return verdict
        
    except Exception as e:
        logger.error("critic_evaluation_error", error=str(e), stage="critic")
        raise
