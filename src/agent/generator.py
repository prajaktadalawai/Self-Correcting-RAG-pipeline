import os
import structlog
from tenacity import retry, stop_after_attempt, wait_fixed
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Any

from src.schemas import DocumentChunk
from src.agent.critic import client, LLM_MODEL # Reuse configured client

logger = structlog.get_logger("agent.generator")

class GeneratedAnswer(BaseModel):
    """Structured output for the final generation to enforce citations."""
    answer: str = Field(description="The final answer to the user's query.")
    used_chunk_ids: list[str] = Field(description="List of chunk_ids that were actually used to form this answer.")

@retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
def generate_final_answer(query: str, chunks: list[DocumentChunk]) -> GeneratedAnswer:
    """
    Generates the final answer using the retrieved context.
    Forces the LLM to output which chunks it used for citations.
    """
    logger.info("generation_start", query=query, chunk_count=len(chunks), stage="generation")
    
    context_text = "\n\n".join([
        f"--- Chunk ID: {c.chunk_id} | Source: {c.metadata.source} (Page {c.metadata.page}) ---\n{c.text}" 
        for c in chunks
    ])
    
    system_prompt = (
        "You are an expert, highly accurate AI assistant.\n"
        "Your task is to answer the user's query using ONLY the provided context.\n"
        "If the context does not contain the answer, you must state that you do not know.\n"
        "Do not hallucinate facts outside of the provided context.\n"
        "You must also return a list of exactly which 'Chunk ID's you used to formulate your answer.\n"
        "You MUST output valid JSON with the exact keys: 'answer' (string) and 'used_chunk_ids' (list of strings)."
    )
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}\n\nContext:\n{context_text}"}
            ],
            temperature=0.2, # slight creativity but constrained
            response_format={"type": "json_object"},
            timeout=20.0
        )
        
        import json
        parsed_dict = json.loads(response.choices[0].message.content)
        result = GeneratedAnswer(**parsed_dict)
        
        logger.info("generation_complete", used_citations=len(result.used_chunk_ids), stage="generation")
        return result
        
    except Exception as e:
        logger.error("generation_error", error=str(e), stage="generation")
        raise
