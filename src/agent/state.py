from typing import TypedDict, Optional, Any
from pydantic import BaseModel, Field

from src.schemas import DocumentChunk

class CriticVerdict(BaseModel):
    """Structured output for the batched Critic LLM call."""
    is_sufficient: bool = Field(description="True if the context contains enough information to answer the query")
    has_contradiction: bool = Field(description="True if the context contains conflicting information across different sources")
    needs_clarification: bool = Field(description="True if the query is too ambiguous to answer even with context")
    reasoning: str = Field(description="Brief explanation of the verdict decisions")

class AgentState(TypedDict):
    """LangGraph state for the agentic RAG pipeline."""
    query: str
    original_query: str
    retrieved_chunks: list[DocumentChunk]
    critic_verdict: Optional[CriticVerdict]
    generation: Optional[str]
    citations: list[dict[str, Any]]
    retry_count: int
    generation_retry_count: int
    verification_status: str
    status: str
