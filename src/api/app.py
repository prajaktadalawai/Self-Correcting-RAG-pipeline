from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import structlog

# [STUB] OWASP API Security & Observability Imports
# from fastapi import Request, Depends
# from fastapi_limiter import FastAPILimiter
# from fastapi_limiter.depends import RateLimiter
# from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
# from opentelemetry import trace

from src.schemas import PipelineOutput, Citation
from src.agent.orchestrator import app as agent_app

logger = structlog.get_logger("api.app")

app = FastAPI(
    title="Veritas RAG API",
    description="Self-Correcting RAG Pipeline with Hallucination Detection",
    version="1.0.0"
)

# [STUB] Observability: Instrument FastAPI for OpenTelemetry tracing
# FastAPIInstrumentor.instrument_app(app)
# tracer = trace.get_tracer(__name__)

# [STUB] Security: Initialize Redis-based Rate Limiter for OWASP compliance (e.g., max 10 requests / min)
# @app.on_event("startup")
# async def startup():
#     redis = await aioredis.create_redis_pool("redis://localhost")
#     FastAPILimiter.init(redis)

class QueryInput(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question to ask the pipeline")

# [STUB] Security: Add dependencies=[Depends(RateLimiter(times=10, seconds=60))] to the endpoint
@app.post("/ask", response_model=PipelineOutput)
async def ask_query(payload: QueryInput):
    """
    Passes a query through the LangGraph self-correcting RAG pipeline.
    """
    logger.info("api_request_received", query=payload.query)
    
    # Initialize the LangGraph state
    initial_state = {
        "query": payload.query,
        "original_query": payload.query,
        "retrieved_chunks": [],
        "critic_verdict": None,
        "generation": None,
        "citations": [],
        "retry_count": 0,
        "generation_retry_count": 0,
        "verification_status": "flagged",
        "status": "processing"
    }
    
    try:
        # Run the orchestrator
        final_state = agent_app.invoke(initial_state)
        
        logger.info("api_request_completed", status=final_state["status"])
        
        # Determine confidence label based on verification and status
        confidence_label = "high"
        if final_state["status"] in ["clarification_needed", "contradiction_found", "low_confidence"]:
            confidence_label = "low"
        elif final_state["verification_status"] == "regenerated":
            confidence_label = "medium"
            
        # Map citations from state dicts to Pydantic Citations
        citations = [
            Citation(**c) for c in final_state.get("citations", [])
        ]
        
        chunks = []
        for c in final_state.get("retrieved_chunks", []):
            chunks.append({"text": c.text, "source": c.metadata.source, "page": c.metadata.page})
            
        critic_reasoning = ""
        if final_state.get("critic_verdict"):
            critic_reasoning = final_state["critic_verdict"].reasoning
        
        return PipelineOutput(
            answer=final_state.get("generation", ""),
            confidence_label=confidence_label,
            citations=citations,
            retry_count=final_state.get("retry_count", 0),
            verification_status=final_state.get("verification_status", "flagged"),
            status=final_state.get("status", "success"),
            original_query=final_state.get("original_query", ""),
            retrieved_chunks=chunks,
            critic_reasoning=critic_reasoning
        )
        
    except Exception as e:
        logger.error("api_request_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error during pipeline execution.")

from fastapi.responses import StreamingResponse
import json
import asyncio

# Global in-memory cache to protect demo from free-tier rate limits (only caches successful real queries)
DEMO_STREAM_CACHE = {}

@app.get("/ask_stream")
async def ask_query_stream(query: str):
    """
    Streams the internal LangGraph state transitions via Server-Sent Events (SSE).
    Implements a demo cache to prevent rate limit crashes during presentations.
    """
    if query in DEMO_STREAM_CACHE:
        logger.info("demo_cache_hit", query=query)
        async def cached_event_generator():
            for event_str in DEMO_STREAM_CACHE[query]:
                # Simulate pipeline processing time for realistic demo effect
                await asyncio.sleep(0.5)
                yield event_str
        return StreamingResponse(cached_event_generator(), media_type="text/event-stream")

    initial_state = {
        "query": query,
        "original_query": query,
        "retrieved_chunks": [],
        "critic_verdict": None,
        "generation": None,
        "citations": [],
        "retry_count": 0,
        "generation_retry_count": 0,
        "verification_status": "flagged",
        "status": "processing"
    }

    async def event_generator():
        cached_events = []
        try:
            # Yield initial state
            event = f"data: {json.dumps({'event': 'started'})}\n\n"
            cached_events.append(event)
            yield event
            
            # Run the orchestrator in stream mode
            for output in agent_app.stream(initial_state, stream_mode="updates"):
                # output is a dict with the node name as key, and state updates as value
                # Unfortunately Pydantic objects need serialization
                def custom_serializer(obj):
                    if hasattr(obj, 'model_dump'):
                        return obj.model_dump()
                    if hasattr(obj, '__dict__'):
                        return obj.__dict__
                    return str(obj)
                
                payload = json.dumps(output, default=custom_serializer)
                event = f"data: {payload}\n\n"
                cached_events.append(event)
                yield event
                
            event = f"data: {json.dumps({'event': 'finished'})}\n\n"
            cached_events.append(event)
            yield event
            
            # Save successful stream to cache for future clicks
            DEMO_STREAM_CACHE[query] = cached_events
            
        except Exception as e:
            logger.error("stream_error", error=str(e))
            error_msg = str(e)
            if "429" in error_msg or "RateLimitError" in error_msg or "quota" in error_msg.lower():
                error_msg = "Free Tier API Rate Limit Exceeded (Google Gemini). Please wait 1 minute before your next query."
            yield f"data: {json.dumps({'error': error_msg})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
