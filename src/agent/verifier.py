import structlog
from tenacity import retry, stop_after_attempt, wait_fixed
from pydantic import BaseModel, Field

from src.schemas import DocumentChunk
from src.config import settings
from sentence_transformers import CrossEncoder

logger = structlog.get_logger("agent.verifier")

class VerificationResult(BaseModel):
    """Structured output for the hallucination verification check."""
    is_entailed: bool = Field(description="True if the generated answer is strictly supported by the context with zero hallucinations.")
    reasoning: str = Field(description="Explanation of why the answer is or isn't supported by the context.")

# Lazy load the NLI model
_model = None
def _get_model():
    global _model
    if _model is None:
        logger.info("verifier_loading_model", model=settings.VERIFIER_MODEL)
        _model = CrossEncoder(settings.VERIFIER_MODEL)
    return _model

@retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
def verify_hallucination(answer: str, chunks: list[DocumentChunk]) -> VerificationResult:
    """
    Acts as a post-generation check to catch hallucinations using a local NLI model.
    """
    logger.info("verification_start", stage="verification")
    
    if not chunks or not answer:
        return VerificationResult(is_entailed=False, reasoning="Empty context or answer.")
        
    context_text = "\n\n".join([c.text for c in chunks])
    
    try:
        import gc
        import torch
        
        model = _get_model()
        # For NLI: premise is context, hypothesis is the answer
        # The model outputs logits for [Contradiction, Entailment, Neutral]
        with torch.no_grad():
            scores = model.predict([(context_text, answer)])[0]
            
        # Aggressively unload model to free RAM
        global _model
        _model = None
        del model
        gc.collect()
        
        # Labels: 0: contradiction, 1: entailment, 2: neutral
        predicted_label_idx = scores.argmax()
        # Accept entailment (1) or neutral (2) as non-hallucinated. Reject contradiction (0).
        is_entailed = (predicted_label_idx != 0)
        
        reasoning = f"NLI predicted class {predicted_label_idx} (0 is contradiction, 1 is entailment, 2 is neutral). Scores: {scores.tolist()}"
        
        logger.info("verification_complete", is_entailed=is_entailed, stage="verification")
        return VerificationResult(is_entailed=is_entailed, reasoning=reasoning)
        
    except Exception as e:
        logger.error("verification_error", error=str(e), stage="verification")
        raise
