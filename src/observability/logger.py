"""
Veritas RAG — Structured JSON Logging

Uses structlog to produce machine-readable JSON logs with consistent
fields across all pipeline stages.

Every log entry includes:
    event, timestamp, level, logger (module name), stage

Usage:
    from src.observability.logger import get_logger
    log = get_logger("ingestion.parser")
    log.info("page_extracted", page=1, method="pymupdf", chars=1520)
"""

import logging
import sys
from typing import Literal

import structlog

# Track whether we've already configured structlog
_configured = False


def configure_logging(
    level: str = "INFO",
    log_format: Literal["json", "console"] = "json",
) -> None:
    """
    Configure structlog once at application startup.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_format: "json" for production, "console" for dev readability
    """
    global _configured
    if _configured:
        return

    # Set stdlib logging level
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    # Choose renderer based on format
    if log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_logger_name,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            renderer,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(
    module_name: str,
    stage: str | None = None,
) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger bound with module and stage context.

    Args:
        module_name: Dotted module path (e.g. "ingestion.parser")
        stage: Pipeline stage (e.g. "ingestion", "retrieval", "reranking")

    Returns:
        Bound structlog logger with persistent context fields.

    Example:
        log = get_logger("ingestion.parser", stage="ingestion")
        log.info("page_extracted", page=1, method="pymupdf")
        # Produces: {"event": "page_extracted", "page": 1, "method": "pymupdf",
        #            "logger": "ingestion.parser", "stage": "ingestion",
        #            "level": "info", "timestamp": "2026-07-20T..."}
    """
    # Auto-configure with defaults if not yet configured
    # (real app calls configure_logging() at startup with settings)
    if not _configured:
        from src.config import settings

        configure_logging(
            level=settings.LOG_LEVEL,
            log_format=settings.LOG_FORMAT,
        )

    log = structlog.get_logger(module_name)

    # Bind stage if provided
    if stage:
        log = log.bind(stage=stage)

    return log
