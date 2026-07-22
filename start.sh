#!/bin/bash
# start.sh - Unified execution script for Veritas RAG

echo "Starting Veritas RAG..."

# Start FastAPI backend (uvicorn) directly
# We use host 0.0.0.0 so it binds correctly in Docker/Cloud environments
echo "Starting FastAPI backend..."
exec uvicorn src.api.app:app --host 0.0.0.0 --port 8000
