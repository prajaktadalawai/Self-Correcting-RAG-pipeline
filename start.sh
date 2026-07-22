#!/bin/bash
# start.sh - Unified execution script for Veritas RAG

echo "Starting Veritas RAG..."

# Start FastAPI backend (uvicorn) directly
# We use host 0.0.0.0 so it binds correctly in Docker/Cloud environments
# Restrict threads to prevent OOM errors on Render's 512MB Free Tier
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Starting FastAPI backend..."
exec uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 1
