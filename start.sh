#!/bin/bash

# Start the FastAPI backend in the background
echo "Starting FastAPI Backend on port 8000..."
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait a moment to ensure backend starts
sleep 3

# For Render, we use the PORT environment variable if it exists, otherwise default to 8501
PORT=${PORT:-8501}

# Start Streamlit in the foreground on the specified port
echo "Starting Streamlit Frontend on port $PORT..."
streamlit run src/streamlit_app.py --server.port $PORT --server.address 0.0.0.0

# Wait for background processes
wait $BACKEND_PID
