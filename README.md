# Veritas RAG: Self-Correcting Pipeline

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-blue)

**Veritas RAG** is a highly robust, self-correcting Retrieval-Augmented Generation (RAG) pipeline designed to eliminate hallucinations by intelligently identifying and routing contradictory, ambiguous, or out-of-bounds queries safely. 

---

## 2. Demo and Visuals

*The live demo features a real-time Streamlit dashboard that visualizes the internal thought process of the Orchestrator, Critic, and Verifier agents as they stream responses via FastAPI.*

**(Insert Demo GIF or Screenshot here)**

- **Live URL (Hugging Face / Render):** [Insert Live Link Here]

---

## 3. Key Features

- **Self-Correcting Architecture**: Employs an internal "Critic" to evaluate retrieved context and a "Verifier" to fact-check the final generated output against the source.
- **Intelligent Routing**: Gracefully handles bad data. If sources contradict each other, or if the user asks a fundamentally ambiguous question, the system routes to a "Low Confidence" state instead of hallucinating.
- **High-Speed Inference**: Powered by LLaMA-3 (via Groq API) for blazing-fast generation and reasoning.
- **Streaming Pipeline**: FastAPI backend uses Server-Sent Events (SSE) to stream real-time pipeline traces directly to the Streamlit UI.
- **Docker-Ready deployment**: Runs seamlessly in a single container for platforms like Hugging Face Spaces or Render.

---

## 4. Prerequisites and System Requirements

- **Environment:** Windows, macOS, or Linux.
- **Hardware:** CPU-only required (API handles LLM inference).
- **Dependencies:** 
  - Python 3.11+
  - A valid [Groq API Key](https://console.groq.com/keys) for ultra-fast LLaMA-3 inference.

---

## 5. Installation Guide

Follow these steps to get the Veritas RAG pipeline running locally on your machine:

```bash
# 1. Clone the repository
git clone https://github.com/prajaktadalawai/Self-Correcting-RAG-pipeline.git

# 2. Navigate into the directory
cd Self-Correcting-RAG-pipeline

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your environment variables
cp .env.example .env
```
*Note: Open the `.env` file and insert your `OPENAI_API_KEY` (use your Groq API key, as the system is configured to route OpenAI SDK calls to the Groq endpoints).*

---

## 6. Usage and Quickstart

The project provides a unified startup script that launches both the FastAPI backend and Streamlit frontend simultaneously.

**To run the application locally:**
```bash
# Make the script executable (macOS/Linux)
chmod +x start.sh

# Run the startup script
./start.sh
```

Once running, open your browser and navigate to `http://localhost:8501`. 
You can use the preset trigger buttons in the sidebar to test the pipeline's handling of Factual Queries, Contradictions, Ambiguities, and Out-of-Bounds requests.

**To run via Docker:**
```bash
docker build -t veritas-rag .
docker run -p 8501:8501 -p 8000:8000 veritas-rag
```

---

## 7. Project Structure

```text
├── data/
│   └── test_corpus/       # OneInbox API specs and Tool Guides
├── scripts/
│   ├── ingest.py          # Populates ChromaDB with vector embeddings
│   ├── evaluate.py        # Automated evaluation harness
│   └── stress_test.py     # API load testing script
├── src/
│   ├── ingestion/
│   │   ├── parser.py      # PyMuPDF text extraction
│   │   └── chunker.py     # Document chunking & metadata tagging
│   ├── retrieval/
│   │   ├── index.py       # ChromaDB dense retrieval
│   │   └── reranker.py    # Cross-encoder reranking (top-20 -> top-6)
│   ├── agent/
│   │   ├── orchestrator.py # LangGraph state graph wiring
│   │   ├── critic.py      # Batched LLM context evaluation (Sufficiency/Contradiction)
│   │   ├── generator.py   # Answer generation with forced citations
│   │   └── verifier.py    # Post-generation hallucination checks
│   ├── observability/
│   │   └── logger.py      # Structured JSON logging
│   ├── api/
│   │   └── app.py         # FastAPI backend (SSE streaming)
│   └── streamlit_app.py   # Real-time Streamlit dashboard
├── PROMPTS.md             # Master list of all LLM system prompts
├── Dockerfile             # Universal cloud deployment configuration
├── start.sh               # Simultaneous frontend/backend boot script
└── requirements.txt       # Python dependencies
```

---

## 8. Data and Model Weights

- **Dataset:** The knowledge base uses the "OneInbox" technical manuals, tool guides, and API specifications provided specifically for this hackathon. The raw PDFs are located in `data/test_corpus`.
- **Embeddings Model:** Uses `all-MiniLM-L6-v2` (SentenceTransformers) hosted via HuggingFace for dense vector retrieval.
- **Reranker:** Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to rerank and refine initial search results.
- **LLM Checkpoints:** No local weights are required. The pipeline relies on the Groq API for lightning-fast LLaMA-3-8B-Instant inference.
