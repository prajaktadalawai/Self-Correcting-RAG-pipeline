import streamlit as st
import requests
import time

st.set_page_config(
    page_title="Veritas RAG Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    h1, h2, h3, h4 { color: #f0f6fc; font-weight: 500; }
    
    /* Clean Badges */
    .status-badge {
        padding: 4px 12px;
        border-radius: 3px;
        font-size: 0.75em;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: inline-block;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    .badge-success { background-color: rgba(46, 160, 67, 0.15); color: #3fb950; border: 1px solid rgba(46, 160, 67, 0.4); }
    .badge-error { background-color: rgba(248, 81, 73, 0.15); color: #f85149; border: 1px solid rgba(248, 81, 73, 0.4); }
    .badge-warning { background-color: rgba(210, 153, 34, 0.15); color: #d29922; border: 1px solid rgba(210, 153, 34, 0.4); }
    .badge-neutral { background-color: rgba(139, 148, 158, 0.15); color: #8b949e; border: 1px solid rgba(139, 148, 158, 0.4); }
    
    /* Trace Container */
    .trace-panel {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 20px;
        height: calc(100vh - 100px);
        overflow-y: auto;
    }
    .trace-item {
        font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
        font-size: 0.85em;
        color: #8b949e;
        margin-bottom: 16px;
        border-left: 2px solid #30363d;
        padding-left: 12px;
    }
    .trace-header {
        color: #c9d1d9;
        font-weight: 600;
        margin-bottom: 4px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .trace-code {
        background-color: #0d1117;
        padding: 8px;
        border-radius: 4px;
        margin-top: 4px;
        border: 1px solid #30363d;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

API_URL = "http://localhost:8000/ask"

# --- Sidebar Controls ---
with st.sidebar:
    st.markdown("### Execution Triggers")
    if st.button("1. Factual Query", use_container_width=True):
        st.session_state.preset_query = "What is OneInbox?"
    if st.button("2. Contradiction Trigger", use_container_width=True):
        st.session_state.preset_query = "How many tool types does OneInbox support?"
    if st.button("3. Ambiguity Trigger", use_container_width=True):
        st.session_state.preset_query = "How do I do it?"
    if st.button("4. Out of Bounds Trigger", use_container_width=True):
        st.session_state.preset_query = "Who is the CEO of Google?"
        
    st.divider()
    if st.button("Clear Dashboard", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# --- Main Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Veritas RAG Interface")
    
    prompt = st.chat_input("Enter query for pipeline execution...")
    if "preset_query" in st.session_state:
        prompt = st.session_state.preset_query
        del st.session_state.preset_query

    info_placeholder = st.empty()
    answer_placeholder = st.empty()

with col2:
    st.markdown("### Pipeline Trace")
    trace_container = st.empty()

if prompt:
    API_STREAM_URL = "http://localhost:8000/ask_stream"
    import json
    
    # Initial UI state
    with trace_container.container():
        st.markdown("<div class='trace-panel'>", unsafe_allow_html=True)
        st.markdown("<div style='color: #8b949e; text-align: center; margin-top: 50px;'>Initializing Pipeline...</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    start_time = time.time()
    
    # We will accumulate state as we read SSE
    current_state = {}
    
    def render_trace(state):
        html = "<div class='trace-panel'>\n"
        
        # 1. Query Processing
        html += f'<div class="trace-item">\n<div class="trace-header">1. Orchestrator: Query Processing</div>\nOriginal Query: {state.get("original_query", "N/A")}\n</div>\n'
        
        # 2. Retrieval
        if "retrieved_chunks" in state and len(state["retrieved_chunks"]) > 0:
            chunks = state["retrieved_chunks"]
            html += f'<div class="trace-item">\n<div class="trace-header">2. Retrieval & Reranking Layer</div>\nChunks retrieved: {len(chunks)}\n</div>\n'
            for idx, c in enumerate(chunks):
                text = c.get('text', '')
                source = c.get('metadata', {}).get('source', 'Unknown')
                page = c.get('metadata', {}).get('page', '?')
                html += f'<div class="trace-item" style="margin-left: 12px; border-left: 1px dashed #30363d;">\n<span style="color: #58a6ff;">[chunk_id: {idx}]</span> {source} (Page {page})\n<div class="trace-code">{text[:150]}...</div>\n</div>\n'
        elif state.get("status") != "processing":
             html += '<div class="trace-item">\n<div class="trace-header">2. Retrieval & Reranking Layer</div>\nChunks retrieved: 0\n</div>\n'
                
        # 3. Critic Layer
        if "critic_verdict" in state and state["critic_verdict"]:
            v = state["critic_verdict"]
            reasoning = v.get("reasoning", "")
            html += f'<div class="trace-item">\n<div class="trace-header">3. Critic Layer Verdict</div>\n<div class="trace-code">{reasoning}</div>\n</div>\n'
            
        # 4. Generation & Verification
        if "generation" in state and state["generation"]:
            citations = state.get("citations", [])
            html += f'<div class="trace-item">\n<div class="trace-header">4. Generator & Verification</div>\nRouted to: {state.get("status", "unknown")}\n<br>Verified Citations: {len(citations)}\n</div>\n'
            for c in citations:
                html += f'<div class="trace-item" style="margin-left: 12px; border-left: 1px dashed #30363d;">\n<span style="color: #3fb950;">[grounded_citation]</span> {c.get("source")} (Page {c.get("page")}) | Tier: {c.get("source_tier")}\n</div>\n'
                
        html += "</div>"
        return html
        
    def render_info(state, latency):
        status = state.get("status", "processing")
        if status == "success":
            status_badge = '<span class="status-badge badge-success">Routing: Generated</span>'
        elif status == "contradiction_found":
            status_badge = '<span class="status-badge badge-error">Routing: Contradiction Block</span>'
        elif status == "clarification_needed":
            status_badge = '<span class="status-badge badge-warning">Routing: Clarification Needed</span>'
        elif status == "low_confidence":
            status_badge = '<span class="status-badge badge-error">Routing: Low Confidence</span>'
        else:
            status_badge = '<span class="status-badge badge-neutral">Routing: Processing...</span>'
            
        verif = state.get("verification_status", "unknown")
        if verif == "verified":
            verif_badge = '<span class="status-badge badge-success">Verifier: Entailed</span>'
        elif verif == "regenerated":
            verif_badge = '<span class="status-badge badge-warning">Verifier: Regenerated</span>'
        elif verif == "flagged":
            verif_badge = '<span class="status-badge badge-error">Verifier: Flagged</span>'
        else:
            verif_badge = '<span class="status-badge badge-neutral">Verifier: Pending...</span>'
            
        # Estimate confidence
        conf = "medium"
        if status in ["clarification_needed", "contradiction_found", "low_confidence"]:
            conf = "low"
        elif verif == "verified":
            conf = "high"
            
        if conf == "high":
            conf_badge = '<span class="status-badge badge-success">Confidence: High</span>'
        elif conf == "medium":
            conf_badge = '<span class="status-badge badge-warning">Confidence: Medium</span>'
        else:
            conf_badge = '<span class="status-badge badge-error">Confidence: Low</span>'
            
        html = f"""
        **Query:** {state.get('original_query', prompt)}
        <br><br>
        {status_badge} {verif_badge} {conf_badge}
        <br>
        <span class='status-badge badge-neutral'>Retry Count: {state.get('retry_count', 0)}</span>
        <span class='status-badge badge-neutral'>Latency: {latency:.2f}s</span>
        """
        return html

    try:
        response = requests.get(f"{API_STREAM_URL}?query={requests.utils.quote(prompt)}", stream=True, timeout=120)
        
        for line in response.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                if decoded.startswith('data: '):
                    data = json.loads(decoded[6:])
                    
                    if "event" in data and data["event"] in ["started", "finished"]:
                        continue
                        
                    if "error" in data:
                        info_placeholder.error(f"Pipeline Error: {data['error']}")
                        break
                        
                    # data is a dict like {"retrieve": {"retrieved_chunks": [...]}}
                    # Update current_state with the node output
                    for node_name, state_update in data.items():
                        current_state.update(state_update)
                        
                    latency = time.time() - start_time
                    
                    # Rerender UI
                    info_placeholder.markdown(render_info(current_state, latency), unsafe_allow_html=True)
                    if "generation" in current_state and current_state["generation"]:
                        answer_placeholder.markdown(f"<div style='margin-top: 16px; font-size: 1.1em; color: #f0f6fc; line-height: 1.6;'>{current_state['generation']}</div>", unsafe_allow_html=True)
                    
                    trace_container.markdown(render_trace(current_state), unsafe_allow_html=True)
                    
    except Exception as e:
        st.error(f"Connection Failed: {str(e)}")
