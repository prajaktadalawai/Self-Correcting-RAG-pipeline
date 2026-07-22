document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('chat-form');
    const input = document.getElementById('query-input');
    const triggers = document.querySelectorAll('.trigger-btn');
    const clearBtn = document.getElementById('clear-btn');
    
    const metricsBar = document.getElementById('metrics-bar');
    const finalAnswer = document.getElementById('final-answer');
    const traceContainer = document.getElementById('trace-container');

    let currentController = null;
    let currentState = {};
    let startTime = 0;

    // Clear Dashboard
    clearBtn.addEventListener('click', () => {
        if (currentController) currentController.abort();
        currentState = {};
        metricsBar.style.display = 'none';
        metricsBar.innerHTML = '';
        finalAnswer.innerHTML = '';
        traceContainer.innerHTML = '';
        input.value = '';
    });

    // Preset Triggers
    triggers.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const query = e.currentTarget.getAttribute('data-query');
            input.value = query;
            form.dispatchEvent(new Event('submit'));
        });
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = input.value.trim();
        if (!query) return;

        if (currentController) currentController.abort();
        currentController = new AbortController();

        currentState = {};
        startTime = Date.now();

        metricsBar.style.display = 'flex';
        finalAnswer.innerHTML = '';
        
        renderInit();

        try {
            const backendUrl = window.BACKEND_URL || "http://localhost:8000";
            const timeoutId = setTimeout(() => currentController.abort(), 60000);

            const response = await fetch(`${backendUrl}/ask_stream?query=${encodeURIComponent(query)}`, {
                signal: currentController.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`HTTP Error ${response.status}: ${response.statusText}`);
            }
            if (!response.body) throw new Error("ReadableStream not supported");

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                
                buffer = lines.pop(); 
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const dataStr = line.replace('data: ', '');
                        try {
                            const data = JSON.parse(dataStr);
                            handleStreamData(data, query);
                        } catch (err) {
                            console.error("JSON parse error:", err);
                        }
                    }
                }
            }
        } catch (error) {
            let errorMsg = error.message;
            if (error.name === 'AbortError') errorMsg = "Request timed out or was cancelled.";
            finalAnswer.innerHTML = `<div class="error-view">Connection Failed: ${errorMsg}</div>`;
        }
    });

    function renderInit() {
        traceContainer.innerHTML = `
            <div class='trace-panel'>
                <div style='color: #8b949e; text-align: center; margin-top: 50px;'>Initializing Pipeline...</div>
            </div>
        `;
    }

    function handleStreamData(data, originalQuery) {
        if (data.event === 'started' || data.event === 'finished') return;
        
        if (data.error) {
            finalAnswer.innerHTML = `<div class="error-view">Pipeline Error: ${data.error}</div>`;
            return;
        }

        // data is { nodeName: { state_updates } }
        for (const [nodeName, stateUpdate] of Object.entries(data)) {
            Object.assign(currentState, stateUpdate);
        }

        const latency = ((Date.now() - startTime) / 1000).toFixed(2);
        
        // Ensure original query is set for render
        if (!currentState.original_query) currentState.original_query = originalQuery;

        renderInfo(latency);
        renderTrace();

        if (currentState.generation) {
            finalAnswer.innerHTML = currentState.generation.replace(/\n/g, '<br>');
        }
    }

    function renderInfo(latency) {
        const status = currentState.status || "processing";
        let statusBadge = '';
        
        if (status === "success") {
            statusBadge = '<span class="status-badge badge-success">Routing: Generated</span>';
        } else if (status === "contradiction_found") {
            statusBadge = '<span class="status-badge badge-error">Routing: Contradiction Block</span>';
        } else if (status === "clarification_needed") {
            statusBadge = '<span class="status-badge badge-warning">Routing: Clarification Needed</span>';
        } else if (status === "low_confidence") {
            statusBadge = '<span class="status-badge badge-error">Routing: Low Confidence</span>';
        } else {
            statusBadge = '<span class="status-badge badge-neutral">Routing: Processing...</span>';
        }

        const verif = currentState.verification_status || "unknown";
        let verifBadge = '';
        if (verif === "verified") {
            verifBadge = '<span class="status-badge badge-success">Verifier: Entailed</span>';
        } else if (verif === "regenerated") {
            verifBadge = '<span class="status-badge badge-warning">Verifier: Regenerated</span>';
        } else if (verif === "flagged") {
            verifBadge = '<span class="status-badge badge-error">Verifier: Flagged</span>';
        } else {
            verifBadge = '<span class="status-badge badge-neutral">Verifier: Pending...</span>';
        }

        let conf = "medium";
        if (["clarification_needed", "contradiction_found", "low_confidence"].includes(status)) {
            conf = "low";
        } else if (verif === "verified") {
            conf = "high";
        }

        let confBadge = '';
        if (conf === "high") {
            confBadge = '<span class="status-badge badge-success">Confidence: High</span>';
        } else if (conf === "medium") {
            confBadge = '<span class="status-badge badge-warning">Confidence: Medium</span>';
        } else {
            confBadge = '<span class="status-badge badge-error">Confidence: Low</span>';
        }

        const retryCount = currentState.retry_count || 0;

        metricsBar.innerHTML = `
            <div style="width: 100%; font-weight: 500; margin-bottom: 0.5rem;">
                <strong>Query:</strong> ${currentState.original_query}
            </div>
            ${statusBadge}
            ${verifBadge}
            ${confBadge}
            <span class='status-badge badge-neutral'>Retry Count: ${retryCount}</span>
            <span class='status-badge badge-neutral'>Latency: ${latency}s</span>
        `;
    }

    function renderTrace() {
        let html = "<div class='trace-panel'>\n";
        
        // 1. Query Processing
        html += `<div class="trace-item">\n<div class="trace-header">1. Orchestrator: Query Processing</div>\nOriginal Query: ${currentState.original_query || "N/A"}\n</div>\n`;
        
        // 2. Retrieval
        if (currentState.retrieved_chunks && currentState.retrieved_chunks.length > 0) {
            const chunks = currentState.retrieved_chunks;
            html += `<div class="trace-item" style="margin-top: 1rem;">\n<div class="trace-header">2. Retrieval & Reranking Layer</div>\nChunks retrieved: ${chunks.length}\n</div>\n`;
            chunks.forEach((c, idx) => {
                const text = c.text || '';
                const source = c.source || 'Unknown';
                const page = c.page || '?';
                html += `<div class="trace-item" style="margin-top: 0.5rem; margin-left: 12px; border-left: 1px dashed #30363d; padding-left: 10px;">\n<span style="color: #58a6ff;">[chunk_id: ${idx}]</span> ${source} (Page ${page})\n<div class="trace-code">${text.substring(0, 150)}...</div>\n</div>\n`;
            });
        } else if (currentState.status && currentState.status !== "processing") {
            html += `<div class="trace-item" style="margin-top: 1rem;">\n<div class="trace-header">2. Retrieval & Reranking Layer</div>\nChunks retrieved: 0\n</div>\n`;
        }
                
        // 3. Critic Layer
        if (currentState.critic_verdict) {
            const reasoning = currentState.critic_verdict.reasoning || "";
            html += `<div class="trace-item" style="margin-top: 1rem;">\n<div class="trace-header">3. Critic Layer Verdict</div>\n<div class="trace-code">${reasoning}</div>\n</div>\n`;
        }
            
        // 4. Generation & Verification
        if (currentState.generation) {
            const citations = currentState.citations || [];
            html += `<div class="trace-item" style="margin-top: 1rem;">\n<div class="trace-header">4. Generator & Verification</div>\nRouted to: ${currentState.status || "unknown"}\n<br>Verified Citations: ${citations.length}\n</div>\n`;
            citations.forEach(c => {
                const source = c.source_file || c.source || '';
                const page = c.page_number || c.page || '';
                const tier = c.source_tier || 'N/A';
                html += `<div class="trace-item" style="margin-top: 0.5rem; margin-left: 12px; border-left: 1px dashed #30363d; padding-left: 10px;">\n<span style="color: #3fb950;">[grounded_citation]</span> ${source} (Page ${page}) | Tier: ${tier}\n</div>\n`;
            });
        }
                
        html += "</div>";
        traceContainer.innerHTML = html;
        traceContainer.scrollTop = traceContainer.scrollHeight;
    }
});
