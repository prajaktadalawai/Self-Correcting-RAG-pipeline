"""Debug fpdf2 rendering issues."""
from fpdf import FPDF

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Helvetica", size=10)

content = """OneInbox Agents - Concept Guide

What is an Agent?
An agent is the voice AI your callers talk to.

Every agent has four building blocks:
1. Voice - what the agent sounds like
2. LLM - the AI model that powers reasoning

Creating an Agent
POST https://api-tokyo.oneinbox.ai/v1/agents
Headers: Authorization: Bearer <api_key>
Content-Type: application/json

Request body:
{
  "name": "Support Agent",
  "system_prompt": "You are a helpful customer support agent for Acme Corp.",
  "voice_id": "<voice_id>",
  "llm_id": "<llm_id>"
}

Agent Configuration Options:
- name: Display name for the agent
- system_prompt: Instructions the agent follows during calls
"""

for i, line in enumerate(content.split("\n")):
    stripped = line.strip()
    if not stripped:
        pdf.ln(3)
        continue
    try:
        pdf.multi_cell(0, 6, stripped)
        print(f"  OK line {i}: {stripped[:60]}")
    except Exception as e:
        print(f"FAIL line {i}: {stripped[:60]}")
        print(f"  Error: {e}")
        # Try encoding fix
        safe = stripped.encode("latin-1", errors="replace").decode("latin-1")
        try:
            pdf.multi_cell(0, 6, safe)
            print(f"  Fixed with latin-1 encoding")
        except Exception as e2:
            print(f"  Still fails: {e2}")

pdf.output("test_debug.pdf")
print("Done - saved test_debug.pdf")
