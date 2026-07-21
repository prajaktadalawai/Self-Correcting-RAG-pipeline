"""
Veritas RAG — Test Corpus Generator

Creates 5 deliberately messy test documents from OneInbox documentation
content to exercise every edge case in the ingestion pipeline:

1. oneinbox_agents_guide_v1.pdf     — Clean PDF (baseline)
2. oneinbox_kb_guide_scan.png       — Low-res scanned image (OCR challenge)
3. oneinbox_tools_guide_v1.pdf      — Clean PDF (contradiction source A)
4. oneinbox_tools_guide_v2.pdf      — Clean PDF (contradiction source B)
5. oneinbox_quickstart_truncated.pdf — Truncated PDF (insufficient context)

The contradiction: v1 says "OneInbox supports 6 tool types" while v2 says
"OneInbox supports 8 tool types including schedule_calendar_event and
query_database".

Usage:
    python scripts/build_test_corpus.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fpdf import FPDF  # fpdf2
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "test_corpus"

# ── Document Content (based on real OneInbox docs) ──────────────

AGENTS_GUIDE_CONTENT = """OneInbox Agents - Concept Guide

What is an Agent?
An agent is the voice AI your callers talk to. It has a personality, a voice, and a set of actions it can take during a live phone or web conversation.

Every agent has four building blocks:
1. Voice - what the agent sounds like (Cartesia, Deepgram, ElevenLabs, OpenAI)
2. LLM - the AI model that powers the agent's reasoning (GPT-4o, Claude, Gemini)
3. Tools - actions the agent can take mid-call (send SMS, transfer call, extract info)
4. Knowledge Base - documents the agent can reference to answer questions

Creating an Agent
To create an agent, make a POST request to the OneInbox API:

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

The system prompt is how you control the agent's personality and behavior. Write it as instructions the agent follows during every call. Reference tools by their name field.

Agent Configuration Options:
- name: Display name for the agent
- system_prompt: Instructions the agent follows during calls
- voice_id: UUID of the voice to use
- llm_id: UUID of the LLM model configuration
- knowledge_base_ids: List of KB UUIDs for document retrieval
- first_message: What the agent says when the call connects
- language: Primary language (default: en)

How a Call Works
When a call connects, the following pipeline runs in real time:
Caller speaks -> speech is transcribed to text -> LLM decides what to say -> response is spoken back to the caller.

OneInbox handles real-time audio streaming, speech recognition, telephony routing, and scaling automatically. The entire loop runs with sub-second latency.

Authentication
OneInbox uses two types of API keys:
- API Key (starts with oi_sk_): Full access, used server-side
- Publishable Key (starts with oi_pk_): Browser-only, can only start calls

Use your API key for all server-side operations. The publishable key is for the Web SDK only.

What You Can Build
- Outbound calling bots: Agent dials leads, qualifies them, books demos
- Inbound support lines: Assign a phone number, callers reach your agent 24/7
- Website voice widgets: Web SDK lets visitors talk to your agent in the browser
"""

KB_GUIDE_CONTENT = """OneInbox Knowledge Bases Guide

Upload your content so your agent can answer questions from it during calls.

What is a Knowledge Base?
A knowledge base is a collection of documents that your agent can search and reference during live conversations. When a caller asks a question, the agent retrieves relevant passages from the KB to formulate an accurate answer.

Supported File Types: PDF, DOCX, XLSX/XLS, TXT, Markdown
Maximum file size: 50MB per source

Adding Content to a Knowledge Base
Use the source endpoint to upload content. You do not need to create a KB first. OneInbox auto-creates one and attaches your source.

POST /v1/knowledge-bases/sources

Three ways to add content:
1. File upload (multipart/form-data): file=@path/to/doc.pdf
2. URL: type=url and source=https://example.com
3. Inline text: type=text and source=Your text content here

The response includes a job_id. Ingestion runs in the background:
parse -> S3 storage -> token count -> vector index

Poll GET /v1/knowledge-bases/{kb_id}/jobs/{job_id} for status.

How Retrieval Works
During a call, the agent performs semantic search over the KB:
1. Caller's question is converted to an embedding vector
2. Vector similarity search finds the most relevant passages
3. Passages are injected into the LLM context
4. LLM generates an answer grounded in the retrieved content

Managing Knowledge Bases
- List all KBs: GET /v1/knowledge-bases
- Get KB details: GET /v1/knowledge-bases/{kb_id}
- List sources: GET /v1/knowledge-bases/{kb_id}/sources
- Delete source: DELETE /v1/knowledge-bases/{kb_id}/sources/{source_id}
- Delete KB: DELETE /v1/knowledge-bases/{kb_id}

Best Practices
- Keep documents focused on a single topic for better retrieval precision
- Use clear headings and structured formatting
- Update documents regularly to keep information current
- Monitor job status to ensure successful ingestion
"""

TOOLS_GUIDE_V1_CONTENT = """OneInbox Tools Guide (Version 1.0)

Give your agent the ability to take action during a call.

What are Tools?
Tools are actions your agent can perform mid-call. They fire automatically based on conversation context. The agent's LLM reads the tool description to decide when to trigger each tool.

OneInbox supports 6 tool types:
1. api_call - Call your own HTTP endpoint during the call
2. transfer_call - Transfer the caller to a real phone number
3. end_call - Hang up the call cleanly
4. send_sms - Send a text message
5. send_email - Send an email
6. extract_information - Silently capture structured data from the conversation

Creating a Tool
POST /v1/tools with a JSON body specifying name, type, and description.

The description field is the trigger signal. The LLM reads it to decide when to fire the tool automatically. Write it as a trigger condition, not a label.

Example: send_sms tool
{
  "name": "notify_team_sms",
  "type": "send_sms",
  "description": "SMS the team when a lead is captured.",
  "messaging_config": {
    "to": "caller",
    "body_template": "New lead: {name} ({phone})"
  }
}

Attaching Tools to an Agent
Creating a tool makes it available in your account, but the agent cannot use it yet. You must attach it to the LLM model linked to your agent:

PATCH /v1/models/<llm_id>
{"tool_ids": ["<tool_id_1>", "<tool_id_2>"]}

Every agent linked to this llm_id picks up the change immediately.

Transfer Call Modes
OneInbox supports two transfer modes:
- Cold (default): Blind SIP REFER, caller is immediately handed off
- Warm: Agent calls the human first, briefs them, then connects the caller

Reading Tool Results After a Call
After the call ends, extracted data and tool activity appear in the call record. Fetch with GET /v1/calls/<call_id>.
"""

TOOLS_GUIDE_V2_CONTENT = """OneInbox Tools Guide (Version 2.0 - Updated July 2026)

Give your agent the ability to take action during a call.

What are Tools?
Tools are actions your agent can perform mid-call. They fire automatically based on conversation context. The agent's LLM reads the tool description to decide when to trigger each tool.

OneInbox supports 8 tool types including schedule_calendar_event and query_database:
1. api_call - Call your own HTTP endpoint during the call
2. transfer_call - Transfer the caller to a real phone number
3. end_call - Hang up the call cleanly
4. send_sms - Send a text message
5. send_email - Send an email
6. extract_information - Silently capture structured data from the conversation
7. schedule_calendar_event - Book a calendar event via Cal.com integration
8. query_database - Run a read-only SQL query against your connected database

Creating a Tool
POST /v1/tools with a JSON body specifying name, type, and description.

The description field is the trigger signal. The LLM reads it to decide when to fire the tool automatically. Write it as a trigger condition, not a label.

Example: schedule_calendar_event tool
{
  "name": "book_demo",
  "type": "schedule_calendar_event",
  "description": "Book a product demo when the caller asks to schedule a meeting.",
  "credential_id": "<calcom_credential_id>",
  "calendar_config": {
    "event_type_id": 123456,
    "duration_minutes": 30,
    "timezone": "America/New_York"
  }
}

Attaching Tools to an Agent
Creating a tool makes it available in your account, but the agent cannot use it yet. You must attach it to the LLM model linked to your agent:

PATCH /v1/models/<llm_id>
{"tool_ids": ["<tool_id_1>", "<tool_id_2>"]}

Every agent linked to this llm_id picks up the change immediately.

Transfer Call Modes
OneInbox supports two transfer modes:
- Cold (default): Blind SIP REFER, caller is immediately handed off
- Warm: Agent calls the human first, briefs them, then connects the caller

Reading Tool Results After a Call
After the call ends, extracted data and tool activity appear in the call record. Fetch with GET /v1/calls/<call_id>.
"""

QUICKSTART_CONTENT = """OneInbox Quickstart Guide

Build your first voice agent, get a phone number, and make a real call.
No third-party telephony account required.

Prerequisites
- A OneInbox account (sign up at https://oneinbox-dashboard.vercel.app/signup)
- An API key (starts with oi_sk_)
- An OpenAI API key (for the LLM)

Step 1: Create an Integration
Store your OpenAI API key in OneInbox:
POST /v1/integrations
{
  "provider": "openai",
  "credentials": {"api_key": "sk-..."}
}

Step 2: Create a Model
Register which OpenAI model to use:
POST /v1/models
{
  "kind": "llm",
  "provider": "openai",
  "model": "gpt-4o-mini",
  "credential_id": "<credential_id>",
  "system_prompt": "You are a helpful assistant.",
  "temperature": 0.7
}

Step 3: Create an Agent
POST /v1/agents
{
  "name": "My First Agent",
  "llm_id": "<llm_id>",
  "first_message": "Hello! How can I help you today?"
}

Step 4: Get a Phone Number
Search available numbers:
GET /v1/phone-numbers/search?country=US

Purchase one:
POST /v1/phone-numbers/purchase
{
  "phone_number": "+15551234567",
  "agent_id": "<agent_id>"
}

Step 5: Make a Test Call
POST /v1/calls
{
  "agent_id": "<agent_id>",
  "to_number": "+1YOUR_PHONE",
  "from_number": "+15551234567"
}

Your phone will ring. When you pick up, you will be talking"""
# NOTE: Quickstart is deliberately truncated here ^^


def _create_pdf(content: str, output_path: Path, title: str) -> None:
    """Create a PDF from text content using fpdf2."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", size=14)
    pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # Body
    pdf.set_font("Helvetica", size=10)
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped:
            pdf.ln(3)
            continue

        # Sanitize: replace chars Helvetica can't render
        stripped = (
            stripped.replace("\u2014", "-")
            .replace("\u2013", "-")
            .replace("\u2018", "'")
            .replace("\u2019", "'")
            .replace("\u201c", '"')
            .replace("\u201d", '"')
        )

        # Ensure line isn't too long for a single cell
        if stripped.startswith("#"):
            pdf.set_font("Helvetica", "B", size=12)
            pdf.multi_cell(0, 8, stripped.lstrip("# ").strip(), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=10)
        elif stripped.startswith(("- ", "* ")):
            pdf.multi_cell(0, 6, "  " + stripped, new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.multi_cell(0, 6, stripped, new_x="LMARGIN", new_y="NEXT")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))


def _create_scanned_image(content: str, output_path: Path) -> None:
    """
    Create a low-resolution 'scanned' image from text content.
    Simulates a real-world scanned document with noise.
    """
    # Create image with text
    width, height = 800, 1100
    img = Image.new("RGB", (width, height), color=(245, 240, 235))
    draw = ImageDraw.Draw(img)

    # Use default font (no external font needed)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Draw text line by line
    y_position = 30
    for line in content.split("\n")[:40]:  # Limit lines to fit
        if line.strip():
            draw.text(
                (30, y_position), line.strip()[:80], fill=(30, 30, 30), font=font
            )
        y_position += 22
        if y_position > height - 50:
            break

    # Add some "scan noise" — slightly reduce quality
    # Downscale then upscale to simulate low-res scan
    small = img.resize((width // 3, height // 3), Image.Resampling.LANCZOS)
    noisy = small.resize((width, height), Image.Resampling.NEAREST)

    # Add slight rotation to simulate crooked scan
    noisy = noisy.rotate(1.5, fillcolor=(245, 240, 235), expand=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    noisy.save(str(output_path), quality=60)


def build_corpus() -> dict[str, Path]:
    """
    Build the complete test corpus. Returns dict of name -> file path.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files: dict[str, Path] = {}

    # 1. Clean Agents Guide PDF
    path = OUTPUT_DIR / "oneinbox_agents_guide_v1.pdf"
    _create_pdf(AGENTS_GUIDE_CONTENT, path, "OneInbox Agents Guide v1")
    files["agents_guide_v1"] = path
    print(f"  [1/5] Created: {path.name}")

    # 2. Scanned KB Guide (low-res image)
    path = OUTPUT_DIR / "oneinbox_kb_guide_scan.png"
    _create_scanned_image(KB_GUIDE_CONTENT, path)
    files["kb_guide_scan"] = path
    print(f"  [2/5] Created: {path.name}")

    # 3. Tools Guide v1 (contradiction source A)
    path = OUTPUT_DIR / "oneinbox_tools_guide_v1.pdf"
    _create_pdf(TOOLS_GUIDE_V1_CONTENT, path, "OneInbox Tools Guide v1.0")
    files["tools_guide_v1"] = path
    print(f"  [3/5] Created: {path.name}")

    # 4. Tools Guide v2 (contradiction source B — says 8 tools vs 6)
    path = OUTPUT_DIR / "oneinbox_tools_guide_v2.pdf"
    _create_pdf(TOOLS_GUIDE_V2_CONTENT, path, "OneInbox Tools Guide v2.0")
    files["tools_guide_v2"] = path
    print(f"  [4/5] Created: {path.name}")

    # 5. Truncated Quickstart (cuts off mid-sentence)
    path = OUTPUT_DIR / "oneinbox_quickstart_truncated.pdf"
    _create_pdf(QUICKSTART_CONTENT, path, "OneInbox Quickstart (TRUNCATED)")
    files["quickstart_truncated"] = path
    print(f"  [5/5] Created: {path.name}")

    return files


if __name__ == "__main__":
    print("=" * 60)
    print("Building Veritas RAG Test Corpus")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    files = build_corpus()

    print()
    print(f"Done! Created {len(files)} test documents:")
    for name, path in files.items():
        size_kb = path.stat().st_size / 1024
        print(f"  {name}: {path.name} ({size_kb:.1f} KB)")

    print()
    print("Contradiction test:")
    print("  v1 says: 'OneInbox supports 6 tool types'")
    print("  v2 says: 'OneInbox supports 8 tool types including")
    print("            schedule_calendar_event and query_database'")
    print()
    print("Truncation test:")
    print("  quickstart_truncated.pdf cuts off at 'you will be talking'")
