import asyncio
from pathlib import Path
import time
import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests

load_dotenv()

st.set_page_config(page_title="RAG PDF Assistant", page_icon="📄", layout="centered")

INNGEST_DEV_SERVER = os.getenv("INNGEST_DEV_SERVER_URL", "http://127.0.0.1:8288")
INNGEST_EVENT_KEY = os.getenv("INNGEST_EVENT_KEY", "")  # Empty for local dev


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(
        app_id="rag_app",
        event_key=INNGEST_EVENT_KEY if INNGEST_EVENT_KEY else None,
        is_production=False
    )


def save_uploaded_pdf(file) -> Path:
    """Save uploaded PDF to local directory"""
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_bytes = file.getbuffer()
    file_path.write_bytes(file_bytes)
    return file_path


def send_event_sync(event_name: str, data: dict) -> str:
    """Send event to Inngest and return event ID"""
    url = f"{INNGEST_DEV_SERVER}/e/{event_name if not INNGEST_EVENT_KEY else 'key'}"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "name": event_name,
        "data": data,
    }
    
    if INNGEST_EVENT_KEY:
        headers["Authorization"] = f"Bearer {INNGEST_EVENT_KEY}"
    
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    
    result = response.json()
    # Return first event ID from the response
    return result.get("ids", [""])[0] if "ids" in result else ""


def get_run_output(event_id: str, timeout_s: float = 120, poll_interval: float = 1.0) -> dict:
    """Poll for function run completion and return output"""
    start_time = time.time()
    
    while time.time() - start_time < timeout_s:
        try:
            # Query the runs endpoint
            url = f"{INNGEST_DEV_SERVER}/v1/events/{event_id}/runs"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                runs = data.get("data", [])
                
                if runs:
                    run = runs[0]
                    status = run.get("status", "").lower()
                    
                    # Check for completion
                    if status in ["completed", "succeeded"]:
                        return run.get("output", {})
                    elif status in ["failed", "cancelled"]:
                        error_msg = run.get("output", {}).get("error", "Unknown error")
                        raise RuntimeError(f"Function failed: {error_msg}")
            
            time.sleep(poll_interval)
            
        except requests.exceptions.RequestException as e:
            st.warning(f"Polling error: {e}")
            time.sleep(poll_interval)
    
    raise TimeoutError(f"Function execution timed out after {timeout_s}s")


# ============================================================================
# UI: PDF Upload & Ingestion
# ============================================================================

st.title("📄 RAG PDF Assistant")
st.markdown("Upload PDFs and ask questions about their content using AI")

st.header("1. Upload PDF")

uploaded = st.file_uploader(
    "Choose a PDF file", 
    type=["pdf"], 
    accept_multiple_files=False,
    help="Upload a PDF document to add to the knowledge base"
)

if uploaded is not None:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info(f"📎 **{uploaded.name}** ({uploaded.size / 1024:.1f} KB)")
    
    with col2:
        if st.button("Ingest", type="primary", use_container_width=True):
            with st.spinner("Processing PDF..."):
                try:
                    # Save file
                    pdf_path = save_uploaded_pdf(uploaded)
                    
                    # Send ingestion event
                    event_id = send_event_sync(
                        "rag/ingest_pdf",
                        {
                            "pdf_path": str(pdf_path.resolve()),
                            "source_id": pdf_path.name,
                        }
                    )
                    
                    # Wait for completion
                    output = get_run_output(event_id, timeout_s=60)
                    ingested_count = output.get("ingested", 0)
                    
                    st.success(f"✅ Successfully ingested {ingested_count} chunks from **{pdf_path.name}**")
                    
                except Exception as e:
                    st.error(f"❌ Ingestion failed: {str(e)}")

st.divider()

# ============================================================================
# UI: Query Interface
# ============================================================================

st.header("2. Ask Questions")

with st.form("query_form", clear_on_submit=False):
    question = st.text_input(
        "Your question",
        placeholder="e.g., What is the best tactic if I have good wingers?",
        help="Ask anything about the uploaded PDFs"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        top_k = st.slider("Number of chunks to retrieve", 1, 10, 5)
    with col2:
        st.write("")  # Spacing
        submitted = st.form_submit_button("Ask", type="primary", use_container_width=True)

if submitted and question.strip():
    with st.spinner("🤔 Searching and generating answer..."):
        try:
            # Send query event
            event_id = send_event_sync(
                "rag/query_pdf_ai",
                {
                    "question": question.strip(),
                    "top_k": int(top_k),
                }
            )
            
            # Wait for answer
            output = get_run_output(event_id, timeout_s=90)
            
            answer = output.get("answer", "")
            sources = output.get("sources", [])
            num_contexts = output.get("num_contexts", 0)
            
            # Display results
            st.subheader("💡 Answer")
            if answer:
                st.markdown(answer)
            else:
                st.warning("No answer generated")
            
            # Show metadata
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Contexts Retrieved", num_contexts)
            with col2:
                st.metric("Sources", len(sources))
            
            # Show sources
            if sources:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(sources, 1):
                        st.text(f"{i}. {Path(source).name}")
            
        except TimeoutError:
            st.error("⏱️ Request timed out. The function might still be running.")
        except Exception as e:
            st.error(f"❌ Query failed: {str(e)}")

# ============================================================================
# Sidebar: Info & Settings
# ============================================================================

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This RAG (Retrieval-Augmented Generation) system:
    
    1. **Ingests** PDFs into a vector database
    2. **Searches** for relevant content
    3. **Generates** answers using AI
    
    **Powered by:**
    - Qdrant (vector DB)
    - Sentence Transformers
    - FLAN-T5 (Q&A model)
    - Inngest (workflow engine)
    """)
    
    st.divider()
    
    st.header("⚙️ Settings")
    st.text(f"Inngest: {INNGEST_DEV_SERVER}")
    
    if st.button("Clear Upload Cache"):
        uploads_dir = Path("uploads")
        if uploads_dir.exists():
            import shutil
            shutil.rmtree(uploads_dir)
            uploads_dir.mkdir()
            st.success("Cache cleared!")