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

INNGEST_API_BASE = os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288")


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=False)


def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_bytes = file.getbuffer()
    file_path.write_bytes(file_bytes)
    return file_path


async def send_rag_ingest_event(pdf_path: Path) -> str:
    """Send ingestion event and return event ID"""
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )
    return result[0] if result else ""


async def send_rag_query_event(question: str, top_k: int) -> str:
    """Send query event and return event ID"""
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )
    return result[0] if result else ""


def fetch_runs(event_id: str) -> list[dict]:
    """Fetch runs for a given event ID"""
    url = f"{INNGEST_API_BASE}/v1/events/{event_id}/runs"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])
    except requests.exceptions.RequestException as e:
        st.warning(f"Error fetching runs: {e}")
        return []


def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 1.0) -> dict:
    """Poll for function run completion and return output"""
    start = time.time()
    last_status = None
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while True:
        elapsed = time.time() - start
        progress = min(elapsed / timeout_s, 1.0)
        progress_bar.progress(progress)
        
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status", "Unknown")
            last_status = status
            status_text.text(f"Status: {status}")
            
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                progress_bar.progress(1.0)
                status_text.empty()
                progress_bar.empty()
                return run.get("output") or {}
            
            if status in ("Failed", "Cancelled"):
                progress_bar.empty()
                status_text.empty()
                error = run.get("output", {}).get("error", "Unknown error")
                raise RuntimeError(f"Function {status}: {error}")
        
        if elapsed > timeout_s:
            progress_bar.empty()
            status_text.empty()
            raise TimeoutError(f"Timed out after {timeout_s}s (last status: {last_status})")
        
        time.sleep(poll_interval_s)


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
                    
                    # Send event
                    event_id = asyncio.run(send_rag_ingest_event(pdf_path))
                    
                    if not event_id:
                        st.error("Failed to send event")
                    else:
                        # Wait for completion
                        output = wait_for_run_output(event_id, timeout_s=60)
                        ingested_count = output.get("ingested", 0)
                        
                        st.success(f"✅ Successfully ingested **{ingested_count} chunks** from {pdf_path.name}")
                
                except TimeoutError as e:
                    st.error(f"⏱️ {str(e)}")
                except RuntimeError as e:
                    st.error(f"❌ {str(e)}")
                except Exception as e:
                    st.error(f"❌ Ingestion failed: {str(e)}")

st.divider()

# ============================================================================
# UI: Query Interface
# ============================================================================

st.header("2. Ask Questions")

with st.form("rag_query_form", clear_on_submit=False):
    question = st.text_input(
        "Your question",
        placeholder="e.g., What is the best tactic if I have good wingers?",
        help="Ask anything about the uploaded PDFs"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        top_k = st.slider("Number of chunks to retrieve", 1, 10, 5)
    with col2:
        st.write("")
        submitted = st.form_submit_button("Ask", type="primary", use_container_width=True)

if submitted and question.strip():
    with st.spinner("🤔 Searching and generating answer..."):
        try:
            # Send query event
            event_id = asyncio.run(send_rag_query_event(question.strip(), int(top_k)))
            
            if not event_id:
                st.error("Failed to send query event")
            else:
                # Wait for answer
                output = wait_for_run_output(event_id, timeout_s=90)
                
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
        
        except TimeoutError as e:
            st.error(f"⏱️ {str(e)}")
        except RuntimeError as e:
            st.error(f"❌ {str(e)}")
        except Exception as e:
            st.error(f"❌ Query failed: {str(e)}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())

# ============================================================================
# Sidebar: Info & Status
# ============================================================================

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This RAG system:
    
    1. **Ingests** PDFs into Qdrant
    2. **Searches** for relevant content
    3. **Generates** answers using AI
    
    **Tech Stack:**
    - Qdrant (vector DB)
    - Sentence Transformers
    - FLAN-T5 (Q&A model)
    - Inngest (workflow engine)
    - Streamlit (UI)
    """)
    
    st.divider()
    
    st.header("⚙️ Settings")
    st.text(f"Inngest: {INNGEST_API_BASE}")
    
    # Check connection
    try:
        resp = requests.get(f"{INNGEST_API_BASE}/health", timeout=2)
        if resp.status_code == 200:
            st.success("✅ Inngest connected")
        else:
            st.error("❌ Inngest not responding")
    except:
        st.error("❌ Inngest not connected")
    
    st.divider()
    
    if st.button("Clear Upload Cache"):
        uploads_dir = Path("uploads")
        if uploads_dir.exists():
            import shutil
            shutil.rmtree(uploads_dir)
            uploads_dir.mkdir()
            st.success("Cache cleared!")