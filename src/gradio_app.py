import json
import logging
import os
from pathlib import Path
from typing import Iterator

import gradio as gr
import httpx

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'gradio_app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Gradio app logging initialized")

# --- Configuration ---
API_BASE_URL = "http://api:8000/api/v1"
DEFAULT_MODEL = "llama3.2:1b"

# --- State Management ---
# Use a simple dictionary to hold the document_id for the current session.
# This makes the app stateful for a single user session.
document_context = {"id": None}

# --- API Functions ---
async def upload_pdf_to_api(pdf_file, progress=gr.Progress(track_tqdm=True)):
    """Sends the uploaded PDF to the backend for processing and indexing."""
    if pdf_file is None:
        return "Please upload a PDF first.", gr.update(interactive=False), gr.update(interactive=False)

    document_context["id"] = None  # Reset context on new upload
    upload_url = f"{API_BASE_URL}/upload/pdf"
    
    try:
        # Gradio provides a temporary path to the uploaded file
        logger.info(f"Processing PDF: {pdf_file.name}")
        
        # Read the file content first to ensure it's accessible
        try:
            with open(pdf_file.name, 'rb') as f:
                file_content = f.read()
            files = {'file': (pdf_file.name, file_content, 'application/pdf')}
        except Exception as file_error:
            logger.error(f"Error reading PDF file: {file_error}", exc_info=True)
            return f"❌ Error reading PDF file: {file_error}", gr.update(interactive=False), gr.update(interactive=False)
        
        async with httpx.AsyncClient(timeout=500.0) as client:
            progress(0, desc="Uploading and processing PDF...")
            try:
                response = await client.post(upload_url, files=files)
                logger.info(f"Upload response status: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    document_id = result.get("document_id")
                    if not document_id:
                        logger.error("No document_id in response")
                        return "⚠️ Error: No document ID received from server", gr.update(interactive=False), gr.update(interactive=False)
                        
                    document_context["id"] = document_id
                    logger.info(f"PDF processed successfully. Document ID: {document_id}")
                    return f"✅ **{pdf_file.name}** processed! You can now ask questions about it.", gr.update(interactive=True), gr.update(interactive=True)
                else:
                    error_msg = f"⚠️ Error processing PDF (Status {response.status_code}): {response.text}"
                    logger.error(error_msg)
                    return error_msg, gr.update(interactive=False), gr.update(interactive=False)
                    
            except httpx.RequestError as e:
                error_msg = f"⚠️ Connection error: {str(e)}\nPlease make sure the API server is running."
                logger.error(error_msg, exc_info=True)
                return error_msg, gr.update(interactive=False), gr.update(interactive=False)
                
    except Exception as e:
        error_msg = f"❌ An unexpected error occurred: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg, gr.update(interactive=False), gr.update(interactive=False)

async def stream_response(query: str, top_k: int, use_hybrid: bool, model: str) -> Iterator[str]:
    """Streams the RAG response from the API, filtering by document_id if available."""
    if not query.strip():
        yield "Please enter a question."
        return

    # For PDF chat, we need a document_id
    if document_context.get("id") is None and not query.strip().lower().startswith(('what', 'who', 'when', 'where', 'why', 'how', 'is', 'are', 'can', 'could', 'would', 'will', 'do', 'does', 'did')):
        yield "Error: No document context found. Please upload and process a PDF first."
        return

    payload = {
        "query": query,
        "top_k": top_k,
        "use_hybrid": use_hybrid,
        "model": model,
        "document_id": document_context.get("id"),  # Pass the stored document_id
        "categories": None if document_context.get("id") else None,  # Only set categories to None for PDF chat
    }
    
    url = f"{API_BASE_URL}/stream"
    current_answer = ""
    final_metadata = ""
    
    logger.info(f"Sending request to {url} with payload: {payload}")

    try:
        async with httpx.AsyncClient(timeout=500.0) as client:
            try:
                async with client.stream("POST", url, json=payload, headers={"Accept": "text/plain"}) as response:
                    if response.status_code != 200:
                        error_msg = f"Error: API returned status {response.status_code}"
                        try:
                            error_details = await response.aread()
                            error_msg += f"\nDetails: {error_details.decode()}"
                        except:
                            pass
                        logger.error(error_msg)
                        yield error_msg
                        return

                    logger.info(f"Streaming response from {url}")
                    
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            logger.debug(f"Skipping non-data line: {line}")
                            continue
                        
                        data_str = line[6:].strip()
                        if not data_str:  # Skip empty data lines
                            continue
                            
                        try:
                            data = json.loads(data_str)
                            logger.debug(f"Received data: {data}")
                            
                            if "chunk" in data:
                                current_answer += data["chunk"]
                                yield current_answer
                                
                            if "sources" in data:
                                sources = data["sources"]
                                chunks_used = data.get("chunks_used", 0)
                                search_mode = data.get("search_mode", "unknown")
                                
                                metadata_parts = ["\n\n---", f"**Search Info:** Mode: *{search_mode}*, Chunks used: *{chunks_used}*"]
                                if sources:
                                    metadata_parts.append("\n**Sources:**")
                                    for i, source in enumerate(sources, 1):
                                        # Display filename for uploaded docs, link for arXiv
                                        source_name = source if not source.startswith("http") else source.split('/')[-1]
                                        metadata_parts.append(f"  {i}. {source_name}")
                                final_metadata = "\n".join(metadata_parts)
                                logger.debug(f"Updated metadata: {final_metadata}")
                                
                            if "error" in data:
                                error_msg = f"Error: {data['error']}"
                                logger.error(error_msg)
                                yield error_msg
                                return
                                
                            if data.get("done", False):
                                current_answer = data.get("answer", current_answer)
                                logger.info("Received done signal from server")
                                break
                                
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error for line: {line}")
                            logger.error(f"Error: {str(e)}")
                            continue
                            
            except httpx.RequestError as e:
                error_msg = f"Connection error: {str(e)}\nMake sure the API server is running."
                logger.error(error_msg, exc_info=True)
                yield error_msg
                return
                
    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        logger.error(error_msg, exc_info=True)
        yield error_msg
        return

    # Final response with metadata
    full_response = current_answer + final_metadata
    logger.info(f"Streaming complete. Response length: {len(full_response)} characters")
    yield full_response

# --- Gradio UI Layout ---
def create_gradio_interface():
    """Create and configure the Gradio interface."""
    with gr.Blocks(theme=gr.themes.Soft(), title="arXiv & PDF RAG Chat") as interface:
        gr.Markdown("# 🔬 Chat with arXiv Papers or Your Own PDF")
        
        with gr.Tabs():
            # Tab 1: Chat with your own PDF
            with gr.TabItem("Chat with Your PDF"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 1. Upload Your PDF")
                        pdf_uploader = gr.File(label="Upload a PDF", file_types=[".pdf"])
                        upload_status = gr.Markdown("Your PDF will be processed here.")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### 2. Ask a Question")
                        # Chat components are disabled until a PDF is processed
                        query_input = gr.Textbox(label="Your Question", placeholder="Upload a PDF to enable chat...", interactive=False)
                        submit_btn = gr.Button("Ask", variant="primary", interactive=False)
                        response_output = gr.Markdown(label="Answer", value="The answer will appear here.")

            # Tab 2: Chat with arXiv papers (original functionality)
            with gr.TabItem("Chat with arXiv"):
                gr.Markdown("Ask questions about machine learning and AI research papers from the pre-indexed arXiv dataset.")
                arxiv_query_input = gr.Textbox(label="Your Question", placeholder="What are transformers in machine learning?")
                arxiv_submit_btn = gr.Button("Ask arXiv", variant="primary")
                arxiv_response_output = gr.Markdown(label="Answer")
                
                with gr.Accordion("Advanced Options", open=False):
                    top_k = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Number of chunks")
                    use_hybrid = gr.Checkbox(value=True, label="Use hybrid search")
                    model_choice = gr.Dropdown(choices=["llama3.2:1b"], value=DEFAULT_MODEL, label="LLM Model")
                    categories = gr.Textbox(label="arXiv Categories (optional)", placeholder="cs.AI, cs.LG")

        # --- Event Handling ---

        # PDF Upload Logic
        pdf_uploader.upload(
            fn=upload_pdf_to_api,
            inputs=[pdf_uploader],
            outputs=[upload_status, query_input, submit_btn],
            show_progress="full"
        )

        # PDF Chat Logic
        submit_btn.click(
            fn=stream_response,
            inputs=[query_input, top_k, use_hybrid, model_choice],
            outputs=[response_output]
        )
        query_input.submit(
            fn=stream_response,
            inputs=[query_input, top_k, use_hybrid, model_choice],
            outputs=[response_output]
        )

        # arXiv Chat Logic
        arxiv_submit_btn.click(
            fn=stream_response,
            inputs=[arxiv_query_input, top_k, use_hybrid, model_choice], # No categories for now to simplify
            outputs=[arxiv_response_output]
        )
        arxiv_query_input.submit(
            fn=stream_response,
            inputs=[arxiv_query_input, top_k, use_hybrid, model_choice], # No categories for now
            outputs=[arxiv_response_output]
        )

    return interface

def main():
    """Main entry point for the Gradio app."""
    print("🚀 Starting arXiv Paper Curator Gradio Interface...")
    interface = create_gradio_interface()
    interface.launch(server_name="0.0.0.0", server_port=7861, show_error=True)

if __name__ == "__main__":
    main()