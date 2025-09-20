import json
import logging
from typing import Iterator

import gradio as gr
import httpx

logger = logging.getLogger(__name__)

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
        files = {'file': (pdf_file.name, open(pdf_file.name, 'rb'), 'application/pdf')}
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            progress(0, desc="Uploading and processing PDF...")
            response = await client.post(upload_url, files=files)
            progress(1, desc="Processing complete!")

            if response.status_code == 200:
                result = response.json()
                document_context["id"] = result.get("document_id")
                # Return a success message and enable the chat components
                return f"✅ **{pdf_file.name}** processed! You can now ask questions about it.", gr.update(interactive=True), gr.update(interactive=True)
            else:
                # Return an error and keep chat disabled
                return f"⚠️ Error processing PDF: {response.text}", gr.update(interactive=False), gr.update(interactive=False)
                
    except Exception as e:
        logger.error(f"An unexpected error occurred during upload: {e}", exc_info=True)
        return f"❌ An unexpected error occurred: {e}", gr.update(interactive=False), gr.update(interactive=False)

async def stream_response(query: str, top_k: int, use_hybrid: bool, model: str) -> Iterator[str]:
    """Streams the RAG response from the API, filtering by document_id if available."""
    if not query.strip():
        yield "Please enter a question."
        return

    payload = {
        "query": query,
        "top_k": top_k,
        "use_hybrid": use_hybrid,
        "model": model,
        "document_id": document_context.get("id"),  # Pass the stored document_id
        "categories": None, # Explicitly set categories to None for PDF chat
    }
    
    url = f"{API_BASE_URL}/stream"
    current_answer = ""
    final_metadata = ""

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("POST", url, json=payload, headers={"Accept": "text/plain"}) as response:
                if response.status_code != 200:
                    yield f"Error: API returned status {response.status_code}"
                    return

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    
                    data_str = line[6:]
                    try:
                        data = json.loads(data_str)
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
                        if "error" in data:
                            yield f"Error: {data['error']}"
                            return
                        if data.get("done", False):
                            current_answer = data.get("answer", current_answer)
                            break
                    except json.JSONDecodeError:
                        continue
    except httpx.RequestError as e:
        yield f"Connection error: {str(e)}\nMake sure the API server is running."
        return
    except Exception as e:
        yield f"An unexpected error occurred: {str(e)}"
        return

    yield current_answer + final_metadata

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