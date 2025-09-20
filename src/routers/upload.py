import logging
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.dependencies import IndexerDep, PDFParserDep

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/pdf", response_model=dict)
async def upload_and_process_pdf(
    pdf_parser: PDFParserDep,
    indexer: IndexerDep,
    file: UploadFile = File(...),
):
    """
    Endpoint to upload a PDF, process it, and index its content.
    Returns a unique document_id for the processed file.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")

    document_id = str(uuid.uuid4())

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)

        logger.info(f"Processing uploaded PDF '{file.filename}' with document_id: {document_id}")

        # 1. Parse the PDF
        pdf_content = await pdf_parser.parse_pdf(tmp_path)
        if not pdf_content:
            raise HTTPException(status_code=500, detail="Failed to parse PDF content.")

        # 2. Index the content into OpenSearch, tagging with the document_id
        stats = await indexer.index_uploaded_document(
            document_id=document_id,
            source=file.filename or "Uploaded Document",
            pdf_content=pdf_content,
        )

        logger.info(
            f"Successfully indexed content for document_id: {document_id} - chunks_indexed={stats.get('chunks_indexed')}"
        )

    except Exception as e:
        logger.error(f"Error processing uploaded PDF: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
    finally:
        # Clean up the temporary file
        if 'tmp_path' in locals() and tmp_path.exists():
            tmp_path.unlink()

    return {"message": "File processed successfully", "document_id": document_id}