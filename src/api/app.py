"""
FastAPI REST API for the AI Document Intelligence System.

Provides endpoints for document upload, querying, and system management.
"""

from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import tempfile
import os

from ..core.rag_engine import RAGEngine
from ..utils.config import config
from ..utils.logger import get_logger


logger = get_logger(__name__)


# Pydantic models for request/response validation
class QueryRequest(BaseModel):
    """Request model for querying the system."""
    question: str = Field(..., min_length=1, max_length=1000, description="Question to ask")
    n_results: Optional[int] = Field(5, ge=1, le=20, description="Number of results to retrieve")
    include_sources: Optional[bool] = Field(True, description="Include source documents")


class QueryResponse(BaseModel):
    """Response model for query results."""
    answer: str
    n_sources: int
    sources: Optional[List[dict]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    document_count: int


class UploadResponse(BaseModel):
    """Document upload response."""
    message: str
    filename: str
    chunks_created: int


# Initialize FastAPI app
app = FastAPI(
    title="AI Document Intelligence API",
    description="RAG-based document Q&A system with LLM integration",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG engine
rag_engine = RAGEngine()


@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    logger.info("API server starting up")
    
    # Try to load existing index
    try:
        rag_engine.load_index()
        logger.info("Loaded existing vector index")
    except Exception as e:
        logger.info("No existing index found, starting fresh", error=str(e))


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("API server shutting down")
    
    # Save index
    try:
        rag_engine.save_index()
        logger.info("Vector index saved")
    except Exception as e:
        logger.error("Failed to save index", error=str(e))


@app.get("/", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns system status and basic information.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        document_count=rag_engine.document_count
    )


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the document knowledge base.
    
    Args:
        request: Query request with question and parameters
        
    Returns:
        Answer with optional source documents
    """
    logger.info("Query received", question=request.question[:100])
    
    try:
        result = rag_engine.query(
            question=request.question,
            n_results=request.n_results,
            include_sources=request.include_sources
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error("Query failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )


@app.post("/documents", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload and process a document.
    
    Args:
        file: Document file to upload (PDF, TXT, JSON)
        background_tasks: Background task queue
        
    Returns:
        Upload confirmation with processing details
    """
    logger.info("Document upload received", filename=file.filename)
    
    # Validate file extension
    allowed_extensions = {".pdf", ".txt", ".json"}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {allowed_extensions}"
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Load and process document
        from ..preprocessing.document_loader import DocumentLoader
        loader = DocumentLoader()
        doc_data = loader.load_file(tmp_path)
        
        if not doc_data:
            raise HTTPException(
                status_code=400,
                detail="Failed to load document"
            )
        
        # Add document to RAG system
        initial_count = rag_engine.document_count
        rag_engine.add_documents([doc_data])
        new_count = rag_engine.document_count
        chunks_created = new_count - initial_count
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        # Save index in background
        if background_tasks:
            background_tasks.add_task(rag_engine.save_index)
        
        logger.info(
            "Document processed",
            filename=file.filename,
            chunks=chunks_created
        )
        
        return UploadResponse(
            message="Document uploaded and processed successfully",
            filename=file.filename,
            chunks_created=chunks_created
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Document upload failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {str(e)}"
        )


@app.delete("/documents")
async def clear_documents():
    """
    Clear all documents from the system.
    
    Returns:
        Confirmation message
    """
    logger.warning("Clearing all documents")
    
    try:
        rag_engine.clear()
        return {"message": "All documents cleared"}
        
    except Exception as e:
        logger.error("Failed to clear documents", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear documents: {str(e)}"
        )


@app.get("/stats")
async def get_stats():
    """
    Get system statistics.
    
    Returns:
        Dictionary with system statistics
    """
    return {
        "document_count": rag_engine.document_count,
        "embedding_model": config.embedding.model_name,
        "embedding_dimension": config.embedding.dimension,
        "llm_model": config.llm.model_name,
    }


def start_server():
    """Start the API server."""
    uvicorn.run(
        "src.api.app:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.debug,
    )


if __name__ == "__main__":
    start_server()
