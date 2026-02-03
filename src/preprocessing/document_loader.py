"""
Document loading and format conversion.

Supports multiple document formats (PDF, TXT, JSON) with unified interface
for ingestion into the RAG pipeline.
"""

import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from pypdf import PdfReader

from ..utils.logger import get_logger


logger = get_logger(__name__)


class DocumentLoader:
    """
    Loads documents from various formats.
    
    Supports:
    - PDF files
    - Text files
    - JSON documents
    - Batch loading from directories
    """
    
    def __init__(self):
        """Initialize document loader."""
        self.supported_extensions = {".pdf", ".txt", ".json"}
        logger.info("Document loader initialized")
    
    def load_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Load text from PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary with text and metadata
        """
        logger.debug("Loading PDF", file_path=file_path)
        
        try:
            reader = PdfReader(file_path)
            
            # Extract text from all pages
            text_parts = []
            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text()
                text_parts.append(page_text)
            
            text = "\n\n".join(text_parts)
            
            # Extract metadata
            metadata = {
                "source": file_path,
                "filename": os.path.basename(file_path),
                "format": "pdf",
                "num_pages": len(reader.pages),
            }
            
            # Add PDF metadata if available
            if reader.metadata:
                metadata.update({
                    "title": reader.metadata.get("/Title", ""),
                    "author": reader.metadata.get("/Author", ""),
                    "subject": reader.metadata.get("/Subject", ""),
                })
            
            logger.info(
                "PDF loaded",
                file_path=file_path,
                pages=len(reader.pages),
                text_length=len(text)
            )
            
            return {"text": text, "metadata": metadata}
            
        except Exception as e:
            logger.error("Failed to load PDF", file_path=file_path, error=str(e))
            raise
    
    def load_text(self, file_path: str) -> Dict[str, Any]:
        """
        Load text from text file.
        
        Args:
            file_path: Path to text file
            
        Returns:
            Dictionary with text and metadata
        """
        logger.debug("Loading text file", file_path=file_path)
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            metadata = {
                "source": file_path,
                "filename": os.path.basename(file_path),
                "format": "txt",
            }
            
            logger.info(
                "Text file loaded",
                file_path=file_path,
                text_length=len(text)
            )
            
            return {"text": text, "metadata": metadata}
            
        except Exception as e:
            logger.error("Failed to load text file", file_path=file_path, error=str(e))
            raise
    
    def load_json(self, file_path: str) -> Dict[str, Any]:
        """
        Load document from JSON file.
        
        Expected JSON format:
        {
            "text": "document content",
            "metadata": {...}
        }
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Dictionary with text and metadata
        """
        logger.debug("Loading JSON file", file_path=file_path)
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Extract text and metadata
            text = data.get("text", "")
            metadata = data.get("metadata", {})
            
            # Add file information to metadata
            metadata.update({
                "source": file_path,
                "filename": os.path.basename(file_path),
                "format": "json",
            })
            
            logger.info(
                "JSON file loaded",
                file_path=file_path,
                text_length=len(text)
            )
            
            return {"text": text, "metadata": metadata}
            
        except Exception as e:
            logger.error("Failed to load JSON file", file_path=file_path, error=str(e))
            raise
    
    def load_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load document from file (auto-detect format).
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with text and metadata, or None if format unsupported
        """
        extension = Path(file_path).suffix.lower()
        
        if extension not in self.supported_extensions:
            logger.warning(
                "Unsupported file format",
                file_path=file_path,
                extension=extension
            )
            return None
        
        if extension == ".pdf":
            return self.load_pdf(file_path)
        elif extension == ".txt":
            return self.load_text(file_path)
        elif extension == ".json":
            return self.load_json(file_path)
        
        return None
    
    def load_directory(
        self,
        directory_path: str,
        recursive: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory_path: Path to directory
            recursive: Whether to search subdirectories
            
        Returns:
            List of documents with text and metadata
        """
        logger.info(
            "Loading documents from directory",
            directory=directory_path,
            recursive=recursive
        )
        
        documents = []
        
        if recursive:
            file_paths = Path(directory_path).rglob("*")
        else:
            file_paths = Path(directory_path).glob("*")
        
        for file_path in file_paths:
            if file_path.is_file():
                doc = self.load_file(str(file_path))
                if doc:
                    documents.append(doc)
        
        logger.info(
            "Directory loading complete",
            directory=directory_path,
            n_documents=len(documents)
        )
        
        return documents
