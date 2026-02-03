"""
Document preprocessing and chunking strategies.

Handles text extraction, cleaning, and intelligent chunking with overlap
for optimal retrieval performance in RAG systems.
"""

from typing import List, Optional, Dict, Any
import re
from dataclasses import dataclass

from ..core.vector_store import Document
from ..utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class ChunkConfig:
    """Configuration for text chunking."""
    chunk_size: int = 500  # characters
    chunk_overlap: int = 50  # characters
    min_chunk_size: int = 100  # minimum viable chunk size


class TextChunker:
    """
    Intelligent text chunking with configurable strategies.
    
    Implements overlapping chunks to maintain context across boundaries
    and respects sentence boundaries for better semantic coherence.
    """
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        """
        Initialize text chunker.
        
        Args:
            config: Chunking configuration
        """
        self.config = config or ChunkConfig()
        logger.info(
            "Text chunker initialized",
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap
        )
    
    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Chunk text into overlapping segments.
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk
            
        Returns:
            List of Document objects representing chunks
        """
        if not text or len(text) < self.config.min_chunk_size:
            return []
        
        # Split into sentences for better boundaries
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence exceeds chunk size, save current chunk
            if current_length + sentence_length > self.config.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata["chunk_index"] = len(chunks)
                
                chunks.append(Document(
                    content=chunk_text,
                    metadata=chunk_metadata
                ))
                
                # Keep overlap sentences for context
                overlap_length = 0
                overlap_sentences = []
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= self.config.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata["chunk_index"] = len(chunks)
            
            chunks.append(Document(
                content=chunk_text,
                metadata=chunk_metadata
            ))
        
        logger.debug(
            "Text chunked",
            original_length=len(text),
            n_chunks=len(chunks)
        )
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitter (can be enhanced with spaCy/NLTK)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


class DocumentCleaner:
    """
    Cleans and normalizes document text.
    
    Removes noise, normalizes whitespace, and handles special characters
    to improve embedding quality and retrieval accuracy.
    """
    
    @staticmethod
    def clean(text: str) -> str:
        """
        Clean document text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove multiple whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.,!?;:\-\(\)]', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    @staticmethod
    def remove_headers_footers(text: str) -> str:
        """
        Remove common headers and footers from documents.
        
        Args:
            text: Text with potential headers/footers
            
        Returns:
            Text with headers/footers removed
        """
        # Remove page numbers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # Remove common footer patterns
        text = re.sub(r'\n\s*Page \d+ of \d+\s*\n', '\n', text)
        
        return text


class DocumentProcessor:
    """
    Complete document processing pipeline.
    
    Combines cleaning, chunking, and metadata extraction into a
    single unified interface for document preparation.
    """
    
    def __init__(self, chunk_config: Optional[ChunkConfig] = None):
        """
        Initialize document processor.
        
        Args:
            chunk_config: Configuration for chunking
        """
        self.cleaner = DocumentCleaner()
        self.chunker = TextChunker(chunk_config)
        logger.info("Document processor initialized")
    
    def process_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        clean: bool = True
    ) -> List[Document]:
        """
        Process a document into chunks ready for embedding.
        
        Args:
            text: Raw document text
            metadata: Document metadata
            clean: Whether to clean the text
            
        Returns:
            List of processed document chunks
        """
        logger.debug(
            "Processing document",
            text_length=len(text),
            has_metadata=metadata is not None
        )
        
        # Clean text if requested
        if clean:
            text = self.cleaner.clean(text)
            text = self.cleaner.remove_headers_footers(text)
        
        # Chunk document
        chunks = self.chunker.chunk_text(text, metadata)
        
        logger.info(
            "Document processed",
            n_chunks=len(chunks)
        )
        
        return chunks
    
    def process_multiple(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Document]:
        """
        Process multiple documents.
        
        Args:
            documents: List of dicts with 'text' and 'metadata' keys
            
        Returns:
            List of all processed chunks
        """
        all_chunks = []
        
        for doc in documents:
            chunks = self.process_document(
                text=doc.get("text", ""),
                metadata=doc.get("metadata", {})
            )
            all_chunks.extend(chunks)
        
        logger.info(
            "Multiple documents processed",
            n_documents=len(documents),
            total_chunks=len(all_chunks)
        )
        
        return all_chunks
