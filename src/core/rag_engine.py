"""
RAG (Retrieval Augmented Generation) engine.

Orchestrates the complete RAG pipeline: document retrieval, context injection,
and response generation with LLMs.
"""

from typing import List, Dict, Any, Optional
import numpy as np

from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore, Document
from .llm_client import LLMClient
from ..preprocessing.document_loader import DocumentLoader
from ..preprocessing.chunker import DocumentProcessor
from ..utils.config import config
from ..utils.logger import get_logger


logger = get_logger(__name__)


class RAGEngine:
    """
    Complete RAG system for document-based question answering.
    
    Workflow:
    1. User submits a query
    2. Query is embedded using sentence transformers
    3. Relevant documents are retrieved from vector store
    4. Context is injected into LLM prompt
    5. LLM generates response using retrieved context
    """
    
    def __init__(
        self,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        vector_store: Optional[VectorStore] = None,
        llm_client: Optional[LLMClient] = None,
        document_processor: Optional[DocumentProcessor] = None,
    ):
        """
        Initialize RAG engine.
        
        Args:
            embedding_generator: Embedding generator instance
            vector_store: Vector store instance
            llm_client: LLM client instance
            document_processor: Document processor instance
        """
        # Initialize components
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        
        self.vector_store = vector_store or VectorStore(
            dimension=self.embedding_generator.dimension
        )
        
        self.llm_client = llm_client or LLMClient()
        self.document_processor = document_processor or DocumentProcessor()
        
        logger.info("RAG engine initialized")
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 32
    ) -> None:
        """
        Add documents to the knowledge base.
        
        Args:
            documents: List of dicts with 'text' and 'metadata' keys
            batch_size: Batch size for embedding generation
        """
        logger.info(
            "Adding documents to RAG system",
            n_documents=len(documents)
        )
        
        # Process documents into chunks
        chunks = self.document_processor.process_multiple(documents)
        
        if not chunks:
            logger.warning("No chunks generated from documents")
            return
        
        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_generator.encode(texts, show_progress=True)
        
        # Add to vector store
        self.vector_store.add_documents(chunks, embeddings)
        
        logger.info(
            "Documents added successfully",
            n_chunks=len(chunks),
            vector_store_size=self.vector_store.size
        )
    
    def add_documents_from_directory(
        self,
        directory_path: str,
        recursive: bool = False
    ) -> None:
        """
        Load and add all documents from a directory.
        
        Args:
            directory_path: Path to directory
            recursive: Whether to search subdirectories
        """
        logger.info("Loading documents from directory", directory=directory_path)
        
        loader = DocumentLoader()
        documents = loader.load_directory(directory_path, recursive)
        
        if documents:
            self.add_documents(documents)
        else:
            logger.warning("No documents found in directory", directory=directory_path)
    
    def query(
        self,
        question: str,
        n_results: int = 5,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: User question
            n_results: Number of documents to retrieve
            include_sources: Whether to include source documents in response
            
        Returns:
            Dictionary with answer and optional sources
        """
        logger.info("Processing query", question=question[:100])
        
        # Generate query embedding
        query_embedding = self.embedding_generator.encode(question)
        
        # Retrieve relevant documents
        results = self.vector_store.search(query_embedding, k=n_results)
        
        if not results:
            logger.warning("No relevant documents found")
            return {
                "answer": "I don't have enough information to answer that question.",
                "sources": []
            }
        
        # Build context from retrieved documents
        context = self._build_context(results)
        
        # Generate prompt
        prompt = self._create_prompt(question, context)
        
        # Generate response
        answer = self.llm_client.generate(
            prompt=prompt,
            system_message="You are a helpful AI assistant. Answer questions based on the provided context. If the context doesn't contain enough information, say so."
        )
        
        response = {
            "answer": answer,
            "n_sources": len(results)
        }
        
        # Include source documents if requested
        if include_sources:
            response["sources"] = [
                {
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "similarity": score
                }
                for doc, score in results
            ]
        
        logger.info(
            "Query processed",
            n_sources=len(results),
            answer_length=len(answer)
        )
        
        return response
    
    def _build_context(
        self,
        results: List[tuple[Document, float]],
        max_context_length: int = 3000
    ) -> str:
        """
        Build context from retrieved documents.
        
        Args:
            results: List of (Document, similarity_score) tuples
            max_context_length: Maximum context length in characters
            
        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0
        
        for doc, score in results:
            # Format document with metadata
            doc_text = f"[Source: {doc.metadata.get('filename', 'unknown')}]\n{doc.content}\n"
            
            # Check if adding this doc would exceed max length
            if current_length + len(doc_text) > max_context_length:
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        context = "\n---\n".join(context_parts)
        
        logger.debug(
            "Context built",
            n_documents=len(context_parts),
            context_length=len(context)
        )
        
        return context
    
    def _create_prompt(self, question: str, context: str) -> str:
        """
        Create prompt for LLM with injected context.
        
        Args:
            question: User question
            context: Retrieved context
            
        Returns:
            Formatted prompt
        """
        prompt = f"""Based on the following context, please answer the question. If the answer cannot be found in the context, say so.

Context:
{context}

Question: {question}

Answer:"""
        
        return prompt
    
    def save_index(self, path: Optional[str] = None) -> None:
        """
        Save vector store to disk.
        
        Args:
            path: Directory path to save to
        """
        self.vector_store.save(path)
        logger.info("Index saved", path=path or config.vector_store.index_path)
    
    def load_index(self, path: Optional[str] = None) -> None:
        """
        Load vector store from disk.
        
        Args:
            path: Directory path to load from
        """
        self.vector_store.load(path)
        logger.info(
            "Index loaded",
            path=path or config.vector_store.index_path,
            size=self.vector_store.size
        )
    
    def clear(self) -> None:
        """Clear all documents from the system."""
        self.vector_store.clear()
        logger.info("RAG engine cleared")
    
    @property
    def document_count(self) -> int:
        """Get number of documents in the system."""
        return self.vector_store.size
