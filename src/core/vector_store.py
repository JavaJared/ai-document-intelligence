"""
Vector database implementation using FAISS for efficient similarity search.

Provides persistent storage and retrieval of document embeddings with
support for large-scale collections and efficient indexing strategies.
"""

from typing import List, Dict, Any, Optional, Tuple
import os
import pickle
import numpy as np
import faiss

from ..utils.config import config
from ..utils.logger import get_logger


logger = get_logger(__name__)


class Document:
    """
    Represents a document with its content and metadata.
    """
    
    def __init__(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ):
        """
        Initialize a document.
        
        Args:
            content: Text content of the document
            metadata: Additional metadata (source, page, etc.)
            doc_id: Unique identifier for the document
        """
        self.content = content
        self.metadata = metadata or {}
        self.doc_id = doc_id or self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate a unique ID for the document."""
        import hashlib
        return hashlib.md5(self.content.encode()).hexdigest()
    
    def __repr__(self) -> str:
        return f"Document(id={self.doc_id}, content_length={len(self.content)})"


class VectorStore:
    """
    Vector database for storing and retrieving document embeddings.
    
    Uses FAISS for efficient similarity search with support for:
    - Multiple index types (Flat, IVF, HNSW)
    - Persistence to disk
    - Metadata filtering
    - Batch operations
    """
    
    def __init__(
        self,
        dimension: int,
        index_path: Optional[str] = None,
        index_type: str = "Flat"
    ):
        """
        Initialize vector store.
        
        Args:
            dimension: Embedding dimension
            index_path: Path to save/load index
            index_type: Type of FAISS index (Flat, IVF, HNSW)
        """
        self.dimension = dimension
        self.index_path = index_path or config.vector_store.index_path
        self.index_type = index_type
        
        # Initialize FAISS index
        self._index = self._create_index()
        
        # Storage for documents and metadata
        self._documents: List[Document] = []
        self._embeddings: Optional[np.ndarray] = None
        
        logger.info(
            "Vector store initialized",
            dimension=dimension,
            index_type=index_type
        )
    
    def _create_index(self) -> faiss.Index:
        """
        Create appropriate FAISS index based on type.
        
        Returns:
            FAISS index instance
        """
        if self.index_type == "Flat":
            # L2 distance for exact search
            index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IVF":
            # Inverted file index for faster approximate search
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        elif self.index_type == "HNSW":
            # Hierarchical Navigable Small World for fast approximate search
            index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        return index
    
    def add_documents(
        self,
        documents: List[Document],
        embeddings: np.ndarray
    ) -> None:
        """
        Add documents and their embeddings to the store.
        
        Args:
            documents: List of documents
            embeddings: Corresponding embeddings array
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        logger.info(
            "Adding documents to vector store",
            n_documents=len(documents)
        )
        
        # Convert embeddings to float32 for FAISS
        embeddings = embeddings.astype(np.float32)
        
        # Train index if needed (for IVF)
        if isinstance(self._index, faiss.IndexIVFFlat) and not self._index.is_trained:
            logger.info("Training IVF index")
            self._index.train(embeddings)
        
        # Add to index
        self._index.add(embeddings)
        
        # Store documents and embeddings
        self._documents.extend(documents)
        
        if self._embeddings is None:
            self._embeddings = embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, embeddings])
        
        logger.info(
            "Documents added",
            total_documents=len(self._documents),
            index_size=self._index.ntotal
        )
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of (Document, similarity_score) tuples
        """
        if self._index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []
        
        # Ensure query is 2D array with float32
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        
        # Search index
        distances, indices = self._index.search(query_embedding, k)
        
        # Convert distances to similarity scores (for L2 distance)
        # Lower distance = higher similarity
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx == -1:  # FAISS returns -1 for missing results
                continue
            
            document = self._documents[idx]
            
            # Apply metadata filters if specified
            if filter_metadata:
                if not all(
                    document.metadata.get(key) == value
                    for key, value in filter_metadata.items()
                ):
                    continue
            
            # Convert L2 distance to similarity score (0-1 range)
            similarity = 1 / (1 + distance)
            results.append((document, float(similarity)))
        
        logger.debug(
            "Search completed",
            n_results=len(results),
            query_shape=query_embedding.shape
        )
        
        return results
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save vector store to disk.
        
        Args:
            path: Directory path to save to
        """
        path = path or self.index_path
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        index_file = os.path.join(path, "index.faiss")
        faiss.write_index(self._index, index_file)
        
        # Save documents and metadata
        docs_file = os.path.join(path, "documents.pkl")
        with open(docs_file, "wb") as f:
            pickle.dump({
                "documents": self._documents,
                "embeddings": self._embeddings,
                "dimension": self.dimension,
                "index_type": self.index_type
            }, f)
        
        logger.info("Vector store saved", path=path)
    
    def load(self, path: Optional[str] = None) -> None:
        """
        Load vector store from disk.
        
        Args:
            path: Directory path to load from
        """
        path = path or self.index_path
        
        index_file = os.path.join(path, "index.faiss")
        docs_file = os.path.join(path, "documents.pkl")
        
        if not os.path.exists(index_file) or not os.path.exists(docs_file):
            logger.warning("Vector store files not found", path=path)
            return
        
        # Load FAISS index
        self._index = faiss.read_index(index_file)
        
        # Load documents and metadata
        with open(docs_file, "rb") as f:
            data = pickle.load(f)
            self._documents = data["documents"]
            self._embeddings = data["embeddings"]
        
        logger.info(
            "Vector store loaded",
            path=path,
            n_documents=len(self._documents)
        )
    
    def clear(self) -> None:
        """Clear all documents from the store."""
        self._index = self._create_index()
        self._documents = []
        self._embeddings = None
        logger.info("Vector store cleared")
    
    @property
    def size(self) -> int:
        """Get number of documents in store."""
        return len(self._documents)
