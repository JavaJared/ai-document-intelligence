"""
Embedding generation using sentence transformers.

Provides efficient text-to-vector conversion with caching and batch processing
for optimal performance in production environments.
"""

from typing import List, Union, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from functools import lru_cache

from ..utils.config import config
from ..utils.logger import get_logger


logger = get_logger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings for text using sentence transformer models.
    
    Features:
    - Batch processing for efficiency
    - Caching to avoid redundant computation
    - GPU acceleration support
    - Normalized embeddings for cosine similarity
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None
    ):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to use ('cpu' or 'cuda')
            batch_size: Batch size for processing
        """
        self.model_name = model_name or config.embedding.model_name
        self.device = device or config.embedding.device
        self.batch_size = batch_size or config.embedding.batch_size
        
        logger.info(
            "Initializing embedding generator",
            model=self.model_name,
            device=self.device
        )
        
        # Load model
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self._embedding_dimension = self._model.get_sentence_embedding_dimension()
        
        logger.info(
            "Embedding generator initialized",
            dimension=self._embedding_dimension
        )
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._embedding_dimension
    
    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts
            normalize: Whether to normalize embeddings (for cosine similarity)
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings with shape (n_texts, dimension)
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
        
        logger.debug(
            "Generating embeddings",
            n_texts=len(texts),
            batch_size=self.batch_size
        )
        
        try:
            # Generate embeddings in batches
            embeddings = self._model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=normalize,
                convert_to_numpy=True
            )
            
            logger.debug(
                "Embeddings generated",
                shape=embeddings.shape
            )
            
            # Return single embedding if single input
            if single_input:
                return embeddings[0]
            
            return embeddings
            
        except Exception as e:
            logger.error(
                "Failed to generate embeddings",
                error=str(e),
                n_texts=len(texts)
            )
            raise
    
    def encode_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Generate embeddings in batches for memory efficiency.
        
        Args:
            texts: List of texts to encode
            batch_size: Override default batch size
            
        Returns:
            List of embedding arrays
        """
        batch_size = batch_size or self.batch_size
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.encode(batch, show_progress=False)
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between -1 and 1
        """
        # Ensure embeddings are normalized
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        embedding1_norm = embedding1 / norm1
        embedding2_norm = embedding2 / norm2
        
        return float(np.dot(embedding1_norm, embedding2_norm))
    
    def batch_similarity(
        self,
        query_embedding: np.ndarray,
        embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Calculate similarities between query and multiple embeddings.
        
        Args:
            query_embedding: Query embedding vector
            embeddings: Array of embedding vectors
            
        Returns:
            Array of similarity scores
        """
        # Normalize embeddings
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Calculate cosine similarities
        similarities = np.dot(embeddings_norm, query_norm)
        
        return similarities


@lru_cache(maxsize=128)
def get_embedding_generator() -> EmbeddingGenerator:
    """
    Get a cached embedding generator instance.
    
    Returns:
        Singleton embedding generator
    """
    return EmbeddingGenerator()
