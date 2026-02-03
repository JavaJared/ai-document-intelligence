"""
Unit tests for core components.

Tests embedding generation, vector store operations, and document processing.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.core.embeddings import EmbeddingGenerator
from src.core.vector_store import VectorStore, Document
from src.preprocessing.chunker import TextChunker, DocumentProcessor, ChunkConfig


class TestEmbeddingGenerator:
    """Tests for embedding generation."""
    
    def test_initialization(self):
        """Test embedding generator initialization."""
        generator = EmbeddingGenerator()
        assert generator.dimension > 0
        assert generator.model_name is not None
    
    def test_encode_single_text(self):
        """Test encoding a single text."""
        generator = EmbeddingGenerator()
        text = "This is a test sentence."
        
        embedding = generator.encode(text)
        
        assert embedding.shape == (generator.dimension,)
        assert isinstance(embedding, np.ndarray)
    
    def test_encode_multiple_texts(self):
        """Test encoding multiple texts."""
        generator = EmbeddingGenerator()
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        
        embeddings = generator.encode(texts)
        
        assert embeddings.shape == (len(texts), generator.dimension)
        assert isinstance(embeddings, np.ndarray)
    
    def test_similarity_calculation(self):
        """Test similarity calculation between embeddings."""
        generator = EmbeddingGenerator()
        
        text1 = "Python programming language"
        text2 = "Python programming tutorial"
        text3 = "Cooking recipe for pasta"
        
        emb1 = generator.encode(text1)
        emb2 = generator.encode(text2)
        emb3 = generator.encode(text3)
        
        # Similar texts should have higher similarity
        sim_12 = generator.similarity(emb1, emb2)
        sim_13 = generator.similarity(emb1, emb3)
        
        assert sim_12 > sim_13
        assert -1 <= sim_12 <= 1
        assert -1 <= sim_13 <= 1


class TestVectorStore:
    """Tests for vector store operations."""
    
    def test_initialization(self):
        """Test vector store initialization."""
        dimension = 384
        store = VectorStore(dimension=dimension)
        
        assert store.dimension == dimension
        assert store.size == 0
    
    def test_add_documents(self):
        """Test adding documents to the store."""
        dimension = 384
        store = VectorStore(dimension=dimension)
        
        # Create test documents
        docs = [
            Document("Test document 1", {"source": "test"}),
            Document("Test document 2", {"source": "test"}),
        ]
        embeddings = np.random.rand(len(docs), dimension).astype(np.float32)
        
        store.add_documents(docs, embeddings)
        
        assert store.size == len(docs)
    
    def test_search(self):
        """Test searching for similar documents."""
        dimension = 384
        store = VectorStore(dimension=dimension)
        
        # Add test documents
        docs = [
            Document("Python programming", {"topic": "tech"}),
            Document("Machine learning basics", {"topic": "tech"}),
            Document("Cooking pasta recipe", {"topic": "food"}),
        ]
        embeddings = np.random.rand(len(docs), dimension).astype(np.float32)
        store.add_documents(docs, embeddings)
        
        # Search with a query embedding
        query_embedding = embeddings[0]  # Use first embedding as query
        results = store.search(query_embedding, k=2)
        
        assert len(results) <= 2
        assert all(isinstance(doc, Document) for doc, _ in results)
        assert all(isinstance(score, float) for _, score in results)
    
    def test_clear(self):
        """Test clearing the vector store."""
        dimension = 384
        store = VectorStore(dimension=dimension)
        
        # Add documents
        docs = [Document("Test", {})]
        embeddings = np.random.rand(1, dimension).astype(np.float32)
        store.add_documents(docs, embeddings)
        
        assert store.size == 1
        
        # Clear
        store.clear()
        assert store.size == 0


class TestTextChunker:
    """Tests for text chunking."""
    
    def test_chunk_text(self):
        """Test chunking text into segments."""
        config = ChunkConfig(chunk_size=100, chunk_overlap=20)
        chunker = TextChunker(config)
        
        text = " ".join(["This is sentence number {}.".format(i) for i in range(20)])
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) > 1
        assert all(isinstance(chunk, Document) for chunk in chunks)
    
    def test_chunk_with_metadata(self):
        """Test chunking preserves metadata."""
        chunker = TextChunker()
        
        text = "Long text " * 100
        metadata = {"source": "test.txt", "author": "John Doe"}
        
        chunks = chunker.chunk_text(text, metadata)
        
        for chunk in chunks:
            assert chunk.metadata["source"] == "test.txt"
            assert chunk.metadata["author"] == "John Doe"
            assert "chunk_index" in chunk.metadata
    
    def test_small_text_no_chunking(self):
        """Test that small text is not chunked."""
        config = ChunkConfig(min_chunk_size=100)
        chunker = TextChunker(config)
        
        text = "Short text."
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) == 0  # Too small to chunk


class TestDocumentProcessor:
    """Tests for document processing pipeline."""
    
    def test_process_document(self):
        """Test processing a document."""
        processor = DocumentProcessor()
        
        text = "This is a test document. " * 50
        metadata = {"source": "test.pdf"}
        
        chunks = processor.process_document(text, metadata)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in chunks)
    
    def test_process_with_cleaning(self):
        """Test document processing with text cleaning."""
        processor = DocumentProcessor()
        
        text = "This   has    extra    spaces.   And weird\t\ttabs."
        chunks = processor.process_document(text, clean=True)
        
        # Check that extra spaces are normalized
        for chunk in chunks:
            assert "   " not in chunk.content
            assert "\t\t" not in chunk.content
    
    def test_process_multiple_documents(self):
        """Test processing multiple documents."""
        processor = DocumentProcessor()
        
        documents = [
            {"text": "First document. " * 30, "metadata": {"id": 1}},
            {"text": "Second document. " * 30, "metadata": {"id": 2}},
        ]
        
        chunks = processor.process_multiple(documents)
        
        assert len(chunks) > 0
        # Verify chunks from both documents are present
        chunk_ids = {chunk.metadata.get("id") for chunk in chunks}
        assert 1 in chunk_ids and 2 in chunk_ids


class TestDocument:
    """Tests for Document class."""
    
    def test_document_creation(self):
        """Test creating a document."""
        content = "Test content"
        metadata = {"source": "test.txt"}
        
        doc = Document(content, metadata)
        
        assert doc.content == content
        assert doc.metadata == metadata
        assert doc.doc_id is not None
    
    def test_document_id_generation(self):
        """Test automatic ID generation."""
        doc1 = Document("Same content")
        doc2 = Document("Same content")
        doc3 = Document("Different content")
        
        # Same content should generate same ID
        assert doc1.doc_id == doc2.doc_id
        # Different content should generate different ID
        assert doc1.doc_id != doc3.doc_id


@pytest.fixture
def mock_embedding_generator():
    """Fixture for mocked embedding generator."""
    with patch('src.core.embeddings.EmbeddingGenerator') as mock:
        generator = Mock()
        generator.dimension = 384
        generator.encode.return_value = np.random.rand(384).astype(np.float32)
        mock.return_value = generator
        yield generator


def test_integration_embedding_and_vector_store(mock_embedding_generator):
    """Integration test for embeddings and vector store."""
    dimension = 384
    store = VectorStore(dimension=dimension)
    
    # Create documents
    docs = [
        Document("Test document 1", {"id": 1}),
        Document("Test document 2", {"id": 2}),
    ]
    
    # Generate embeddings
    embeddings = np.random.rand(len(docs), dimension).astype(np.float32)
    
    # Add to store
    store.add_documents(docs, embeddings)
    
    # Search
    query_embedding = np.random.rand(dimension).astype(np.float32)
    results = store.search(query_embedding, k=2)
    
    assert len(results) == 2
    assert all(doc.metadata.get("id") in [1, 2] for doc, _ in results)
