# AI Document Intelligence System

A production-grade Generative AI platform for intelligent document processing using RAG (Retrieval Augmented Generation), vector databases, and LLMs.

## 🎯 Project Overview

This system demonstrates enterprise-level AI engineering capabilities including:
- **Generative AI Applications**: RAG-based document Q&A using LLMs
- **Vector Database Integration**: FAISS for semantic search and retrieval
- **Cloud-Native Architecture**: AWS-ready with infrastructure-as-code
- **Production Best Practices**: Comprehensive testing, logging, and error handling

## 🏗️ Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│  Documents  │─────▶│   Embedding  │─────▶│   Vector    │
│   (Input)   │      │   Pipeline   │      │   Database  │
└─────────────┘      └──────────────┘      └─────────────┘
                                                    │
                                                    ▼
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Response  │◀─────│  LLM + RAG   │◀─────│  Semantic   │
│   (Output)  │      │   Engine     │      │   Search    │
└─────────────┘      └──────────────┘      └─────────────┘
```

## 🚀 Key Features

### 1. Document Processing Pipeline
- Multi-format support (PDF, TXT, JSON)
- Intelligent text chunking with overlap
- Metadata extraction and tagging
- Preprocessing and data curation

### 2. Vector Database & Retrieval
- FAISS vector store for efficient similarity search
- Sentence transformers for embeddings
- Context-aware retrieval with relevance scoring
- Support for large-scale document collections

### 3. LLM Integration
- OpenAI GPT integration (easily swappable)
- Context injection and prompt engineering
- Streaming responses for better UX
- Response validation and quality checks

### 4. Cloud Infrastructure
- AWS Lambda deployment configuration
- S3 integration for document storage
- CloudFormation/Terraform templates
- Environment-based configuration

### 5. Software Engineering Best Practices
- Comprehensive unit and integration tests
- Type hints and static analysis
- Structured logging and monitoring
- Design patterns (Factory, Strategy, Repository)
- Code documentation and docstrings

## 📁 Project Structure

```
ai-document-intelligence/
├── src/
│   ├── core/
│   │   ├── embeddings.py      # Embedding generation
│   │   ├── vector_store.py    # Vector database operations
│   │   ├── llm_client.py      # LLM integration
│   │   └── rag_engine.py      # RAG orchestration
│   ├── preprocessing/
│   │   ├── document_loader.py # Document ingestion
│   │   ├── chunker.py         # Text chunking strategies
│   │   └── cleaner.py         # Data preprocessing
│   ├── api/
│   │   └── app.py             # FastAPI REST endpoint
│   └── utils/
│       ├── config.py          # Configuration management
│       ├── logger.py          # Structured logging
│       └── validators.py      # Input validation
├── tests/
│   ├── unit/                  # Unit tests
│   └── integration/           # Integration tests
├── infrastructure/
│   ├── terraform/             # IaC for AWS
│   └── docker/                # Container configuration
├── data/
│   ├── documents/             # Sample documents
│   └── vector_db/             # Vector store persistence
├── requirements.txt
├── setup.py
└── README.md
```

## 🛠️ Tech Stack

- **Python 3.9+**: Core language
- **LangChain**: LLM framework and RAG orchestration
- **FAISS**: Vector similarity search
- **Sentence Transformers**: Embedding models
- **FastAPI**: REST API framework
- **AWS SDK (boto3)**: Cloud integration
- **pytest**: Testing framework
- **Docker**: Containerization

## 💻 Installation

```bash
# Clone the repository
git clone <repository-url>
cd ai-document-intelligence

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

## 🚀 Quick Start

```python
from src.core.rag_engine import RAGEngine
from src.preprocessing.document_loader import DocumentLoader

# Initialize the system
rag_engine = RAGEngine()

# Load and process documents
loader = DocumentLoader()
documents = loader.load_directory("data/documents")
rag_engine.add_documents(documents)

# Query the system
response = rag_engine.query("What are the key findings in the Q3 report?")
print(response)
```

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test suite
pytest tests/unit/test_embeddings.py
```

## 🌐 API Usage

```bash
# Start the API server
uvicorn src.api.app:app --reload

# Query endpoint
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize the documents"}'

# Upload document
curl -X POST "http://localhost:8000/documents" \
  -F "file=@document.pdf"
```

## 📊 Performance Considerations

- **Embedding Caching**: Reduces redundant computation
- **Batch Processing**: Efficient handling of multiple documents
- **Async Operations**: Non-blocking I/O for API endpoints
- **Connection Pooling**: Optimized database connections

## 🔐 Security

- API key management via environment variables
- Input validation and sanitization
- Rate limiting on API endpoints
- Secure document storage with encryption

## 📈 Future Enhancements

- [ ] Multi-modal support (images, tables)
- [ ] Fine-tuned embedding models
- [ ] Distributed vector database (Pinecone, Weaviate)
- [ ] Real-time document processing pipeline
- [ ] Advanced prompt engineering with few-shot learning
- [ ] Observability with OpenTelemetry

## 👥 Contributing

This project follows software development best practices:
- Code reviews required for all changes
- Maintain test coverage above 80%
- Follow PEP 8 style guidelines
- Document all public APIs

## 📝 License

MIT License
