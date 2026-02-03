# Quick Start Guide

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git
- 4GB RAM minimum
- (Optional) OpenAI API key for LLM functionality

## Installation (5 minutes)

### Step 1: Clone or Download the Project

```bash
# If you have the project as a zip file
unzip ai-document-intelligence.zip
cd ai-document-intelligence

# Or if cloning from Git
git clone <your-repo-url>
cd ai-document-intelligence
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- LangChain (LLM framework)
- FAISS (vector database)
- Sentence Transformers (embeddings)
- FastAPI (API framework)
- OpenAI (LLM integration)
- And all other dependencies

### Step 4: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env file
# Minimum required: Set OPENAI_API_KEY if you want LLM responses
# Otherwise, system will work with mock responses
```

## Running the Example (2 minutes)

### Quick Demo Script

```bash
python example_usage.py
```

This will:
1. Initialize the RAG engine
2. Load sample documents
3. Create embeddings
4. Store in vector database
5. Run example queries
6. Display results

**Expected Output:**
```
================================================================================
AI Document Intelligence System - Demo
================================================================================

1. Initializing RAG engine...
   ✓ RAG engine initialized
   - Embedding model: all-MiniLM-L6-v2
   - Embedding dimension: 384

2. Creating sample documents...
   ✓ Created 3 sample documents

3. Processing and indexing documents...
   ✓ Documents indexed successfully
   - Total chunks: 15

4. Running example queries...

   Query 1: What is Python programming language?
   --------------------------------------------------------------------------
   Answer: Python is a high-level, interpreted programming language known...
   Sources used: 2

   ... (additional queries)

5. Saving vector index...
   ✓ Index saved successfully

6. Testing index persistence...
   ✓ Index loaded successfully
   - Documents in loaded index: 15

================================================================================
Demo completed successfully!
================================================================================
```

## Starting the API Server (1 minute)

```bash
# Start the FastAPI server
python -m uvicorn src.api.app:app --reload
```

Server will start at: http://localhost:8000

### Test the API

**1. Open API Documentation:**
Visit http://localhost:8000/docs in your browser

**2. Health Check:**
```bash
curl http://localhost:8000/
```

**3. Upload a Document:**
```bash
curl -X POST "http://localhost:8000/documents" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/documents/ai_overview.txt"
```

**4. Query the System:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is machine learning?",
    "n_results": 5,
    "include_sources": true
  }'
```

## Using in Python Code (2 minutes)

```python
from src.core.rag_engine import RAGEngine

# Initialize
engine = RAGEngine()

# Add documents
documents = [
    {
        "text": "Your document text here...",
        "metadata": {"source": "example.txt"}
    }
]
engine.add_documents(documents)

# Query
result = engine.query("Your question here?")
print(result["answer"])
print(f"Used {result['n_sources']} sources")

# Save for later use
engine.save_index()
```

## Running Tests (1 minute)

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/unit/test_core.py -v
```

## Common Issues & Solutions

### Issue: "ModuleNotFoundError"
**Solution:** Make sure virtual environment is activated and dependencies installed
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Issue: "OPENAI_API_KEY not set"
**Solution:** System will work in mock mode. For real LLM:
1. Get API key from https://platform.openai.com/api-keys
2. Add to .env file: `OPENAI_API_KEY=your_key_here`

### Issue: "FAISS import error"
**Solution:** Install FAISS for your platform:
```bash
# CPU version (default)
pip install faiss-cpu

# GPU version (if you have CUDA)
pip install faiss-gpu
```

### Issue: "Port 8000 already in use"
**Solution:** Change port in .env or command:
```bash
python -m uvicorn src.api.app:app --reload --port 8001
```

## Next Steps

1. **Customize**: Modify configuration in `.env` file
2. **Add Documents**: Place files in `data/documents/` directory
3. **Explore API**: Use the interactive docs at `/docs`
4. **Read Code**: Check out `src/core/rag_engine.py` for main logic
5. **Deploy**: See deployment guides in `infrastructure/` directory

## Project Structure Quick Reference

```
ai-document-intelligence/
├── src/                    # Source code
│   ├── core/              # Core RAG components
│   ├── preprocessing/     # Document processing
│   ├── api/               # FastAPI application
│   └── utils/             # Utilities
├── tests/                 # Test suite
├── data/                  # Data storage
│   ├── documents/         # Input documents
│   └── vector_db/         # Vector index
├── infrastructure/        # Deployment configs
│   ├── terraform/         # AWS IaC
│   └── docker/            # Containerization
├── example_usage.py       # Demo script
└── requirements.txt       # Dependencies
```

## Getting Help

- **Documentation**: See README.md and TECHNICAL_GUIDE.md
- **Interview Prep**: Check INTERVIEW_PREP.md
- **Code Comments**: All functions have docstrings
- **API Docs**: http://localhost:8000/docs when server running

## Performance Notes

- **First run**: Takes longer (downloads embedding model ~80MB)
- **Subsequent runs**: Much faster (model cached)
- **GPU acceleration**: Set `DEVICE=cuda` in .env if you have NVIDIA GPU
- **Large documents**: Increase chunk_size in config for better performance

## What You've Built

✅ Production-grade RAG system
✅ REST API with FastAPI
✅ Vector database with FAISS
✅ Document processing pipeline
✅ Comprehensive testing
✅ Cloud deployment ready
✅ Monitoring and logging
✅ Infrastructure as code

**Total lines of code**: ~2,500 lines of production Python

Enjoy exploring your AI Document Intelligence System! 🚀
