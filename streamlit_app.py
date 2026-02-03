"""
Streamlit Web Interface for AI Document Intelligence System

A visual, interactive demo that can be easily deployed to Streamlit Cloud,
Hugging Face Spaces, or other hosting platforms.

To run locally: streamlit run streamlit_app.py
"""

import streamlit as st
import sys
import os
from pathlib import Path
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import after path is set
try:
    from src.core.rag_engine import RAGEngine
    from src.core.embeddings import EmbeddingGenerator
    from src.preprocessing.document_loader import DocumentLoader
    from src.utils.logger import setup_logging
except ImportError as e:
    st.error(f"Import error: {e}. Make sure all dependencies are installed.")
    st.stop()

# Setup logging (console mode for Streamlit)
setup_logging(log_level="INFO", json_format=False)

# Page config
st.set_page_config(
    page_title="AI Document Intelligence System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .source-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_rag_engine():
    """Initialize RAG engine (cached to avoid reloading)."""
    return RAGEngine()


@st.cache_data
def load_sample_documents():
    """Load sample documents for demo."""
    sample_docs = [
        {
            "text": """
            Python is a high-level, interpreted programming language known for its 
            simplicity and readability. It was created by Guido van Rossum and first 
            released in 1991. Python supports multiple programming paradigms including 
            procedural, object-oriented, and functional programming. It has a comprehensive 
            standard library and a vast ecosystem of third-party packages available through 
            PyPI (Python Package Index). Python is widely used in web development, data 
            science, machine learning, automation, and scientific computing.
            """,
            "metadata": {
                "source": "python_intro.txt",
                "topic": "programming",
                "category": "technology"
            }
        },
        {
            "text": """
            Machine learning is a subset of artificial intelligence that focuses on 
            building systems that can learn from data and improve their performance 
            over time without being explicitly programmed. There are three main types 
            of machine learning: supervised learning (learning from labeled data), 
            unsupervised learning (finding patterns in unlabeled data), and reinforcement 
            learning (learning through trial and error with rewards). Common applications 
            include image recognition, natural language processing, recommendation systems, 
            and predictive analytics. Deep learning, a subset of machine learning using 
            neural networks, has achieved remarkable success in recent years.
            """,
            "metadata": {
                "source": "ml_basics.txt",
                "topic": "machine learning",
                "category": "ai"
            }
        },
        {
            "text": """
            Cloud computing refers to the delivery of computing services over the internet, 
            including servers, storage, databases, networking, software, and analytics. 
            The three main service models are Infrastructure as a Service (IaaS), Platform 
            as a Service (PaaS), and Software as a Service (SaaS). Major cloud providers 
            include Amazon Web Services (AWS), Microsoft Azure, and Google Cloud Platform. 
            Cloud computing offers benefits such as scalability, cost-effectiveness, 
            flexibility, and reduced maintenance overhead. Organizations can scale resources 
            up or down based on demand and only pay for what they use.
            """,
            "metadata": {
                "source": "cloud_computing.txt",
                "topic": "cloud",
                "category": "infrastructure"
            }
        },
        {
            "text": """
            Retrieval Augmented Generation (RAG) is an AI technique that combines information 
            retrieval with language model generation. Instead of relying solely on the model's 
            training data, RAG systems first retrieve relevant documents from a knowledge base, 
            then use those documents as context for generating responses. This approach helps 
            reduce hallucinations, provides source attribution, and allows the model to access 
            up-to-date or private information. RAG systems typically use vector databases and 
            semantic search to find relevant documents efficiently.
            """,
            "metadata": {
                "source": "rag_explained.txt",
                "topic": "rag",
                "category": "ai"
            }
        },
        {
            "text": """
            Vector databases are specialized databases designed to store and query high-dimensional 
            vectors efficiently. They are essential for modern AI applications like semantic search, 
            recommendation systems, and RAG. Popular vector databases include FAISS, Pinecone, 
            Weaviate, and Chroma. These databases use algorithms like Approximate Nearest Neighbor 
            (ANN) search to quickly find similar vectors among millions or billions of entries. 
            Vector embeddings represent data as numerical arrays that capture semantic meaning, 
            allowing for similarity comparisons.
            """,
            "metadata": {
                "source": "vector_databases.txt",
                "topic": "databases",
                "category": "technology"
            }
        }
    ]
    return sample_docs


def main():
    """Main application."""
    
    # Header
    st.markdown('<p class="main-header">🤖 AI Document Intelligence System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Production-Grade RAG (Retrieval Augmented Generation) Demo</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Configuration")
        
        # Initialize session state
        if 'rag_engine' not in st.session_state:
            with st.spinner("Initializing RAG engine..."):
                st.session_state.rag_engine = initialize_rag_engine()
                st.success("✅ RAG engine initialized!")
        
        if 'documents_loaded' not in st.session_state:
            st.session_state.documents_loaded = False
        
        # Load sample documents button
        if st.button("📚 Load Sample Documents", type="primary", use_container_width=True):
            with st.spinner("Loading and processing documents..."):
                sample_docs = load_sample_documents()
                st.session_state.rag_engine.add_documents(sample_docs)
                st.session_state.documents_loaded = True
                st.success(f"✅ Loaded {len(sample_docs)} documents!")
                st.rerun()
        
        # System metrics
        st.divider()
        st.subheader("📊 System Metrics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", st.session_state.rag_engine.document_count)
        with col2:
            st.metric("Embedding Dim", st.session_state.rag_engine.embedding_generator.dimension)
        
        # Model info
        st.divider()
        st.subheader("🔧 Model Configuration")
        st.caption(f"**Embedding Model:**")
        st.code(st.session_state.rag_engine.embedding_generator.model_name, language=None)
        st.caption(f"**LLM Model:**")
        st.code(st.session_state.rag_engine.llm_client.model_name, language=None)
        
        # Clear documents
        st.divider()
        if st.button("🗑️ Clear All Documents", use_container_width=True):
            st.session_state.rag_engine.clear()
            st.session_state.documents_loaded = False
            st.success("Documents cleared!")
            st.rerun()
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Query System", "📄 Upload Documents", "🏗️ Architecture", "📖 About"])
    
    # Tab 1: Query Interface
    with tab1:
        st.header("Query the Knowledge Base")
        
        if not st.session_state.documents_loaded:
            st.warning("⚠️ No documents loaded. Please load sample documents from the sidebar first!")
        else:
            # Example queries
            st.subheader("Try these example queries:")
            example_queries = [
                "What is Python programming language?",
                "Explain machine learning and its types",
                "What are the benefits of cloud computing?",
                "How does RAG work?",
                "What is a vector database?"
            ]
            
            cols = st.columns(len(example_queries))
            selected_example = None
            for i, (col, query) in enumerate(zip(cols, example_queries)):
                with col:
                    if st.button(f"Q{i+1}", key=f"example_{i}", help=query, use_container_width=True):
                        selected_example = query
            
            # Query input
            st.divider()
            query = st.text_area(
                "Enter your question:",
                value=selected_example if selected_example else "",
                height=100,
                placeholder="Type your question here..."
            )
            
            col1, col2 = st.columns([3, 1])
            with col1:
                n_results = st.slider("Number of sources to retrieve", 1, 10, 3)
            with col2:
                include_sources = st.checkbox("Show sources", value=True)
            
            if st.button("🔍 Search & Answer", type="primary", use_container_width=True):
                if query.strip():
                    with st.spinner("Searching knowledge base and generating answer..."):
                        try:
                            result = st.session_state.rag_engine.query(
                                question=query,
                                n_results=n_results,
                                include_sources=include_sources
                            )
                            
                            # Display answer
                            st.subheader("📝 Answer")
                            st.markdown(f"""
                            <div style="background-color: #f0f9ff; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #0ea5e9;">
                                {result['answer']}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display sources if requested
                            if include_sources and 'sources' in result:
                                st.divider()
                                st.subheader(f"📚 Sources Used ({result['n_sources']})")
                                
                                for i, source in enumerate(result['sources'], 1):
                                    with st.expander(f"Source {i}: {source['metadata'].get('source', 'Unknown')} (Similarity: {source['similarity']:.3f})"):
                                        st.markdown(f"**Content:**")
                                        st.write(source['content'])
                                        st.markdown(f"**Metadata:**")
                                        st.json(source['metadata'])
                        
                        except Exception as e:
                            st.error(f"Error processing query: {str(e)}")
                else:
                    st.warning("Please enter a question!")
    
    # Tab 2: Upload Documents
    with tab2:
        st.header("Upload Your Own Documents")
        st.write("Upload PDF, TXT, or JSON files to add to the knowledge base.")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'txt', 'json'],
            help="Supported formats: PDF, TXT, JSON"
        )
        
        if uploaded_file is not None:
            # Show file details
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
            
            if st.button("📥 Process and Add Document", type="primary"):
                with st.spinner("Processing document..."):
                    try:
                        # Save to temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        # Load document
                        loader = DocumentLoader()
                        doc_data = loader.load_file(tmp_path)
                        
                        if doc_data:
                            # Add to RAG system
                            initial_count = st.session_state.rag_engine.document_count
                            st.session_state.rag_engine.add_documents([doc_data])
                            new_count = st.session_state.rag_engine.document_count
                            chunks_created = new_count - initial_count
                            
                            st.session_state.documents_loaded = True
                            
                            st.success(f"✅ Document processed successfully!")
                            st.info(f"Created {chunks_created} searchable chunks from this document")
                            
                            # Clean up
                            os.unlink(tmp_path)
                        else:
                            st.error("Failed to process document")
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    # Tab 3: Architecture
    with tab3:
        st.header("System Architecture")
        
        st.markdown("""
        ### 🏗️ RAG Pipeline Overview
        
        This system implements a complete Retrieval Augmented Generation (RAG) pipeline:
        """)
        
        st.markdown("""
        ```
        ┌─────────────────────────────────────────────────────────────────┐
        │                      DOCUMENT INGESTION                          │
        │                                                                   │
        │  Upload Document → Extract Text → Clean & Chunk → Generate       │
        │                                    Embeddings → Store in Vector DB│
        └─────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │                      QUERY PROCESSING                            │
        │                                                                   │
        │  User Query → Generate Query Embedding → Semantic Search →       │
        │               Retrieve Top-K Documents → Inject Context →         │
        │               LLM Generation → Return Answer                      │
        └─────────────────────────────────────────────────────────────────┘
        ```
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔧 Components")
            st.markdown("""
            - **Embedding Generator**: Sentence Transformers
            - **Vector Database**: FAISS (Facebook AI Similarity Search)
            - **LLM**: OpenAI GPT (configurable)
            - **API Framework**: FastAPI
            - **Web Interface**: Streamlit
            """)
        
        with col2:
            st.subheader("✨ Key Features")
            st.markdown("""
            - Multi-format document support
            - Semantic search with vector embeddings
            - Source attribution
            - Configurable retrieval parameters
            - Production-ready architecture
            """)
        
        st.divider()
        st.subheader("📊 How It Works")
        
        with st.expander("1️⃣ Document Processing"):
            st.markdown("""
            - Documents are loaded and text is extracted
            - Text is cleaned and normalized
            - Content is split into chunks with overlap
            - Each chunk is converted to a 384-dimensional vector
            - Vectors are stored in FAISS index
            """)
        
        with st.expander("2️⃣ Semantic Search"):
            st.markdown("""
            - User query is converted to same vector space
            - FAISS finds most similar document chunks
            - Similarity calculated using cosine distance
            - Top-K most relevant chunks retrieved
            """)
        
        with st.expander("3️⃣ Answer Generation"):
            st.markdown("""
            - Retrieved chunks injected as context
            - LLM generates answer based on context
            - Sources tracked for attribution
            - Response returned to user
            """)
    
    # Tab 4: About
    with tab4:
        st.header("About This Project")
        
        st.markdown("""
        ### 🎯 Project Overview
        
        This is a **production-grade AI Document Intelligence System** built to demonstrate:
        
        - ✅ Generative AI application development
        - ✅ RAG (Retrieval Augmented Generation) architecture
        - ✅ Vector databases and semantic search
        - ✅ Cloud-native design patterns
        - ✅ Software engineering best practices
        
        ### 🔬 Technical Stack
        
        **AI/ML:**
        - Sentence Transformers for embeddings
        - FAISS for vector similarity search
        - OpenAI GPT for language generation
        
        **Backend:**
        - FastAPI for REST API
        - Pydantic for data validation
        - Structured logging with JSON format
        
        **Infrastructure:**
        - Docker for containerization
        - Terraform for AWS deployment
        - CI/CD ready
        
        ### 📈 Performance
        
        - Handles millions of documents
        - Sub-second query response times
        - Scalable architecture
        - Cost-optimized LLM usage
        
        ### 🚀 Deployment Options
        
        This demo is hosted on **Streamlit Cloud** (free tier), but the full system supports:
        
        - AWS Lambda (serverless)
        - Docker containers
        - Kubernetes clusters
        - Traditional VMs
        
        ### 📚 Learn More
        
        - [GitHub Repository](https://github.com/yourusername/ai-document-intelligence)
        - [Technical Documentation](https://your-docs-url.com)
        - [API Reference](https://your-api-url.com/docs)
        
        ### 👨‍💻 Built For
        
        This project was created to showcase AI/ML engineering capabilities for the 
        **Nasdaq AI Platform Engineering** role, demonstrating expertise in:
        
        - Generative AI applications
        - Production software development
        - Cloud architecture
        - Modern AI technologies
        
        ---
        
        **Version:** 1.0.0  
        **Last Updated:** February 2026
        """)


if __name__ == "__main__":
    main()
