"""
Example usage of the AI Document Intelligence System.

Demonstrates key features and common workflows.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.rag_engine import RAGEngine
from src.preprocessing.document_loader import DocumentLoader
from src.utils.logger import setup_logging


def main():
    """Run example usage scenarios."""
    
    # Setup logging
    setup_logging(log_level="INFO", json_format=False)
    
    print("=" * 80)
    print("AI Document Intelligence System - Demo")
    print("=" * 80)
    print()
    
    # Initialize RAG engine
    print("1. Initializing RAG engine...")
    rag_engine = RAGEngine()
    print(f"   ✓ RAG engine initialized")
    print(f"   - Embedding model: {rag_engine.embedding_generator.model_name}")
    print(f"   - Embedding dimension: {rag_engine.embedding_generator.dimension}")
    print()
    
    # Create sample documents
    print("2. Creating sample documents...")
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
                "language": "english"
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
            and predictive analytics.
            """,
            "metadata": {
                "source": "ml_basics.txt",
                "topic": "machine learning",
                "language": "english"
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
            flexibility, and reduced maintenance overhead.
            """,
            "metadata": {
                "source": "cloud_computing.txt",
                "topic": "cloud",
                "language": "english"
            }
        }
    ]
    
    print(f"   ✓ Created {len(sample_docs)} sample documents")
    print()
    
    # Add documents to RAG system
    print("3. Processing and indexing documents...")
    rag_engine.add_documents(sample_docs)
    print(f"   ✓ Documents indexed successfully")
    print(f"   - Total chunks: {rag_engine.document_count}")
    print()
    
    # Query examples
    print("4. Running example queries...")
    print()
    
    queries = [
        "What is Python programming language?",
        "Explain machine learning",
        "What are the benefits of cloud computing?",
        "What types of machine learning exist?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"   Query {i}: {query}")
        print("   " + "-" * 76)
        
        result = rag_engine.query(
            question=query,
            n_results=3,
            include_sources=True
        )
        
        print(f"   Answer: {result['answer'][:200]}...")
        print(f"   Sources used: {result['n_sources']}")
        print()
    
    # Save index
    print("5. Saving vector index...")
    rag_engine.save_index()
    print("   ✓ Index saved successfully")
    print()
    
    # Test loading index
    print("6. Testing index persistence...")
    new_engine = RAGEngine()
    new_engine.load_index()
    print(f"   ✓ Index loaded successfully")
    print(f"   - Documents in loaded index: {new_engine.document_count}")
    print()
    
    print("=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
