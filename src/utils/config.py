"""
Configuration management for the AI Document Intelligence System.

This module handles environment-based configuration, supporting both
development and production environments with proper validation.
"""

import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 32
    device: str = "cpu"  # or "cuda" for GPU


@dataclass
class LLMConfig:
    """Configuration for LLM integration."""
    api_key: Optional[str] = None
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: int = 30


@dataclass
class VectorStoreConfig:
    """Configuration for vector database."""
    index_path: str = "data/vector_db/index"
    similarity_metric: str = "cosine"  # cosine, l2, ip
    n_results: int = 5


@dataclass
class APIConfig:
    """Configuration for API server."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: list = None
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]


@dataclass
class AWSConfig:
    """Configuration for AWS services."""
    region: str = "us-east-1"
    s3_bucket: Optional[str] = None
    dynamodb_table: Optional[str] = None
    use_aws: bool = False


class Config:
    """
    Central configuration management.
    
    Loads configuration from environment variables with sensible defaults.
    Validates required settings based on deployment environment.
    """
    
    def __init__(self):
        self.environment = os.getenv("ENVIRONMENT", "development")
        
        # Initialize configurations
        self.embedding = EmbeddingConfig(
            model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            dimension=int(os.getenv("EMBEDDING_DIMENSION", "384")),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
            device=os.getenv("DEVICE", "cpu")
        )
        
        self.llm = LLMConfig(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1000")),
            timeout=int(os.getenv("LLM_TIMEOUT", "30"))
        )
        
        self.vector_store = VectorStoreConfig(
            index_path=os.getenv("VECTOR_DB_PATH", "data/vector_db/index"),
            similarity_metric=os.getenv("SIMILARITY_METRIC", "cosine"),
            n_results=int(os.getenv("N_RETRIEVAL_RESULTS", "5"))
        )
        
        self.api = APIConfig(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            debug=os.getenv("DEBUG", "False").lower() == "true",
            cors_origins=os.getenv("CORS_ORIGINS", "*").split(",")
        )
        
        self.aws = AWSConfig(
            region=os.getenv("AWS_REGION", "us-east-1"),
            s3_bucket=os.getenv("S3_BUCKET"),
            dynamodb_table=os.getenv("DYNAMODB_TABLE"),
            use_aws=os.getenv("USE_AWS", "False").lower() == "true"
        )
        
        # Validate configuration
        self._validate()
    
    def _validate(self):
        """Validate critical configuration settings."""
        if self.environment == "production":
            if not self.llm.api_key:
                raise ValueError("OPENAI_API_KEY is required in production")
            
            if self.aws.use_aws and not self.aws.s3_bucket:
                raise ValueError("S3_BUCKET is required when USE_AWS is enabled")
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"


# Global configuration instance
config = Config()
