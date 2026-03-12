
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class RAGConfig:
    """Configuration settings for RAG pipeline."""
    # Retrieval Settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    # Hybrid Search Weights
    semantic_weight: float = 0.5
    keyword_weight: float = 0.5
    rrf_k: int = 60
    # Reranker Settings
    enable_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 15  # Increased from 10 to give LLM more rescue options
    reranker_device: str = "cpu"
    # LLM Reranker (replaces cross-encoder when use_llm_reranker=True)
    use_llm_reranker: bool = True  # Use LLM reranker instead of cross-encoder
    llm_reranker_model: str = "google/gemini-2.0-flash-001"  # Fast, intelligent model
    # Section Title Repetition for BM25 boost
    title_repetition_count: int = 3  # Number of times to repeat section title in chunk
    title_repetition_enabled: bool = True  # Enable/disable feature
    # Vector DB
    collection_name: str = "malaysian_legal_acts_v2"  # v2 uses BGE 768-dim embeddings
    # Model Settings (defaults)
    embedding_model: str = "BAAI/bge-base-en-v1.5"  # Upgraded: top MTEB retrieval, 768-dim
    llm_provider: str = "openrouter"  # "openrouter" or "gemini"
    llm_model: str = "openrouter/free"  # Auto-routes to best available free model
    temperature: float = 0.1
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

# Add get_act_category function
def get_act_category(act_number: int) -> str:
    """
    Map act numbers to legal categories.
    """
    category_map = {
        # Contracts and Commercial Law
        136: "commercial",
        137: "commercial",
        135: "commercial",
        383: "commercial",

        # Property Law
        118: "property",
        318: "property",

        # Civil Procedure
        91: "civil_procedure",

        # Strata Titles (property related)
        318: "property",

        # Courts of Judicature Act
        2019: "civil_procedure",  # Use same category as Civil Procedure
    }
    return category_map.get(act_number, "other")


def setup_logging(name: str) -> logging.Logger:
    """
    Set up logging with standard configuration.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# Configure logging
logger = setup_logging(__name__)


# Add helper functions for getting paths
def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent


def get_data_dir() -> Path:
    """Get the data directory."""
    return get_project_root() / "data"


def get_processed_dir() -> Path:
    """Get the processed data directory."""
    return get_data_dir() / "processed"


def get_vector_db_dir() -> Path:
    """Get the vector database directory."""
    return get_data_dir() / "vector_db"
