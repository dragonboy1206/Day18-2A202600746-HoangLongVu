"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM tương thích giao thức OpenAI ---
MIMO_API_KEY = os.getenv("MIMO_API_KEY", "")
OPENAI_API_KEY = MIMO_API_KEY or os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    "https://api.xiaomimimo.com/v1",
).rstrip("/")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "mimo-v2.5-pro")
MIMO_THINKING = os.getenv("MIMO_THINKING", "disabled")
DISABLE_LLM = os.getenv("DISABLE_LLM", "").lower() in {"1", "true", "yes"}
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "60"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
ENRICH_MAX_WORKERS = max(1, int(os.getenv("ENRICH_MAX_WORKERS", "4")))
ENRICH_CACHE_PATH = os.getenv(
    "ENRICH_CACHE_PATH",
    os.path.join(os.path.dirname(__file__), "reports", "enrichment_cache.json"),
)
RAGAS_MAX_WORKERS = max(1, int(os.getenv("RAGAS_MAX_WORKERS", "4")))
RAGAS_EMBEDDING_MODEL = os.getenv(
    "RAGAS_EMBEDDING_MODEL",
    "all-MiniLM-L6-v2",
)

# Một số phiên bản RAGAS/LangChain đọc trực tiếp các biến môi trường này.
if OPENAI_API_KEY:
    os.environ.setdefault("OPENAI_API_KEY", OPENAI_API_KEY)
os.environ.setdefault("OPENAI_BASE_URL", OPENAI_BASE_URL)
os.environ.setdefault("OPENAI_API_BASE", OPENAI_BASE_URL)

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
