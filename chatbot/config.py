"""
Central configuration for the RAG Chatbot.

Responsibilities
----------------
1. Load environment variables
2. Validate configuration
3. Store model configuration
4. Store vector DB configuration
5. Store chunking configuration
5. Store logging configuration
"""

from pathlib import Path
import os

from dotenv import load_dotenv

# ============================================================================
# Base Paths
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

VECTOR_DB_DIR = BASE_DIR / "vector_db"

TABLE_STORE_DIR = BASE_DIR / "table_store"

LOG_DIR = BASE_DIR / "logs"

CSV_FILE = DATA_DIR / "data.csv"

VECTOR_DB_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TABLE_STORE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ============================================================================
# Environment
# ============================================================================

load_dotenv(BASE_DIR / ".env")

# ============================================================================
# LLM Configuration
# ============================================================================

LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "google",
).lower()

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "",
).strip()

LLM_TEMPERATURE = float(
    os.getenv(
        "LLM_TEMPERATURE",
        "0",
    )
)

MAX_OUTPUT_TOKENS = int(
    os.getenv(
        "MAX_OUTPUT_TOKENS",
        "512",
    )
)

# ============================================================================
# API Keys
# ============================================================================

GOOGLE_API_KEY = os.getenv(
    "GOOGLE_API_KEY",
    "",
).strip()

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    "",
).strip()

# ============================================================================
# Embeddings
# ============================================================================

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# ============================================================================
# Chroma
# ============================================================================

CHROMA_COLLECTION_NAME = "csv_rag_collection"

CHROMA_DB_PATH = str(VECTOR_DB_DIR)

# ============================================================================
# Chunking
# ============================================================================

CHUNK_SIZE = 700

CHUNK_OVERLAP = 100

# ============================================================================
# Retrieval
# ============================================================================

TOP_K_RESULTS = 8

SIMILARITY_THRESHOLD = 0.62

MIN_CHUNK_SIMILARITY = 0.58

MAX_CONTEXT_CHUNKS = 5

# Unique values inspected when matching filter text to a column.
MAX_VALUE_MATCH_CANDIDATES = 5000

# ============================================================================
# Agentic RAG limits
# ============================================================================

MAX_CLARIFICATION_TURNS = 2

MAX_RETRIEVAL_RETRIES = 2

MAX_VERIFICATION_RETRIES = 1



# ============================================================================
# Logging
# ============================================================================

JSON_LOG_FILE = LOG_DIR / "rag_runs.jsonl"

ERROR_LOG_FILE = LOG_DIR / "errors.log"

# ============================================================================
# Pricing
# ============================================================================

MODEL_PRICING = {

    "google": {

        "input_per_million": 0.30,

        "output_per_million": 2.50,

    },

    "openai": {

        "input_per_million": 0.00,

        "output_per_million": 0.00,

    },

}

EMBEDDING_COST_PER_MILLION = 0.00

# ============================================================================
# FastAPI
# ============================================================================

API_TITLE = "CSV RAG Chatbot"

API_VERSION = "1.0.0"

# ============================================================================
# Validation
# ============================================================================

def validate_config():

    errors = []

    if not CSV_FILE.exists():

        errors.append(
            f"CSV not found: {CSV_FILE}"
        )

    supported = {

        "google",

        "openai",

    }

    if LLM_PROVIDER not in supported:

        errors.append(
            f"Unsupported provider: {LLM_PROVIDER}"
        )

    if LLM_PROVIDER == "google" and not GOOGLE_API_KEY:

        errors.append(
            "GOOGLE_API_KEY missing."
        )

    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:

        errors.append(
            "OPENAI_API_KEY missing."
        )

    if not LLM_MODEL:

        errors.append(
            "LLM_MODEL missing."
        )

    if errors:

        raise RuntimeError(
            "\n".join(errors)
        )

# ============================================================================
# Debug
# ============================================================================

if __name__ == "__main__":

    validate_config()

    print("Configuration Loaded Successfully\n")

    print(f"Provider      : {LLM_PROVIDER}")

    print(f"Model         : {LLM_MODEL}")

    print(f"Embedding     : {EMBEDDING_MODEL}")

    print(f"Vector DB     : {CHROMA_DB_PATH}")

    print(f"CSV           : {CSV_FILE}")