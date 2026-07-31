DEFAULT_INDEX_NAME: str = "notebooklm-index"
EMBEDDING_DIM: int = 384
METRIC: str = "cosine"
CLOUD_PROVIDER: str = "aws"
CLOUD_REGION: str = "us-east-1"
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVER_TOP_K: int = 4
NAMESPACE = None

LLM_MODEL_NAME: str = "llama-3.1-8b-instant"

CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200

PUBLIC_TEMP_DIR: str = "uploads"
LOGS_DIR: str = "logs"
ARTIFACT_FOLDER: str = "artifacts"

SQLITE_DB_PATH: str = "data/app.db"

MAX_THREADS_PER_USER: int = 2
CONTENT_TTL_MINUTES: int = 60

NO_OF_LAST_MESSAGES_TO_KEEP: int = 6

LENGTH_OF_SUMMARY_GENERATED: int = 100

MINIMUM_LENGTH_OF_LONG_TERM_MEMORY: int = 5
