
# ============== PineCone ==============================
DEFAULT_INDEX_NAME: str = "notebooklm-index"
EMBEDDING_DIM: int = 384
METRIC: str = "cosine"
CLOUD_PROVIDER: str = "aws"
CLOUD_REGION: str = "us-east-1"
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVER_TOP_K: int = 4 # top 4 most relevant documents to retrieve from vector store
NAMESPACE = None
DATA_INGEST_NUM_OF_WORKERS=8



# ============== LLM ===================================
LLM_MODEL_NAME: str = "openai/gpt-oss-120b"



# ============== RAG ===============================
CHUNK_SIZE: int = 800 # around _ words
CHUNK_OVERLAP: int = 150


# ============== Local Storage =====================
PUBLIC_TEMP_DIR: str = "uploads"
LOGS_DIR: str = "logs"
ARTIFACT_FOLDER: str = "artifacts"

SQLITE_DB_PATH: str = "data/app.db" # no use right now


# =================== USER =======================
MAX_THREADS_PER_USER: int = 2 # no use right now
CONTENT_TTL_MINUTES: int = 60


# ============== LangGraph ==============================

NO_OF_LAST_MESSAGES_TO_KEEP: int = 4

LENGTH_OF_SUMMARY_GENERATED: int = 50 # in words

MINIMUM_LENGTH_OF_LONG_TERM_MEMORY: int = 10

LLM_OUTPUT_MAX_WORDS: int = 100

MAXIMUM_CONNECTION_POOL_SIZE: int = 20

MAX_TOOL_CALL_LIMIT:int = 3




# ==================== Test Paper ======================================
DEFAULT_TOTAL_NO_OF_QUESTIONS:int = 2
DEFAULT_EXAM_TYPE:str = "IIT-JEE"
DEFAULT_SUBJECT_NAME:str = "Maths"
DEFAULT_DEFFICULTY_LEVEL:str = "Medium"
