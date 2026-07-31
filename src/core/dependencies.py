from src.core.logger import logger
from src.core.config import get_app_config
from src.llm.llm_loader import get_llm
from src.embeddings.embedding_loader import get_embeddings
from src.retrievers.pinecone_client import get_pinecone_client
from src.core.memory import get_checkpointer, get_store
from src.db.thread_manager import get_thread_manager


def warmup_dependencies():
    logger.info("Warming up all dependencies singletons...")
    _ = get_app_config()
    _ = get_llm()
    _ = get_embeddings()
    _ = get_pinecone_client()
    _ = get_checkpointer()
    _ = get_store()
    _ = get_thread_manager()
    logger.info("All dependencies singletons warmed up successfully.")
