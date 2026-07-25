import sys
from src.logger import logger
from src.config.app_config import AppConfig, get_app_config
from src.llm.llm_loader import get_llm
from src.embeddings.embedding_loader import get_embeddings
from src.retreiver.pinecone_client import get_pinecone_client
from src.memory import get_checkpointer
from src.entity.config import RetrieverConfig
from src.retreiver.retreiver import Retriever

from db.thread_manager import ThreadManager, get_thread_manager

__all__ = [
    "AppConfig",
    "get_app_config",
    "get_llm",
    "get_embeddings",
    "get_pinecone_client",
    "get_checkpointer",
    "ThreadManager",
    "get_thread_manager",
    "RetrieverConfig",
    "Retriever",
    "get_retriever",
    "warmup_dependencies",
]


def get_retriever(retriever_config: RetrieverConfig) -> Retriever:
    return Retriever(retriever_config=retriever_config)


def warmup_dependencies():
    logger.info("Warming up all dependencies singletons...")
    _ = get_app_config()
    _ = get_llm()
    _ = get_embeddings()
    _ = get_pinecone_client()
    _ = get_checkpointer()
    _ = get_thread_manager()
    logger.info("All dependencies singletons warmed up successfully.")

