import os
from src.core.logger import logger
from src.core.config import get_app_config
from src.llm.llm_loader import get_llm
from src.embeddings.embedding_loader import get_embeddings
from src.retrievers.pinecone_client import get_pinecone_client
from src.core.memory import get_checkpointer, get_store
from src.db.thread_manager import get_thread_manager
from src.core.memory import init_db_services,close_db_services

def setup_langsmith():
    cfg = get_app_config()
    if cfg.langsmith_api_key:
        tracing_str = "true" if cfg.langsmith_tracing else "false"
        os.environ["LANGCHAIN_TRACING_V2"] = tracing_str
        os.environ["LANGSMITH_TRACING"] = tracing_str
        
        endpoint = cfg.langsmith_endpoint or "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint
        os.environ["LANGSMITH_ENDPOINT"] = endpoint
        
        os.environ["LANGCHAIN_API_KEY"] = cfg.langsmith_api_key
        os.environ["LANGSMITH_API_KEY"] = cfg.langsmith_api_key
        
        if cfg.langsmith_project:
            os.environ["LANGCHAIN_PROJECT"] = cfg.langsmith_project
            os.environ["LANGSMITH_PROJECT"] = cfg.langsmith_project
        logger.info("LangSmith tracing enabled for project: %s", cfg.langsmith_project or "default")


def warmup_dependencies():
    logger.info("Warming up all dependencies singletons...")
    _ = get_app_config()
    _ = setup_langsmith()
    _ = get_llm()
    _ = get_embeddings()
    _ = get_pinecone_client()
    _ = get_checkpointer()
    _ = get_store()
    _ = get_thread_manager()
    logger.info("All dependencies singletons warmed up successfully.")


async def connection_pool():
    logger.info("Stablishing connection with db")
    await init_db_services()


async def close_connection_pool():
    logger.info("Closing connection with db")
    await close_db_services()

