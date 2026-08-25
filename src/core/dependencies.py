import os
from src.core.logger import logger
from src.core.config import get_app_config
from src.llm.llm_loader import get_llm
from src.embeddings.embedding_loader import get_embeddings
from src.retrievers.pinecone_client import get_pinecone_client
from src.core.memory import get_checkpointer, get_store
from src.db.thread_manager import get_thread_manager
from src.core.memory import init_db_services,close_db_services
from src.services.data_ingestion_service import get_shared_docling_converter


# initiaing monitoring
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


def setup_langfuse():
    cfg = get_app_config()
    if cfg.langfuse_public_key:
        os.environ["LANGFUSE_PUBLIC_KEY"] = cfg.langfuse_public_key
    if cfg.langfuse_secret_key:
        os.environ["LANGFUSE_SECRET_KEY"] = cfg.langfuse_secret_key
    if cfg.langfuse_host:
        os.environ["LANGFUSE_HOST"] = cfg.langfuse_host
    if cfg.langfuse_base_url:
        os.environ["LANGFUSE_BASE_URL"] = cfg.langfuse_base_url
    logger.info("Langfuse environment monitoring configured.")


# Automatically configure environment variables from AppConfig on module load
setup_langsmith()
setup_langfuse()


# heavy dependencies are preloding here before server start
def warmup_dependencies():
    logger.info("Warming up all dependencies singletons...")
    _ = get_app_config()
    _ = setup_langsmith()
    _ = setup_langfuse()
    _ = get_llm()
    _ = get_embeddings()
    _ = get_pinecone_client()
    _ = get_checkpointer()
    _ = get_store()
    _ = get_thread_manager()
    _ = get_shared_docling_converter()  # Pre-initialize Docling Converter singleton into RAM



    logger.info("All dependencies singletons warmed up successfully.")



# connection pool loading and closing for limited connection in neon db ounce the server starts and ends
async def connection_pool():
    logger.info("Stablishing connection with db")
    await init_db_services()


async def close_connection_pool():
    logger.info("Closing connection with db")
    await close_db_services()

