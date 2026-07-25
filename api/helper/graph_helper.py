import asyncio
from src.logger import logger
from src.constants import DEFAULT_INDEX_NAME
from src.entity.config import RetrieverConfig
from src.core.dependencies import get_retriever


async def delete_thread_data(thread_id: str, delay_seconds: int = 0):

    if delay_seconds > 0:
        logger.info("delete_thread_data: waiting %ds before cleaning thread %s", delay_seconds, thread_id)
        await asyncio.sleep(delay_seconds)

    await _delete_pinecone_namespace(thread_id)
    logger.info("delete_thread_data: cleanup complete for thread %s", thread_id)


async def _delete_pinecone_namespace(thread_id: str):
    try:
        retreiver_config = RetrieverConfig(namespace=thread_id)
        retreiver = get_retriever(retriever_config=retreiver_config)
        await retreiver.delete_namespace(index_name=DEFAULT_INDEX_NAME, namespace=thread_id)
    except Exception as e:
        logger.error("Failed to delete Pinecone namespace %s: %s", thread_id, e)