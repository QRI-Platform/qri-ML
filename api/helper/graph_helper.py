import asyncio
from src.logger import logger
from src.constants import DEFAULT_INDEX_NAME
from src.entity.config import RetrieverConfig
from src.retreiver.retreiver import get_retriever
from src.memory import get_checkpointer

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


async def load_conversation(thread_id: str):
    try:
        cp = get_checkpointer()
        state = await cp.aget_tuple(config={'configurable': {'thread_id': thread_id}})
        
        if state is None or not state.checkpoint:
            logger.info(f"Thread {thread_id} not found or empty.")
            return []
            
        # Checkpoint ke 'channel_values' mein saari state values (jaise 'messages') hoti hain
        messages = state.checkpoint.get('channel_values', {}).get('messages', [])
        return messages

    except Exception as e:
        logger.error(f"Error loading conversation for thread {thread_id}: {e}")
        return []