import asyncio
from src.core.logger import logger
from src.core.constants import DEFAULT_INDEX_NAME
from src.domain.config_entities import RetrieverConfig
from src.retrievers.pinecone_retriever import get_retriever
from src.core.memory import get_checkpointer, get_store
from src.graphs.builder import get_graph
from langsmith import traceable


async def delete_thread_data(thread_id: str, delay_seconds: int = 0):
    if delay_seconds > 0:
        logger.info("delete_thread_data: waiting %ds before cleaning thread %s", delay_seconds, thread_id)
        await asyncio.sleep(delay_seconds)

    await delete_pinecone_namespace(thread_id)
    logger.info("delete_thread_data: cleanup complete for thread %s", thread_id)


async def delete_pinecone_namespace(thread_id: str):
    try:
        retriever_config = RetrieverConfig(namespace=thread_id)
        retriever = get_retriever(retriever_config=retriever_config)
        await retriever.delete_namespace(index_name=DEFAULT_INDEX_NAME, namespace=thread_id)
    except Exception as e:
        logger.error("Failed to delete Pinecone namespace %s: %s", thread_id, e)


@traceable(name="load_conversation", run_type="chain")
async def load_conversation(thread_id: str, user_id: str):
    try:
        cp = get_checkpointer()
        state = await cp.aget_tuple(config={'configurable': {'thread_id': thread_id, 'user_id': user_id}})

        if state is None or not state.checkpoint:
            logger.info(f"Thread {thread_id} not found or empty.")
            return []

        raw_messages = state.checkpoint.get('channel_values', {}).get('messages', [])
        serialized_messages = []
        for msg in raw_messages:
            if hasattr(msg, "dict"):
                serialized_messages.append(msg.dict())
            elif hasattr(msg, "model_dump"):
                serialized_messages.append(msg.model_dump())
            elif isinstance(msg, dict):
                serialized_messages.append(msg)
            else:
                serialized_messages.append({
                    "content": getattr(msg, "content", str(msg)),
                    "type": getattr(msg, "type", "message")
                })
        return serialized_messages

    except Exception as e:
        logger.error(f"Error loading conversation for thread {thread_id}: {e}")
        return []


@traceable(name="delete_user_conversation", run_type="chain")
async def delete_user_conversation(thread_id: str, user_id: str):
    try:
        graph = get_graph()
        await graph.adelete(config={"configurable": {"thread_id": thread_id, "user_id": user_id}})
        return True
    except Exception as e:
        logger.error(str(e))
        return False


@traceable(name="get_user_long_term_memory", run_type="chain")
async def get_user_long_term_memory(user_id: str):
    try:
        store = get_store()
        memories = store.search(("user", str(user_id), "details"))
        serialized = []
        for item in memories:
            serialized.append({
                "key": getattr(item, "key", ""),
                "value": getattr(item, "value", {})
            })
        return serialized
    except Exception as e:
        logger.error("Error retrieving long term memory: %s", e)
        return []
