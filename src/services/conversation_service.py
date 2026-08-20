import asyncio
from src.core.logger import logger
from src.core.constants import DEFAULT_INDEX_NAME
from src.domain.config_entities import RetrieverConfig
from src.retrievers.pinecone_retriever import get_retriever
from src.core.memory import get_checkpointer, get_store
from src.graphs.builder import get_graph
from langfuse import observe
from langchain_core.messages import messages_to_dict


async def delete_thread_data(thread_id: str, delay_seconds: int = 0):
    if delay_seconds > 0:
        logger.info("delete_thread_data: waiting %ds before cleaning thread %s", delay_seconds, thread_id)
        await asyncio.sleep(delay_seconds)

    await delete_pinecone_namespace(thread_id)
    logger.info("delete_thread_data: cleanup complete for thread %s", thread_id)


async def delete_pinecone_namespace(thread_id: str) -> bool:
    try:
        retriever_config = RetrieverConfig(namespace=thread_id)
        retriever = get_retriever(retriever_config=retriever_config)
        return await retriever.delete_namespace(index_name=DEFAULT_INDEX_NAME, namespace=thread_id)
    except Exception as e:
        logger.error("Failed to delete Pinecone namespace %s: %s", thread_id, e)
        return False


@observe(name="load_conversation")
async def load_conversation(thread_id: str, user_id: str):
    try:
        cp = get_checkpointer()
        state = await cp.aget_tuple(config={'configurable': {'thread_id': thread_id, 'user_id': user_id}})

        if state is None or not state.checkpoint:
            logger.info(f"Thread {thread_id} not found or empty.")
            return []

        raw_messages = state.checkpoint.get('channel_values', {}).get('messages', [])

        
        serialized_messages = messages_to_dict(raw_messages)
        return serialized_messages

    except Exception as e:
        logger.error(f"Error loading conversation for thread {thread_id}: {e}")
        return []


@observe(name="delete_user_conversation")
async def delete_user_conversation(thread_id: str, user_id: str):
    try:
        cp = get_checkpointer()
        state = await cp.aget_tuple(config={"configurable": {"thread_id": thread_id,"user_id":user_id}})
        if state is None:
            logger.info("Thread %s not found, nothing to delete.", thread_id)
            return False
        await cp.adelete_thread(thread_id=thread_id)
        logger.info("Thread %s deleted successfully.", thread_id)
        return True
    except Exception as e:
        logger.error("delete_user_conversation failed: %s", str(e))
        return False


@observe(name="get_user_long_term_memory")
async def get_user_long_term_memory(user_id: str):
    try:
        store = get_store()
        memories = await store.asearch(("user", str(user_id), "details"))
        # NOTE: messages_to_dict() does NOT apply here — these are LangGraph
        # BaseStore memory items, not LangChain message objects.
        return [{"key": item.key, "value": item.value} for item in memories]
    except Exception as e:
        logger.error("Error retrieving long term memory: %s", e)
        return []


async def delete_long_term_memory_key(user_id: str, key: str):
    """Delete a single key from the user's long-term memory namespace."""
    try:
        store = get_store()
        await store.adelete(
            namespace=("user", str(user_id), "details"),
            key=key,
        )
        logger.info("Deleted LTM key='%s' for user=%s", key, user_id)
    except Exception as e:
        logger.error("Error deleting LTM key '%s' for user %s: %s", key, user_id, e)
        raise


async def upsert_long_term_memory(user_id: str, key: str, value: str):
    """Add or update a key-value pair in the user's long-term memory."""
    try:
        store = get_store()
        key_name = key.strip().lower().replace(" ", "_")
        await store.aput(
            namespace=("user", str(user_id), "details"),
            key=key_name,
            value={"data": value.strip()},
        )
        logger.info("Upserted LTM key='%s' for user=%s", key_name, user_id)
        return key_name
    except Exception as e:
        logger.error("Error upserting LTM key '%s' for user %s: %s", key, user_id, e)
        raise


async def delete_has_attributes(user_id: str, thread_id: str) -> bool:
    """Deletes has documents boolean attributes"""

    try:
        graph = get_graph()
        config = {
            "configurable": {"thread_id": thread_id, "user_id": user_id},
        }           
        await graph.aupdate_state(config, {"has_documents": False})

        logger.info("toggled has_documents to False for user_id %s and thread_id %s", user_id, thread_id)
        return True

    except Exception as e:
        logger.error("Error while toggling has_documents user_id %s thread_id %s error %s", user_id, thread_id, e)
        return False