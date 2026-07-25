import sys
import asyncio
from src.logger import logger
from src.exception import MyException
from src.models.workflow_models import State
from src.core.dependencies import get_llm, get_thread_manager, get_retriever
from src.constants import NO_OF_LAST_MESSAGES_TO_KEEP
from src.prompt import SUMMARIZER_PROMPT, SUMMARIZER_EXTEND_PROMPT
from langchain_core.messages import RemoveMessage, HumanMessage
from src.constants import DEFAULT_INDEX_NAME
from src.entity.config import RetrieverConfig

async def summerizer(state: State):
    try:
        logger.info("summerizer node started for thread=%s", state.thread_id)
        llm = get_llm()
        summary = state.summary or ""
        messages = state.messages
        messages_to_summarize = messages[:-NO_OF_LAST_MESSAGES_TO_KEEP]

        if not messages_to_summarize:
            logger.info("Not enough messages to summarize — skipping")
            return {}

        prompt_text = (
            SUMMARIZER_EXTEND_PROMPT.format(summary=summary)
            if summary else SUMMARIZER_PROMPT
        )

        summary_messages = messages_to_summarize + [HumanMessage(content=prompt_text)]
        response = await llm.ainvoke(summary_messages)
        delete_messages = [RemoveMessage(id=m.id) for m in messages_to_summarize]
        logger.info("Summarization completed, deleting %d old messages", len(delete_messages))
        return {"summary": response.content, "messages": delete_messages}
    except Exception as e:
        logger.error("summerizer node failed: %s", str(e))
        raise MyException(e, sys)


async def thread_manager_node(state: State) -> dict:
    try:
        logger.info("thread_manager_node started for user=%s thread=%s", state.user_id, state.thread_id)
        tm = get_thread_manager()

        if tm.thread_exists(thread_id=state.thread_id):
            logger.info("thread=%s already registered — skipping registration", state.thread_id)
            return {}

        evicted_thread_id = tm.register_thread(user_id=state.user_id, thread_id=state.thread_id)
        logger.info("thread=%s registered for user=%s", state.thread_id, state.user_id)

        if evicted_thread_id:
            logger.info("Evicted oldest thread=%s — scheduling async cleanup", evicted_thread_id)
            asyncio.create_task(_cleanup_evicted_thread(evicted_thread_id))

        return {}
    except Exception as e:
        logger.error("thread_manager_node failed: %s", str(e))
        raise MyException(e, sys)


async def _cleanup_evicted_thread(thread_id: str):
    try:
        logger.info("Cleaning up evicted thread=%s", thread_id)
        retriever_config = RetrieverConfig(namespace=thread_id)
        retriever = get_retriever(retriever_config=retriever_config)
        await retriever.delete_namespace(index_name=DEFAULT_INDEX_NAME, namespace=thread_id)
        logger.info("Evicted thread=%s cleanup complete", thread_id)
    except Exception as e:
        logger.error("Cleanup failed for evicted thread=%s: %s", thread_id, str(e))

