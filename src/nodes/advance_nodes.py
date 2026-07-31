import sys
from src.core.logger import logger
from src.core.exceptions import MyException
from src.domain.state import State
from src.llm.llm_loader import get_llm
from src.retrievers.pinecone_retriever import get_retriever
from src.core.constants import NO_OF_LAST_MESSAGES_TO_KEEP, DEFAULT_INDEX_NAME
from src.prompts.templates import SUMMARIZER_PROMPT, SUMMARIZER_EXTEND_PROMPT
from langchain_core.messages import RemoveMessage, HumanMessage
from src.domain.config_entities import RetrieverConfig
from langchain_core.runnables import RunnableConfig


async def summerizer(state: State, config: RunnableConfig):
    try:
        thread_id = config["configurable"]["thread_id"]
        logger.info("summerizer node started for thread=%s", thread_id)
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


async def thread_manager_node(state: State, config: RunnableConfig) -> dict:
    try:
        pass
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
