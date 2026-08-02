import sys
from src.core.logger import logger
from src.core.exceptions import MyException
from src.domain.state import State
from src.llm.llm_loader import get_llm
from src.retrievers.pinecone_retriever import get_retriever
from src.core.constants import NO_OF_LAST_MESSAGES_TO_KEEP, DEFAULT_INDEX_NAME
from src.prompts.templates import SUMMARIZER_PROMPT
from langchain_core.messages import RemoveMessage, HumanMessage,SystemMessage
from src.domain.config_entities import RetrieverConfig
from langchain_core.runnables import RunnableConfig
from src.core.constants import LENGTH_OF_SUMMARY_GENERATED as NO_OF_WORDS_TO_SUMMARIZE
from langsmith import traceable


@traceable(name="summerizer_node", run_type="chain")
async def summerizer(state: State, config: RunnableConfig):
    try:
        thread_id = config.get("configurable", {}).get("thread_id", "unknown")
        logger.info("summerizer node started for thread=%s", thread_id)
        
        messages = state.messages
        
        # Check if we have enough messages to trim/summarize
        if len(messages) <= NO_OF_LAST_MESSAGES_TO_KEEP:
            logger.info("Not enough messages to summarize — skipping")
            return {}

        messages_to_summarize = messages[:-NO_OF_LAST_MESSAGES_TO_KEEP]

        if not messages_to_summarize:
            logger.info("No old messages to delete/summarize — skipping")
            return {}

        # Prepare prompt input
        summary_messages = SUMMARIZER_PROMPT.invoke({
            "messages": messages_to_summarize,
            "no_of_words": NO_OF_WORDS_TO_SUMMARIZE
        })

        llm = get_llm()
        response = await llm.ainvoke(summary_messages)

        # Delete old messages from memory graph state
        delete_messages = [RemoveMessage(id=m.id) for m in messages_to_summarize if m.id]
        
        logger.info("Summarization completed, deleting %d old messages", len(delete_messages))
        state.messages = SystemMessage(content=f"user summerized messages:{response.content}") + messages[-NO_OF_LAST_MESSAGES_TO_KEEP:]  # Keep only the last few messages
        return {
            "messages": [delete_messages],
        }
        
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
