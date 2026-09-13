
from typing import Annotated,List,Optional
from langchain_core.tools import tool, InjectedToolArg
from langchain_core.runnables import RunnableConfig
from src.core.logger import logger
from src.domain.config_entities import RetrieverConfig
from src.retrievers.pinecone_retriever import get_retriever
import asyncio
from src.core.exceptions import MyException
import sys
@tool
async def retreiver(
    queries: List[str],
    config: Annotated[RunnableConfig, InjectedToolArg],
    filters: Optional[List[str]] = None,
):
    """Search uploaded documents using one or more focused semantic queries.

    Use this tool when the user asks about information that may be present in an uploaded document.
    Create concise search queries from the user's request. If the conversation contains an upload
    message with a filename and the user refers to that file, pass its exact filename in filters.
    Pass the exact filename, for example filters=["report.pdf","image.png"]. Leave filters as None when the
    search should cover all uploaded documents.

    Args:
        queries: Focused natural-language questions or search queries for the uploaded documents.
        filters: Optional list of exact uploaded filenames to restrict the search to.
    """
    try:
        thread_id = config["configurable"]["thread_id"]
        logger.info("retreiver started, queries=%d", len(queries))
        retriever_config = RetrieverConfig(namespace=thread_id)
        retriever = get_retriever(retriever_config=retriever_config)
        vector_store = await retriever.create_retriever()

        if filters:
            filters = [f"{thread_id}_{filename}" for filename in filters]


        results = []

        # Run all Pinecone similarity searches in parallel — no reason to wait
        # for query N to finish before firing query N+1. asyncio.gather() sends
        # all requests simultaneously and collects results when all are done.
        search_tasks = [
            retriever.get_similar_documents(
                vector_store=vector_store, query=query, filter=filters
            )
            for query in queries
        ]
        gathered = await asyncio.gather(*search_tasks)
        for docs in gathered:
            results.extend(docs)
        logger.info("retreiver returned %d documents", len(results))
        return {"retreived_results": results}
    except Exception as e:
        logger.error("retreiver failed: %s", str(e))
        raise MyException(e, sys)