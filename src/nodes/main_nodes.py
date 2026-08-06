import sys
import asyncio
from src.core.logger import logger
from src.core.exceptions import MyException
from src.services.data_ingestion_service import DataIngestion
from src.domain.config_entities import DataIngestionConfig, RetrieverConfig
from src.domain.artifacts import DataIngestionArtifact
from src.llm.llm_loader import get_llm
from src.retrievers.pinecone_retriever import get_retriever
from src.retrievers.pinecone_client import get_pinecone_client
from src.core.memory import get_store
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from src.prompts.templates import QUERY_GENERATION_PROMPT, ORCHESTRATOR_PROMPT, CHAT_PROMPT, SUMMARY_NODE_PROMPT
from src.core.constants import NO_OF_LAST_MESSAGES_TO_KEEP, LENGTH_OF_SUMMARY_GENERATED, MINIMUM_LENGTH_OF_LONG_TERM_MEMORY, DEFAULT_INDEX_NAME, LLM_OUTPUT_MAX_WORDS
from src.domain.state import State, QueryGenerationOutput, OrchastratorOutput, ChatOutput
from langsmith import traceable
from typing import Optional,List,Optional
from langchain_core.messages import HumanMessage
from src.tools.solver_tool import solver
import re

@traceable(name="ingestion_node", run_type="chain")
async def ingestion_node(state: State, config: RunnableConfig):
    try:
        thread_id = config["configurable"]["thread_id"]
        logger.info("ingestion_node started for thread=%s, files=%d", thread_id, len(state.file_paths))
        data_ingestion_config = DataIngestionConfig(
            files_path=state.file_paths,
            namespace=thread_id,
        )
        retriever_config = RetrieverConfig(namespace=thread_id)
        retriever = get_retriever(retriever_config=retriever_config)
        data_ingestion = DataIngestion(
            data_ingestion_config=data_ingestion_config,
            retriever=retriever,
        )
        await data_ingestion.ingest()
        logger.info("ingestion_node completed for thread=%s", thread_id)
        return {}
    except Exception as e:
        logger.error("ingestion_node failed: %s", str(e))
        raise MyException(e, sys)


@traceable(name="orchastrator_node", run_type="chain")
async def orchastrator_node(state: State, config: RunnableConfig) -> dict:
    try:
        logger.info("orchastrator_node started")
        thread_id = config.get("configurable", {}).get("thread_id", "unknown")

        # --- Check if this thread's namespace has any vectors in Pinecone ---
        has_documents = False
        try:
            pc = get_pinecone_client()
            index = pc.Index(DEFAULT_INDEX_NAME)
            stats = index.describe_index_stats()
            ns_stats = stats.get("namespaces", {}).get(str(thread_id), {})
            vector_count = ns_stats.get("vector_count", 0)
            has_documents = vector_count > 0
            logger.info("Pinecone namespace=%s has %d vectors (has_documents=%s)", thread_id, vector_count, has_documents)
        except Exception as pc_err:
            logger.warning("Could not query Pinecone stats, defaulting has_documents=False: %s", pc_err)

        llm = get_llm()
        structured_llm = llm.with_structured_output(OrchastratorOutput)
        prompt_input = ORCHESTRATOR_PROMPT.invoke({"messages": state.messages,"has_documents": has_documents})
        result = await structured_llm.ainvoke(prompt_input)

        logger.info("Orchestrator LLM decision: require_db_search=%s", result.require_db_search)
        
        # --- Safe Move ----
        if result.require_db_search and not has_documents:
            logger.warning("Orchestrator LLM requested DB search but no documents found — overriding to require_db_search=False")
            result.require_db_search = False
        return {"require_db_search": result.require_db_search, "has_documents": has_documents}
    except Exception as e:
        logger.error("orchastrator_node failed: %s", str(e))
        raise MyException(e, sys)


@traceable(name="query_generation_node", run_type="chain")
async def query_generation_node(state: State) -> dict:
    try:
        logger.info("query_generation_node started")
        llm = get_llm()
        structured_llm = llm.with_structured_output(QueryGenerationOutput)
        
        prompt_input = QUERY_GENERATION_PROMPT.invoke({"messages": state.messages})
        result = await structured_llm.ainvoke(prompt_input)
        
        logger.info("Generated %d queries", len(result.queries))
        return {"queries": result.queries}
    except Exception as e:
        logger.error("query_generation_node failed: %s", str(e))
        raise MyException(e, sys)


@traceable(name="retreiver_node", run_type="chain")
async def retreiver_node(state: State, config: RunnableConfig):
    try:
        thread_id = config["configurable"]["thread_id"]
        logger.info("retreiver_node started, queries=%d", len(state.queries))
        retriever_config = RetrieverConfig(namespace=thread_id)
        retriever = get_retriever(retriever_config=retriever_config)
        vector_store = await retriever.create_retriever()

        filters: Optional[List[str]] = None
        # Match @filename.ext patterns the user typed in chat, e.g. @report.pdf
        pattern = r'@(\w+\.(?:pdf|txt|docx|doc))'
        if isinstance(state.messages[-1], HumanMessage):
            raw_mentions = re.findall(pattern, state.messages[-1].content)
            if raw_mentions:
                # Filenames are stored in Pinecone as "{thread_id}_{original_filename}"
                # during ingestion (see DataIngestion._inject_filename_metadata).
                # We must reconstruct the same tagged name to match.
                filters = [f"{thread_id}_{name}" for name in raw_mentions]
                logger.info(
                    "retreiver_node: @mention filter applied — %s", filters
                )

        results = []

        # Run all Pinecone similarity searches in parallel — no reason to wait
        # for query N to finish before firing query N+1. asyncio.gather() sends
        # all requests simultaneously and collects results when all are done.
        search_tasks = [
            retriever.get_similar_documents(
                vector_store=vector_store, query=query, filter=filters
            )
            for query in state.queries
        ]
        gathered = await asyncio.gather(*search_tasks)
        for docs in gathered:
            results.extend(docs)
        logger.info("retreiver_node returned %d documents", len(results))
        return {"retreived_results": results}
    except Exception as e:
        logger.error("retreiver_node failed: %s", str(e))
        raise MyException(e, sys)


@traceable(name="chat_node", run_type="chain")
async def chat_node(state: State, config: RunnableConfig, store: BaseStore):
    try:
          # local import to avoid circulars

        user_id = config.get("configurable", {}).get("user_id", "unknown")
        logger.info("chat_node started for user=%s, messages_count=%d", user_id, len(state.messages))

        # Long term memory retrieve
        lsm = await store.asearch(("user", str(user_id), "details"))
        user_memories = "\n".join([f"- {item.key}: {item.value.get('data')}" for item in lsm if item.value]) if lsm else "None"

        # Context & summary preparation
        context = "\n\n".join([doc.page_content for doc in state.retreived_results]) if state.retreived_results else "None"

        # Invoke Prompt Template directly
        prompt_input = CHAT_PROMPT.invoke({
            "user_memories": user_memories,
            "context": context,
            "messages": state.messages,
            "max_words": LLM_OUTPUT_MAX_WORDS
        })

        # Bind tools, then layer structured-output with include_raw=True so we
        # keep access to .tool_calls on the raw AIMessage. branches:
        #   - result["parsed"] populated  -> final answer (handle memory, return ai_response)
        #   - result["parsed"] is None   -> tool was called; let tools_condition route to tool_node
        llm = get_llm().bind_tools([solver])
        unified_model = llm.with_structured_output(
            ChatOutput,
            method="json_schema",
            include_raw=True,
            strict=True,
        )
        result = await unified_model.ainvoke(prompt_input)
        raw_msg: AIMessage = result["raw"]
        parsed_output: Optional[ChatOutput] = result.get("parsed")

        if parsed_output is not None:
            # Final structured turn
            if parsed_output.memory_key and parsed_output.memory_value and parsed_output.memory_key.lower() not in ["null", "none"]:
                key_name = parsed_output.memory_key.strip().lower().replace(" ", "_")
                logger.info("Storing long-term memory for user %s: %s = %s", user_id, key_name, parsed_output.memory_value)
                await store.aput(
                    namespace=("user", str(user_id), "details"),
                    key=key_name,
                    value={"data": parsed_output.memory_value.strip()},
                )

            return {"messages": [raw_msg], "ai_response": parsed_output.response}

        # Tool-calling turn: raw_msg carries .tool_calls. tools_condition in
        # builder.py will route to tool_node; tool_node results come back here.
        logger.info("chat_node routed to tool_node (tool_calls=%d)", len(getattr(raw_msg, "tool_calls", []) or []))
        return {"messages": [raw_msg]}
    except Exception as e:
        logger.error("chat_node failed: %s", str(e))
        raise MyException(e, sys)

# async def summary_node(state: State):
#     try:
#         logger.info("=" * 50)
#         logger.info("Summary node started.")

#         if len(state.messages) <= NO_OF_LAST_MESSAGES_TO_KEEP:
#             logger.info(
#                 "Skipping summary. Messages (%d) <= threshold (%d).",
#                 len(state.messages),
#                 NO_OF_LAST_MESSAGES_TO_KEEP,
#             )
#             return {}

#         llm = get_llm()
#         logger.info("LLM initialized successfully.")

#         old_messages = state.messages[:-NO_OF_LAST_MESSAGES_TO_KEEP]
#         logger.info(
#             "Messages selected for summarization: %d",
#             len(old_messages),
#         )

#         summary_chain = SUMMARY_NODE_PROMPT | llm

#         logger.info("Generating conversation summary...")

#         response = summary_chain.invoke(
#             {
#                 "no_of_words": LENGTH_OF_SUMMARY_GENERATED,
#                 "messages": old_messages,
#             }
#         )

#         summary_text = response.content

#         logger.info("Summary generated successfully.")
#         logger.debug("Summary: %s", summary_text)

#         logger.info("=" * 50)

#         return {"summarized_conv": summary_text}

#     except Exception as e:
#         logger.exception("Error occurred in summary node.")
#         raise MyException(e, sys)
