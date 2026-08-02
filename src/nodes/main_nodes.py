import sys
from src.core.logger import logger
from src.core.exceptions import MyException
from src.services.data_ingestion_service import DataIngestion
from src.domain.config_entities import DataIngestionConfig, RetrieverConfig
from src.domain.artifacts import DataIngestionArtifact
from src.llm.llm_loader import get_llm
from src.retrievers.pinecone_retriever import get_retriever
from src.core.memory import get_store
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from src.prompts.templates import QUERY_GENERATION_PROMPT, ORCHESTRATOR_PROMPT, CHAT_PROMPT, SUMMARY_NODE_PROMPT
from src.core.constants import NO_OF_LAST_MESSAGES_TO_KEEP, LENGTH_OF_SUMMARY_GENERATED, MINIMUM_LENGTH_OF_LONG_TERM_MEMORY
from src.domain.state import State, QueryGenerationOutput, OrchastratorOutput, ChatOutput
from langsmith import traceable


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
async def orchastrator_node(state: State) -> dict:
    try:
        logger.info("orchastrator_node started")
        llm = get_llm()
        structured_llm = llm.with_structured_output(OrchastratorOutput)

        
        prompt_input = ORCHESTRATOR_PROMPT.invoke({"messages": state.messages})
        result = await structured_llm.ainvoke(prompt_input)
        
        logger.info("Orchestrator decision: require_db_search=%s", result.require_db_search)
        return {"require_db_search": result.require_db_search}
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
        results = []
        for query in state.queries:
            docs = await retriever.get_similar_documents(vector_store=vector_store, query=query)
            results.extend(docs)
        logger.info("retreiver_node returned %d documents", len(results))
        return {"retreived_results": results}
    except Exception as e:
        logger.error("retreiver_node failed: %s", str(e))
        raise MyException(e, sys)


@traceable(name="chat_node", run_type="chain")
async def chat_node(state: State, config: RunnableConfig, store: BaseStore):
    try:
        user_id = config.get("configurable", {}).get("user_id", "unknown")
        logger.info("chat_node started for user=%s, messages_count=%d", user_id, len(state.messages))

        # Long term memory retrieve
        lsm = store.search(("user", str(user_id), "details"))
        user_memories = "\n".join([f"- {item.key}: {item.value.get('data')}" for item in lsm if item.value]) if lsm else "None"

        # Context & summary preparation
        context = "\n\n".join([doc.page_content for doc in state.retreived_results]) if state.retreived_results else "None"
       
        # Invoke Prompt Template directly
        prompt_input = CHAT_PROMPT.invoke({
            "user_memories": user_memories,
            "context": context,
            "messages": state.messages,
        })

        llm = get_llm()
        structured_llm = llm.with_structured_output(ChatOutput)
        chat_res: ChatOutput = await structured_llm.ainvoke(prompt_input)

        # Long-term Memory Storage Logic
        if chat_res.memory_key and chat_res.memory_value and chat_res.memory_key.lower() not in ["null", "none"]:
            key_name = chat_res.memory_key.strip().lower().replace(" ", "_")
            logger.info("Storing long-term memory for user %s: %s = %s", user_id, key_name, chat_res.memory_value)
            store.put(
                namespace=("user", str(user_id), "details"),
                key=key_name,
                value={"data": chat_res.memory_value.strip()}
            )

        response = AIMessage(content=chat_res.response)
        return {"messages": [response], "ai_response": chat_res.response}
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
