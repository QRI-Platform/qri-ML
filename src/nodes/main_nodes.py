import sys
from src.logger import logger
from src.exception import MyException
from src.models.workflow_models import State, QueryGenerationOutput, OrchastratorOutput
from src.components.data_ingestion import DataIngestion
from src.entity.config import DataIngestionConfig, RetrieverConfig
from src.entity.artifact import DataIngestionArtifact
from src.llm.llm_loader import get_llm
from src.retreiver.retreiver import get_retriever
from langchain_core.messages import SystemMessage
from src.prompt import QUERY_GENERATION_PROMPT, ORCHESTRATOR_PROMPT, CHAT_PROMPT, SUMMARY_NODE_PROMPT
from src.constants import NO_OF_LAST_MESSAGES_TO_KEEP, LENGTH_OF_SUMMARY_GENERATED

async def ingestion_node(state: State):
    try:
        logger.info("ingestion_node started for thread=%s, files=%d", state.thread_id, len(state.file_paths))
        data_ingestion_config = DataIngestionConfig(
            files_path=state.file_paths,
            namespace=state.thread_id,
        )
        retriever_config = RetrieverConfig(namespace=state.thread_id)
        retriever = get_retriever(retriever_config=retriever_config)
        data_ingestion = DataIngestion(
            data_ingestion_config=data_ingestion_config,
            retriever=retriever,
        )
        await data_ingestion.ingest()
        logger.info("ingestion_node completed for thread=%s", state.thread_id)
        return {}
    except Exception as e:
        logger.error("ingestion_node failed: %s", str(e))
        raise MyException(e, sys)


async def orchastrator_node(state: State) -> dict:
    try:
        logger.info("orchastrator_node started")
        llm = get_llm()
        structured_llm = llm.with_structured_output(OrchastratorOutput, method="json_mode")
        messages = [SystemMessage(content=ORCHESTRATOR_PROMPT), *state.messages]
        result = structured_llm.invoke(messages)
        logger.info("Orchestrator decision: require_db_search=%s", result.require_db_search)
        return {"require_db_search": result.require_db_search}
    except Exception as e:
        logger.error("orchastrator_node failed: %s", str(e))
        raise MyException(e, sys)


async def query_generation_node(state: State) -> dict:
    try:
        logger.info("query_generation_node started")
        llm = get_llm()
        structured_llm = llm.with_structured_output(QueryGenerationOutput, method="json_mode")
        messages = [SystemMessage(content=QUERY_GENERATION_PROMPT), *state.messages]
        result = structured_llm.invoke(messages)
        logger.info("Generated %d queries", len(result.queries))
        return {"queries": result.queries}
    except Exception as e:
        logger.error("query_generation_node failed: %s", str(e))
        raise MyException(e, sys)


async def retreiver_node(state: State):
    try:
        logger.info("retreiver_node started, queries=%d", len(state.queries))
        retriever_config = RetrieverConfig(namespace=state.thread_id)
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


async def chat_node(state: State):
    try:
        logger.info("chat_node started, messages_count=%d", len(state.messages))
        final_messages = state.messages
        summarized = getattr(state, "summarized_conv", None) or state.summary
        if summarized:
            final_messages = (
                [SystemMessage(content=summarized)]
                + state.messages[-NO_OF_LAST_MESSAGES_TO_KEEP:]
            )
            logger.info(
                "Conversation compressed successfully. Current message count: %d",
                len(final_messages),
            )
        llm = get_llm()
        context = (
            "\n\n".join([doc.page_content for doc in state.retreived_results])
            if state.retreived_results else ""
        )
        system_content = "You are a helpful assistant. Answer the user's question clearly and concisely."
        if context:
            system_content += f"\n\nContext:\n{context}"
        if state.summary and not state.summarized_conv:
            system_content += f"\n\nConversation summary so far:\n{state.summary}"

        messages = [SystemMessage(content=system_content)] + final_messages

        response = await llm.ainvoke(messages)
        logger.info("chat_node completed, response_length=%d", len(response.content))
        return {"messages": [response], "ai_response": response.content}
    except Exception as e:
        logger.error("chat_node failed: %s", str(e))
        raise MyException(e, sys)


async def summary_node(state: State):
    try:
        logger.info("=" * 50)
        logger.info("Summary node started.")

        if len(state.messages) <= NO_OF_LAST_MESSAGES_TO_KEEP:
            logger.info(
                "Skipping summary. Messages (%d) <= threshold (%d).",
                len(state.messages),
                NO_OF_LAST_MESSAGES_TO_KEEP,
            )
            return {}

        llm = get_llm()
        logger.info("LLM initialized successfully.")

        old_messages = state.messages[:-NO_OF_LAST_MESSAGES_TO_KEEP]
        logger.info(
            "Messages selected for summarization: %d",
            len(old_messages),
        )

        summary_chain = SUMMARY_NODE_PROMPT | llm

        logger.info("Generating conversation summary...")

        response = summary_chain.invoke(
            {
                "no_of_words": LENGTH_OF_SUMMARY_GENERATED,
                "messages": old_messages,
            }
        )

        summary_text = response.content

        logger.info("Summary generated successfully.")
        logger.debug("Summary: %s", summary_text)

        logger.info("=" * 50)

        return {"summarized_conv": summary_text}

    except Exception as e:
        logger.exception("Error occurred in summary node.")
        raise MyException(e, sys)

