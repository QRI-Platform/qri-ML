from src.logger import logger
from src.models.workflow_models import State, Query_generation_output, Orchastrator_output
from src.components.data_ingestion import DataIngestion
from src.entity.config import DataIngestionConfig, RetrieverConfig
from src.entity.artifact import DataIngestionArtifact
from src.exception import MyException
import sys
from src.llm.llm_loader import llm
from langchain_core.messages import SystemMessage
from src.prompts import QUERY_GENERATION_PROMPT, ORCHESTRATOR_PROMPT, CHAT_PROMPT
from src.retreiver.retreiver import Retreiver


async def ingestion_node(state: State):
    try:
        data_ingestion_config: DataIngestionConfig = DataIngestionConfig(files_path=state.file_paths, namespace=state.user_id)
        retreiver_config: RetrieverConfig = RetrieverConfig(namespace=state.user_id)
        data_ingestion: DataIngestion = DataIngestion(data_ingestion_config=data_ingestion_config, retreiver_config=retreiver_config)
        data_ingestion_artifact: DataIngestionArtifact = await data_ingestion.ingest()
        return {"retreiver": data_ingestion_artifact.retreiver}
    except Exception as e:
        raise MyException(e, sys)


async def orchastrator_node(state: State) -> dict:
    logger.info("Orchestrator node started")
    structured_llm = llm.with_structured_output(Orchastrator_output, method="json_mode")
    messages = [
        SystemMessage(content=ORCHESTRATOR_PROMPT),
        *state.messages
    ]
    result = structured_llm.invoke(messages)
    logger.info(f"Orchestrator routing decision: require_db_search={result.require_db_search}")
    return {"require_db_search": result.require_db_search}


async def query_generation_node(state: State) -> dict:
    logger.info("Query generation node started")
    structured_llm = llm.with_structured_output(Query_generation_output)
    messages = [
        SystemMessage(content=QUERY_GENERATION_PROMPT),
        *state.messages
    ]
    result = structured_llm.invoke(messages)
    logger.info(f"Generated {len(result.queries)} queries")
    return {"queries": result.queries}


async def retreiver_loader(state: State):
    try:
        retreiver_config = RetrieverConfig(namespace=state.user_id)
        retreiver = Retreiver(retriever_config=retreiver_config)
        vector_store = await retreiver.create_retreiver()
        return {"retreiver": vector_store}
    except Exception as e:
        raise MyException(e, sys)


async def retreiver_node(state: State):
    try:
        retreiver_config = RetrieverConfig(namespace=state.user_id)
        retreiver = Retreiver(retriever_config=retreiver_config)
        vector_store = state.retreiver or await retreiver.create_retreiver()
        results = []
        for query in state.queries:
            docs = await retreiver.get_similar_product(vector_store=vector_store, query=query)
            results.extend(docs)
        logger.info(f"Retreiver node returned {len(results)} documents")
        return {"retreived_results": results}
    except Exception as e:
        raise MyException(e, sys)



async def chat_node(state: State):
    " this is chat node"
    try:
        if state.summery:
            state.messages = [SystemMessage(content=state.messages)] + state.messages
        context = "\n\n".join([doc.page_content for doc in state.retreived_results]) if state.retreived_results else ""
        last_message = state.messages[-1].content if state.messages else ""
        formatted = CHAT_PROMPT.format_messages(
            context=f"\n\nContext:\n{context}" if context else "",
            question=last_message
        )
        response = await llm.ainvoke(formatted)
        logger.info("Chat node completed")
        return {"messages": [response], "ai_response": response.content}
    except Exception as e:
        raise MyException(e, sys)