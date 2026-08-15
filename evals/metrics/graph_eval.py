import sys
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from unittest.mock import patch

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langsmith.evaluation import EvaluationResult
from evals.common import get_llm
from langchain_core.prompts import ChatPromptTemplate

from src.core.logger import logger
from src.core.exceptions import MyException
from src.domain.state import State
from src.graphs.builder import get_graph
from src.services.data_ingestion_service import DataIngestion
from src.retrievers.pinecone_retriever import get_retriever, Retriever
from src.domain.config_entities import RetrieverConfig, DataIngestionConfig


# --- Pydantic Schema for LLM Judge ---
class AccuracyOutput(BaseModel):
    score: float = Field(
        ...,
        description="Float score between 0.0 and 1.0 indicating how accurately and completely the generated response answers the question."
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation justifying the assigned score."
    )


async def setup_graph_rag_once(file_paths: List[str], index_name: str, namespace: str):
    """Initializes vector store and ingests evaluation documents into Pinecone namespace for graph evaluation."""
    logger.info(
        "Graph Eval: Ingesting files %s into Pinecone index '%s' namespace '%s'...",
        file_paths,
        index_name,
        namespace,
    )
    retriever_config = RetrieverConfig(
        index_name=index_name,
        namespace=namespace,
    )
    data_ingestion_config = DataIngestionConfig(
        index_name=index_name,
        files_path=file_paths,
        namespace=namespace,
    )

    retriever = get_retriever(retriever_config=retriever_config)
    data_ingestor = DataIngestion(data_ingestion_config=data_ingestion_config, retriever=retriever)
    await data_ingestor.ingest()
    logger.info("Graph Eval: Ingestion into Pinecone namespace '%s' completed successfully.", namespace)


async def eval_graph_target(inputs: dict) -> dict:
    """Target function executing complete LangGraph pipeline end-to-end natively without mocking retrievers.

    Args:
        inputs: Dict containing 'question', optional 'user_memories',
                optional 'user_id', and optional 'thread_id'.

    Returns:
        Dict containing predicted 'ai_response', 'require_db_search',
        'has_tool_call', 'tool_calls', and 'messages'.
    """
    try:
        logger.info("Executing eval_graph_target")

        question = inputs.get("question", "Hello, this is a test message.")
        user_memories = inputs.get("user_memories", {})
        user_id = inputs.get("user_id", "eval_user_id")
        thread_id = inputs.get("thread_id", "eval_graph_thread")

        # 1. Setup in-memory checkpointer & store populated with initial user memories
        store = InMemoryStore()
        checkpointer = MemorySaver()
        for k, v in user_memories.items():
            await store.aput(namespace=("user", str(user_id), "details"), key=k, value={"data": v})

        # 2. Construct initial State
        state: State = {
            "messages": [HumanMessage(content=question)],
            "file_paths": [],
            "require_db_search": False,
            "has_documents": False,
            "queries": [],
            "retreived_results": [],
            "ai_response": None,
        }

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id,
            }
        }

        # 3. Clear get_graph cache to recompile with in-memory checkpointer and store
        get_graph.cache_clear()

        with patch("src.graphs.builder.get_checkpointer", return_value=checkpointer), \
             patch("src.graphs.builder.get_store", return_value=store):

            graph = get_graph()
            res = await graph.ainvoke(state, config=config)

        # 4. Extract graph execution results
        res_messages = res.get("messages", [])
        ai_response = res.get("ai_response")
        if not ai_response and res_messages:
            last_msg = res_messages[-1]
            if isinstance(last_msg, AIMessage):
                ai_response = last_msg.content
            elif hasattr(last_msg, "content"):
                ai_response = str(last_msg.content)

        tool_calls = []
        has_tool_call = False
        for msg in res_messages:
            t_calls = getattr(msg, "tool_calls", []) or []
            if t_calls:
                has_tool_call = True
                tool_calls.extend(t_calls)

        require_db_search = res.get("require_db_search", False)

        return {
            "ai_response": ai_response or "",
            "require_db_search": require_db_search,
            "has_tool_call": has_tool_call,
            "tool_calls": tool_calls,
            "messages": res_messages,
        }

    except Exception as e:
        logger.error("Error in eval_graph_target: %s", str(e), exc_info=True)
        raise MyException(e, sys)


async def eval_graph(inputs: dict, outputs: dict, reference_outputs: dict) -> EvaluationResult:
    """Evaluator function comparing predicted complete graph output against ground truth.

    Evaluates:
      1. DB Search routing accuracy (require_db_search)
      2. Tool call trigger correctness (has_tool_call)
      3. Semantic response alignment via LLM Judge
    """
    try:
        # 1. Routing Accuracy Check (if reference expects it)
        expected_db_search = reference_outputs.get("require_db_search")
        actual_db_search = outputs.get("require_db_search")
        if expected_db_search is not None and expected_db_search != actual_db_search:
            return EvaluationResult(
                key="graph_accuracy",
                score=0.0,
                comment=f"Orchestrator routing mismatch. Expected require_db_search={expected_db_search}, Got: {actual_db_search}",
                metadata={"routing_matched": False},
            )

        # 2. Tool Call Accuracy Check (if reference expects it)
        expected_tool_call = reference_outputs.get("has_tool_call")
        actual_tool_call = outputs.get("has_tool_call")
        if expected_tool_call is not None and expected_tool_call != actual_tool_call:
            return EvaluationResult(
                key="graph_accuracy",
                score=0.0,
                comment=f"Tool call mismatch. Expected has_tool_call={expected_tool_call}, Got: {actual_tool_call}",
                metadata={"tool_call_matched": False},
            )

        # If a tool call was expected and triggered properly, score 1.0
        if expected_tool_call and actual_tool_call:
            return EvaluationResult(
                key="graph_accuracy",
                score=1.0,
                comment="Successfully routed through graph and executed tool node.",
                metadata={"tool_call_matched": True},
            )

        # 3. Semantic Content Evaluation via LLM Judge
        expected_response = reference_outputs.get("ai_response", "")
        actual_response = outputs.get("ai_response", "")

        llm = get_llm()
        prompt = ChatPromptTemplate.from_template(
            """You are an accurate AI evaluation judge evaluating end-to-end graph workflow execution outputs.
Compare the generated LLM response with the expected reference response.

[User Question]: {question}
[Expected Response]: {expected_response}
[Generated LLM Response]: {llm_response}

Score the similarity and accuracy between 0.0 and 1.0."""
        )

        chain = prompt | llm.with_structured_output(AccuracyOutput)
        judge_result = await chain.ainvoke({
            "question": inputs.get("question", ""),
            "expected_response": expected_response,
            "llm_response": actual_response,
        })

        return EvaluationResult(
            key="graph_accuracy",
            score=judge_result.score,
            comment=judge_result.reasoning,
            metadata={
                "routing_matched": True,
                "tool_call_matched": True,
                "reasoning": judge_result.reasoning,
            },
        )

    except Exception as e:
        logger.error("Error in eval_graph evaluator: %s", str(e), exc_info=True)
        return EvaluationResult(
            key="graph_accuracy",
            score=0.0,
            comment=f"Evaluator exception: {str(e)}",
        )
