import sys
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.documents import Document
from evals.common import get_llm, EvaluationResult
from langchain_core.prompts import ChatPromptTemplate

from src.core.logger import logger
from src.core.exceptions import MyException
from src.services.data_ingestion_service import DataIngestion
from src.retrievers.pinecone_retriever import get_retriever, Retriever
from src.domain.config_entities import RetrieverConfig, DataIngestionConfig


# --- Pydantic Schema for LLM Judge ---
class RetrieverAccuracyOutput(BaseModel):
    score: float = Field(
        ..., 
        description="Float score between 0.0 and 1.0 indicating if the retrieved documents contain the expected answer."
    )
    reasoning: str = Field(
        ..., 
        description="Brief explanation of why the retrieved documents match or do not match the expected answer."
    )


# Module-level singletons for cached retriever/vector store
_CACHED_RETRIEVER: Optional[Retriever] = None
_CACHED_VECTOR_STORE = None


async def setup_rag_once(file_paths:List[str],index_name):
    """Initializes and caches the Pinecone vector store and retriever once."""
    global _CACHED_RETRIEVER, _CACHED_VECTOR_STORE
    if _CACHED_VECTOR_STORE is not None and _CACHED_RETRIEVER is not None:
        return _CACHED_RETRIEVER, _CACHED_VECTOR_STORE

    index_name: str = index_name
    file_paths: List[str] = file_paths

    retriever_config = RetrieverConfig(
        index_name=index_name
    )
    data_ingestion_config = DataIngestionConfig(
        index_name=index_name,
        files_path=file_paths
    )

    retriever = get_retriever(retriever_config=retriever_config)

    # Ingest document into vector store
    data_ingestor = DataIngestion(data_ingestion_config=data_ingestion_config, retriever=retriever)
    await data_ingestor.ingest()

    vector_store = await retriever.create_retriever()
    
    _CACHED_RETRIEVER = retriever
    _CACHED_VECTOR_STORE = vector_store


async def eval_rag_target(inputs: dict) -> dict:
    """Target function for evaluation that runs Pinecone similarity search."""
    try:
        retriever, vector_store = _CACHED_RETRIEVER,_CACHED_VECTOR_STORE
        question = inputs.get("question", "")

        results = await retriever.get_similar_documents(
            vector_store=vector_store,
            query=question,
        )

        return {"retreived_results": [p.page_content for p in results]}
    except Exception as e:
        logger.error("Error in eval_rag_target: %s", str(e), exc_info=True)
        raise MyException(e, sys)


async def eval_rag(input: dict = None, output: dict = None, expected_output: dict = None, **kwargs) -> EvaluationResult:
    """Evaluator function comparing retrieved documents against ground truth."""
    try:
        inp = input or kwargs.get("inputs", {})
        out = output or kwargs.get("outputs", {})
        exp = expected_output or kwargs.get("reference_outputs", {})

        got_results = out.get("retreived_results", [])
        expected_response = exp.get("expected_retreiver_response", "")
        question = inp.get("question", "")

        # Format all retrieved chunks into a numbered context block
        retrieved_context = "\n\n---\n\n".join(
            [f"[Chunk {idx + 1}]:\n{chunk}" for idx, chunk in enumerate(got_results)]
        ) if got_results else "No documents retrieved."

        llm = get_llm()
        prompt = ChatPromptTemplate.from_template(
            """You are an accurate AI evaluation judge for a RAG (Retrieval-Augmented Generation) system.

Your job is to evaluate whether ANY of the retrieved document chunks contain the information required by the expected reference response.

[User Question]:
{question}

[Expected Reference Answer]:
{expected_response}

[Retrieved Documents]:
{retrieved_context}

Evaluation Instructions:
1. Examine all the retrieved chunks carefully.
2. If at least ONE retrieved chunk contains the relevant facts, answer, or semantic information stated in the Expected Reference Answer, assign a score close to or equal to 1.0.
3. If none of the retrieved chunks contain the required information, assign a score of 0.0.
4. Give a brief, concise reasoning justifying your score."""
        )

        chain = prompt | llm.with_structured_output(RetrieverAccuracyOutput)
        judge_result = await chain.ainvoke({
            "question": question,
            "expected_response": expected_response,
            "retrieved_context": retrieved_context,
        })

        return EvaluationResult(
            name="retriever_accuracy",
            value=judge_result.score,
            comment=judge_result.reasoning,
            metadata={
                "retrieved_chunks_count": len(got_results),
                "reasoning": judge_result.reasoning,
            },
        )

    except Exception as e:
        logger.error("Error in eval_rag evaluator: %s", str(e), exc_info=True)
        return EvaluationResult(
            name="retriever_accuracy",
            value=0.0,
            comment=f"Evaluator exception: {str(e)}",
        )