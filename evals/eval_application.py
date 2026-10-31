import asyncio
import glob
import json
import uuid

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, ErrorConfig
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from evals.common import OLLAMA_JUDGE_MODEL, get_judge
from evals.harness import (
    load_goldens,
    print_summary,
    summarize_by_metric,
)
from src.core import memory as memory_module
from src.core.logger import logger
from src.domain.config_entities import DataIngestionConfig, RetrieverConfig
from src.graphs.builder import get_graph
from src.retrievers.pinecone_retriever import get_retriever
from src.services.data_ingestion_service import DataIngestion

GOLDEN_PATH = "Application_Eval"
DATASET_PATH = "data/evals"
JUDGE_MODEL = OLLAMA_JUDGE_MODEL
THRESHOLD = 0.7
EMBEDDING_MODEL = "text-embedding-3-large"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 3
BATCH_SIZE = 3
APPLICATION_NAME = "application"

judge = get_judge(model=JUDGE_MODEL)


async def _generate_answer(
    query: str,
    context: list[str],
    thread_id: str,
) -> str:
    graph = get_graph()
    tool_call_id = f"eval-retriever-{thread_id}"
    messages = [
        HumanMessage(content=query),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "retreiver",
                    "args": {"queries": [query]},
                    "id": tool_call_id,
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            name="retreiver",
            tool_call_id=tool_call_id,
            content=json.dumps(
                {"retreived_results": [{"page_content": item} for item in context]}
            ),
        ),
    ]
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "user_id": "eval-user"}
    }

    result = await graph.ainvoke({"messages": messages}, config=config)
    return result.get("ai_response", "")


def _build_metrics() -> list[GEval]:
    shared_params = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ]
    correctness = GEval(
        name="Correctness",
        evaluation_steps=[
            "Compare factual claims in the actual output with the expected output.",
            "Penalize claims that are factually false or contradict the expected output.",
            "Do not penalize brevity, omitted elaboration, or additional correct information.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="All stated claims are correct and consistent."),
            Rubric(score_range=(5, 8), expected_outcome="Mostly correct, with a minor inaccuracy."),
            Rubric(score_range=(0, 4), expected_outcome="Contains a clear factual error or contradiction."),
        ],
        evaluation_params=shared_params,
        threshold=THRESHOLD,
        strict_mode=True,
        model=judge,
        async_mode=False,

    )
    completeness = GEval(
        name="Completeness",
        evaluation_steps=[
            "Identify the key points in the expected output.",
            "Assess how many key points the actual output addresses.",
            "Judge coverage only; correctness is scored separately.",
            "Do not penalize additional information beyond the expected output.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="Addresses essentially all key points."),
            Rubric(score_range=(5, 8), expected_outcome="Covers the main points but misses some detail."),
            Rubric(score_range=(0, 4), expected_outcome="Misses several key points."),
        ],
        evaluation_params=shared_params,
        threshold=THRESHOLD,
        strict_mode=True,
        model=judge,
        async_mode=False,
    )
    style = GEval(
        name="Style",
        evaluation_steps=[
            "Judge only the teaching style and tone of the actual output.",
            "Reward clear, intuitive, conversational explanations in prose.",
            "Do not judge factual correctness, completeness, or answer length.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="Intuitive, conversational teaching voice."),
            Rubric(score_range=(7, 8), expected_outcome="Clear, conversational, well-explained prose."),
            Rubric(score_range=(4, 6), expected_outcome="Understandable but formal, flat, or list-heavy."),
            Rubric(score_range=(0, 3), expected_outcome="Dry, unclear, or jargon-heavy response."),
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=THRESHOLD,
        strict_mode=True,
        model=judge,
        async_mode=False,
    )
    return [correctness, completeness, style]


async def run(run_name: str | None = None):
    goldens = load_goldens(GOLDEN_PATH)
    file_paths = sorted(glob.glob(f"{DATASET_PATH}/*.pdf"))
    if not file_paths:
        raise FileNotFoundError(f"No PDF files found in {DATASET_PATH}")

    namespace = f"application_eval_{uuid.uuid4().hex}"
    retriever_config = RetrieverConfig(
        model_name=EMBEDDING_MODEL,
        k=TOP_K,
        namespace=namespace,
    )
    retriever = get_retriever(retriever_config)

    try:
        ingestion = DataIngestion(
            data_ingestion_config=DataIngestionConfig(
                files_path=file_paths,
                index_name=retriever_config.index_name,
                namespace=namespace,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            ),
            retriever=retriever,
        )
        await ingestion.ingest()
        vector_store = await retriever.create_retriever()
        test_cases = []

        async def process_query(golden: dict, index: int) -> LLMTestCase:
            docs = await retriever.get_similar_documents(
                vector_store=vector_store,
                query=golden["question"],
            )
            context = [doc.page_content for doc in docs]
            actual_output = await _generate_answer(
                golden["question"],
                context,
                f"{namespace}-{index}",
            )
            return LLMTestCase(
                input=golden["question"],
                actual_output=actual_output,
                expected_output=golden["ideal_answer"],
                retrieval_context=context,
            )

        for start in range(0, len(goldens), BATCH_SIZE):
            batch = goldens[start : start + BATCH_SIZE]
            test_cases.extend(
                await asyncio.gather(
                    *(process_query(golden, start + offset) for offset, golden in enumerate(batch))
                )
            )

        result = evaluate(
            test_cases=test_cases,
            metrics=_build_metrics(),
            async_config=AsyncConfig(run_async=False, max_concurrent=1),
            error_config=ErrorConfig(ignore_errors=True),
            hyperparameters={
                "embedding_model": EMBEDDING_MODEL,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "top_k": TOP_K,
                "judge_model": JUDGE_MODEL,
                "golden_set": GOLDEN_PATH,
            },
        )
        return summarize_by_metric(result), result
    finally:
        logger.info("Deleting namespace")
        await retriever.delete_namespace(
            index_name=retriever_config.index_name,
            namespace=namespace,
        )


def _configure_eval_runtime():
    """Keep eval execution isolated from the app's Neon Postgres-backed memory."""
    memory_module._checkpointer = MemorySaver()
    memory_module._store = InMemoryStore()


def run_local(run_name: str | None = None):
    """Run application evaluation against a fresh ingestion of the PDF dataset."""
    return asyncio.run(run(run_name=run_name))


def run_application_eval(verbose: bool = False, run_name: str | None = None):
    _configure_eval_runtime()
    summary, result = run_local(run_name=run_name)
    if verbose:
        print_summary(APPLICATION_NAME, summary)

    return summary, result


if __name__ == "__main__":
    run_application_eval(verbose=True)
