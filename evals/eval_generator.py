
import asyncio
import json

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, ErrorConfig
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
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
from src.graphs.builder import get_graph

# ==================== Constants ====================
GOLDEN_PATH = "Generator_Eval"
JUDGE_MODEL = OLLAMA_JUDGE_MODEL
THRESHOLD = 0.7
GENERATOR_NAME = "generator"
BATCH_SIZE = 1
MAX_RETRIES = 3
RATE_LIMIT_BACKOFF_SECONDS = 2

import os

os.environ["DEEPEVAL_CONCURRENCY_LIMIT"] = "1"


# tell deep eval to use llm
judge = get_judge(model=JUDGE_MODEL)

async def _generate_answer(query: str, context: list[str], thread_id: str, retry_count: int = 0) -> str:
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
    state = {"messages": messages}
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "user_id": "eval-user"}
    }

    try:
        result = await graph.ainvoke(
            state,
            config=config,
        )
        return result.get("ai_response", "")
    except Exception as exc:
        error_text = str(exc)
        if "429" in error_text and retry_count < MAX_RETRIES:
            wait_time = RATE_LIMIT_BACKOFF_SECONDS * (retry_count + 1)
            logger.warning(
                "Groq rate limit hit for eval query %s; retrying in %ss (attempt %s/%s)",
                thread_id,
                wait_time,
                retry_count + 1,
                MAX_RETRIES,
            )
            await asyncio.sleep(wait_time)
            return await _generate_answer(query, context, thread_id, retry_count + 1)
        raise


async def run(run_name=None):
    loaded_goldens = load_goldens(GOLDEN_PATH)

    test_cases = []
    async def process_query(g, index):
        context = g["ideal_context"]
        answer = await _generate_answer(g["query"], context, f"{index}")
        return LLMTestCase(
                input=g["query"],
                actual_output=answer,
                retrieval_context=context,
            )

    for i in range(0, len(loaded_goldens), BATCH_SIZE):
        tasks = [
            process_query(g, i + offset)
            for offset, g in enumerate(loaded_goldens[i:i + BATCH_SIZE])
        ]
        test_cases.extend(await asyncio.gather(*tasks))

    metrics = [
        FaithfulnessMetric(
            threshold=THRESHOLD,
            strict_mode=True,
            model=judge,
            include_reason=True,
            async_mode=False,
        ),
        AnswerRelevancyMetric(
            threshold=THRESHOLD,
            strict_mode=True,
            model=judge,
            include_reason=True,
            async_mode=False,
        ),
    ]
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(run_async=False, max_concurrent=1),
        error_config=ErrorConfig(ignore_errors=True),
    )

    summary = summarize_by_metric(result)
    return summary, result


def _configure_eval_runtime():
    """Keep eval execution isolated from the application's Neon Postgres setup."""
    memory_module._checkpointer = MemorySaver()
    memory_module._store = InMemoryStore()


def run_local(run_name=None):
    """Run the generator evaluation with preloaded retrieval tool results."""
    return asyncio.run(run(run_name=run_name))


async def _run_generator_eval_async(verbose=False, run_name=None):
    _configure_eval_runtime()
    summary, result = await run(run_name=run_name)
    if verbose:
        print_summary("generator", summary)
    return summary, result


def run_generator_eval(verbose=False, run_name=None):
    """Run the generator eval with a single top-level event loop."""
    return asyncio.run(_run_generator_eval_async(verbose=verbose, run_name=run_name))


if __name__ == "__main__":
    run_generator_eval(verbose=True)


