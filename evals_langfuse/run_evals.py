import sys
from dotenv import load_dotenv
load_dotenv()
import os
sys.path.append(os.getcwd())
import asyncio

from langfuse import Langfuse
from src.core.dependencies import setup_langfuse
from src.core.logger import logger
from src.core.exceptions import MyException
from src.utils.main_utils import load_json_file
from evals.metrics.orchastrator_eval import eval_orchastrator, eval_orchastrator_target
from evals.config import OrchastratorEvalConfig, GeneratorEvalConfig, RetreiverEvalConfig
from evals.metrics.generator_eval import eval_generator, eval_generator_target
from evals.metrics.rag_eval import eval_rag, setup_rag_once, eval_rag_target
from src.retrievers.pinecone_retriever import get_retriever
from src.domain.config_entities import RetrieverConfig

# Ensure Langfuse environment is setup from AppConfig
setup_langfuse()
client = Langfuse()


async def run_langfuse_dataset_eval(dataset_name: str, dataset_path: str, eval_name: str, target_func, eval_func, description: str = ""):
    """Helper function to create/sync dataset in Langfuse and run benchmark evaluation items."""
    try:
        # 1. Get or create dataset
        try:
            dataset = client.get_dataset(dataset_name)
            logger.info("Found existing Langfuse dataset: '%s'", dataset_name)
        except Exception:
            logger.info("Creating dataset '%s' in Langfuse...", dataset_name)
            dataset = client.create_dataset(name=dataset_name, description=description)

        # 2. Sync local JSON examples to Langfuse dataset
        raw_examples = await load_json_file(dataset_path)
        existing_items = getattr(dataset, "items", []) or []

        if len(existing_items) != len(raw_examples):
            logger.info("Syncing dataset '%s': local has %d examples. Syncing to Langfuse...", dataset_name, len(raw_examples))
            for ex in raw_examples:
                inp = ex.get("inputs", {})
                out = ex.get("reference_outputs") or ex.get("outputs", {})
                client.create_dataset_item(
                    dataset_name=dataset_name,
                    input=inp,
                    expected_output=out,
                )
            dataset = client.get_dataset(dataset_name)
            existing_items = getattr(dataset, "items", []) or []

        logger.info("Dataset '%s' contains %d items.", dataset_name, len(existing_items))

        # 3. Target task wrapper for dataset items
        async def task_wrapper(*, item, **kwargs):
            inp = item.input if isinstance(item.input, dict) else {"input": item.input}
            return await target_func(inp)

        # 4. Run experiment natively via Langfuse SDK
        logger.info("Executing evaluation runner for experiment: '%s'", eval_name)
        experiment_results = dataset.run_experiment(
            name=eval_name,
            task=task_wrapper,
            evaluators=[eval_func],
            max_concurrency=2,
        )
        return experiment_results

    except Exception as e:
        logger.error("Error running Langfuse dataset evaluation '%s': %s", dataset_name, str(e), exc_info=True)
        raise MyException(e, sys)


# ===================== Orchestrator Evaluation =====================

async def evaluating_orchastrator():
    """Load evaluation dataset and run Langfuse benchmark evaluation for orchestrator node."""
    logger.info("Initializing Orchestrator Evaluation Run...")
    config = OrchastratorEvalConfig()
    results = await run_langfuse_dataset_eval(
        dataset_name=config.dataset_name,
        dataset_path=config.dataset_path,
        eval_name=config.eval_name,
        target_func=eval_orchastrator_target,
        eval_func=eval_orchastrator,
        description="Orchestrator node query routing evaluation dataset."
    )
    logger.info("Orchestrator Evaluation Completed Successfully.")
    return results


# ===================== Generator Evaluation =====================

async def evaluating_generator():
    """Load evaluation dataset and run Langfuse benchmark evaluation for generator (chat_node)."""
    logger.info("Initializing Generator Evaluation Run...")
    config = GeneratorEvalConfig()
    results = await run_langfuse_dataset_eval(
        dataset_name=config.dataset_name,
        dataset_path=config.dataset_path,
        eval_name=config.eval_name,
        target_func=eval_generator_target,
        eval_func=eval_generator,
        description="Generator chat_node evaluation dataset."
    )
    logger.info("Generator Evaluation Completed Successfully.")
    return results


# ===================== Retriever Evaluation =====================

async def evaluating_retreiver():
    """Load evaluation dataset and run Langfuse benchmark evaluation for retriever."""
    logger.info("Initializing Retriever Evaluation Run...")
    config = RetreiverEvalConfig()

    logger.info("Setting up Retriever Configs & Vector Store...")
    await setup_rag_once(
        file_paths=config.file_paths,
        index_name=config.index_name
    )

    try:
        results = await run_langfuse_dataset_eval(
            dataset_name=config.dataset_name,
            dataset_path=config.dataset_path,
            eval_name=config.eval_name,
            target_func=eval_rag_target,
            eval_func=eval_rag,
            description="RAG retriever chunk relevance evaluation dataset."
        )
        logger.info("Retriever Evaluation Completed Successfully.")
        return results
    finally:
        retriever = get_retriever(retriever_config=RetrieverConfig())
        try:
            logger.info("Cleaning up vector store: Deleting namespace '%s'...", config.index_name)
            await retriever.delete_namespace(index_name=config.index_name, namespace="")
            logger.info("Index '%s' cleaned up successfully.", config.index_name)
        except Exception as cleanup_err:
            logger.warning("Failed to clean up Pinecone namespace: %s", str(cleanup_err))


async def main():
    await asyncio.gather(
        evaluating_orchastrator(),
        evaluating_generator(),
        evaluating_retreiver()
    )


if __name__ == "__main__":
    asyncio.run(main())
