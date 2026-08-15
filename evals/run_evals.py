import sys
import os
sys.path.append(os.getcwd())
import asyncio
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from langsmith import Client, aevaluate
from src.core.logger import logger
from src.core.exceptions import MyException
from src.utils.main_utils import load_json_file
from evals.metrics.orchastrator_eval import eval_orchastrator, eval_orchastrator_target
from evals.config import OrchastratorEvalConfig, GeneratorEvalConfig, RetreiverEvalConfig, GraphEvalConfig
from evals.metrics.generator_eval import eval_generator, eval_generator_target
from evals.metrics.rag_eval import eval_rag, setup_rag_once, eval_rag_target
from evals.metrics.graph_eval import eval_graph, eval_graph_target, setup_graph_rag_once
from src.retrievers.pinecone_retriever import get_retriever
from src.domain.config_entities import RetrieverConfig
client = Client()


async def evaluating_orchastrator():
    """Load evaluation dataset and run LangSmith benchmark evaluation for orchestrator node."""
    try:
        logger.info("Initializing Orchestrator Evaluation Run...")
        config = OrchastratorEvalConfig()

        # Check if dataset exists, create if not
        if client.has_dataset(dataset_name=config.dataset_name):
            logger.info("Found existing LangSmith dataset: '%s'", config.dataset_name)
            dataset = client.read_dataset(dataset_name=config.dataset_name)
        else:
            logger.info("Creating dataset '%s' in LangSmith...", config.dataset_name)
            dataset = client.create_dataset(dataset_name=config.dataset_name)

        # Check and sync dataset examples from local JSON
        existing_examples = list(client.list_examples(dataset_id=dataset.id))
        raw_examples = await load_json_file(config.dataset_path)

        if len(existing_examples) != len(raw_examples):
            logger.info("Syncing dataset '%s': local has %d examples, LangSmith has %d. Syncing...", config.dataset_name, len(raw_examples), len(existing_examples))
            for ex in existing_examples:
                client.delete_example(ex.id)
            inputs = [ex["inputs"] for ex in raw_examples]
            outputs = [ex["outputs"] for ex in raw_examples]
            client.create_examples(
                inputs=inputs,
                outputs=outputs,
                dataset_id=dataset.id,
            )
            logger.info("Successfully synced %d examples to dataset ID '%s'", len(raw_examples), dataset.id)
        else:
            logger.info("Dataset '%s' contains %d up-to-date examples.", config.dataset_name, len(existing_examples))

        # Run evaluation
        logger.info("Executing evaluation runner for experiment prefix: '%s'", config.eval_name)
        experiment_results = await aevaluate(
            eval_orchastrator_target,
            data=config.dataset_name,
            evaluators=[eval_orchastrator],
            experiment_prefix=config.eval_name,
            client=client,
        )

        logger.info("Orchestrator Evaluation Completed Successfully.")
        return experiment_results

    except Exception as e:
        logger.error("Failed to run orchestrator evaluation: %s", str(e), exc_info=True)
        raise MyException(e, sys)




# ===================== Generator =====================================

async def evaluating_generator():
    """Load evaluation dataset and run LangSmith benchmark evaluation for generator (chat_node)."""
    try:
        logger.info("Initializing Generator Evaluation Run...")
        config = GeneratorEvalConfig()

        # Check if dataset exists, create if not
        if client.has_dataset(dataset_name=config.dataset_name):
            logger.info("Found existing LangSmith dataset: '%s'", config.dataset_name)
            dataset = client.read_dataset(dataset_name=config.dataset_name)
        else:
            logger.info("Creating dataset '%s' in LangSmith...", config.dataset_name)
            dataset = client.create_dataset(
                dataset_name=config.dataset_name,
                description="Evaluation dataset for chat_node generator responses, tool calls, and memory retrieval.",
            )

        # Check and sync dataset examples from local JSON
        existing_examples = list(client.list_examples(dataset_id=dataset.id))
        raw_examples = await load_json_file(config.dataset_path)

        if len(existing_examples) != len(raw_examples):
            logger.info(
                "Syncing dataset '%s': local has %d examples, LangSmith has %d. Syncing...",
                config.dataset_name,
                len(raw_examples),
                len(existing_examples),
            )
            for ex in existing_examples:
                client.delete_example(ex.id)

            inputs = [ex["inputs"] for ex in raw_examples]
            outputs = [
                ex.get("reference_outputs") or ex.get("outputs") 
                for ex in raw_examples
            ]

            client.create_examples(
                inputs=inputs,
                outputs=outputs,
                dataset_id=dataset.id,
            )
            logger.info("Successfully synced %d examples to dataset ID '%s'", len(raw_examples), dataset.id)
        else:
            logger.info("Dataset '%s' contains %d up-to-date examples.", config.dataset_name, len(existing_examples))

        # Run evaluation
        logger.info("Executing evaluation runner for experiment prefix: '%s'", config.eval_name)
        experiment_results = await aevaluate(
            eval_generator_target,
            data=config.dataset_name,
            evaluators=[eval_generator],
            experiment_prefix=config.eval_name,
            client=client,
            max_concurrency=config.max_concurrency if hasattr(config, "max_concurrency") else 2,
        )

        logger.info("Generator Evaluation Completed Successfully.")
        return experiment_results

    except Exception as e:
        logger.error("Failed to run generator evaluation: %s", str(e), exc_info=True)
        raise MyException(e, sys)




# ===================== Retriever =====================================

async def evaluating_retreiver():
    """Load evaluation dataset and run LangSmith benchmark evaluation for retriever."""
    try:
        logger.info("Initializing Retriever Evaluation Run...")
        config = RetreiverEvalConfig()

        logger.info("Setting up Retriever Configs & Vector Store...")
        # 1. Await setup_rag_once properly
        await setup_rag_once(
            file_paths=config.file_paths,
            index_name=config.index_name
        )

        # 2. Check if dataset exists, create if not
        if client.has_dataset(dataset_name=config.dataset_name):
            logger.info("Found existing LangSmith dataset: '%s'", config.dataset_name)
            dataset = client.read_dataset(dataset_name=config.dataset_name)
        else:
            logger.info("Creating dataset '%s' in LangSmith...", config.dataset_name)
            dataset = client.create_dataset(
                dataset_name=config.dataset_name,
                description="Evaluation dataset for RAG retriever chunk matching and context relevance.",
            )

        # 3. Check and sync dataset examples from local JSON
        existing_examples = list(client.list_examples(dataset_id=dataset.id))
        raw_examples = await load_json_file(config.dataset_path)

        if len(existing_examples) != len(raw_examples):
            logger.info(
                "Syncing dataset '%s': local has %d examples, LangSmith has %d. Syncing...",
                config.dataset_name,
                len(raw_examples),
                len(existing_examples),
            )
            for ex in existing_examples:
                client.delete_example(ex.id)

            inputs = [ex["inputs"] for ex in raw_examples]
            outputs = [
                ex.get("reference_outputs") or ex.get("outputs") 
                for ex in raw_examples
            ]

            client.create_examples(
                inputs=inputs,
                outputs=outputs,
                dataset_id=dataset.id,
            )
            logger.info("Successfully synced %d examples to dataset ID '%s'", len(raw_examples), dataset.id)
        else:
            logger.info("Dataset '%s' contains %d up-to-date examples.", config.dataset_name, len(existing_examples))

        # 4. Run evaluation using eval_rag_target and eval_rag
        logger.info("Executing evaluation runner for experiment prefix: '%s'", config.eval_name)
        experiment_results = await aevaluate(
            eval_rag_target,                     
            data=config.dataset_name,
            evaluators=[eval_rag],               
            experiment_prefix=config.eval_name,
            client=client,
            max_concurrency=getattr(config, "max_concurrency", 2),
        )

        logger.info("Retriever Evaluation Completed Successfully.")
        return experiment_results

    except Exception as e:
        logger.error("Failed to run retriever evaluation: %s", str(e), exc_info=True)
        raise MyException(e, sys)

    finally:
        # 5. Cleanup: Guaranteed deletion of namespace after eval (Success or Error)
        retriever = get_retriever(retriever_config=RetrieverConfig())
        try:
            logger.info(
                "Cleaning up vector store: Deleting namespace '%s' from index '%s'...",
                config.index_name,
            )
            await retriever.delete_namespace(
                index_name=config.index_name,
                namespace = ""
            )
            logger.info("index '%s' cleaned up successfully.", config.index_name)
        except Exception as cleanup_err:
            logger.warning("Failed to clean up Pinecone namespace: %s", str(cleanup_err))


# ===================== Complete Graph =====================================

async def evaluating_graph():
    """Load evaluation dataset and run LangSmith benchmark evaluation for complete StateGraph pipeline."""
    config = GraphEvalConfig()
    try:
        logger.info("Initializing Graph End-to-End Evaluation Run...")

        logger.info("Setting up Graph Configs & Vector Store...")
        await setup_graph_rag_once(
            file_paths=config.file_paths,
            index_name=config.index_name,
            namespace=config.thread_id,
        )

        # Check if dataset exists, create if not
        if client.has_dataset(dataset_name=config.dataset_name):
            logger.info("Found existing LangSmith dataset: '%s'", config.dataset_name)
            dataset = client.read_dataset(dataset_name=config.dataset_name)
        else:
            logger.info("Creating dataset '%s' in LangSmith...", config.dataset_name)
            dataset = client.create_dataset(
                dataset_name=config.dataset_name,
                description="Evaluation dataset for complete LangGraph input-output flow.",
            )

        # Check and sync dataset examples from local JSON
        existing_examples = list(client.list_examples(dataset_id=dataset.id))
        raw_examples = await load_json_file(config.dataset_path)

        if len(existing_examples) != len(raw_examples):
            logger.info(
                "Syncing dataset '%s': local has %d examples, LangSmith has %d. Syncing...",
                config.dataset_name,
                len(raw_examples),
                len(existing_examples),
            )
            for ex in existing_examples:
                client.delete_example(ex.id)

            inputs = [ex["inputs"] for ex in raw_examples]
            outputs = [
                ex.get("reference_outputs") or ex.get("outputs") 
                for ex in raw_examples
            ]

            client.create_examples(
                inputs=inputs,
                outputs=outputs,
                dataset_id=dataset.id,
            )
            logger.info("Successfully synced %d examples to dataset ID '%s'", len(raw_examples), dataset.id)
        else:
            logger.info("Dataset '%s' contains %d up-to-date examples.", config.dataset_name, len(existing_examples))

        # Run evaluation
        logger.info("Executing evaluation runner for experiment prefix: '%s'", config.eval_name)
        experiment_results = await aevaluate(
            eval_graph_target,
            data=config.dataset_name,
            evaluators=[eval_graph],
            experiment_prefix=config.eval_name,
            client=client,
            max_concurrency=getattr(config, "max_concurrency", 2),
        )

        logger.info("Graph Evaluation Completed Successfully.")
        return experiment_results

    except Exception as e:
        logger.error("Failed to run graph evaluation: %s", str(e), exc_info=True)
        raise MyException(e, sys)

    finally:
        # Cleanup vector store namespace after graph eval execution
        retriever = get_retriever(retriever_config=RetrieverConfig(index_name=config.index_name, namespace=config.thread_id))
        try:
            logger.info(
                "Cleaning up vector store: Deleting namespace '%s' from index '%s'...",
                config.thread_id,
                config.index_name,
            )
            await retriever.delete_namespace(
                index_name=config.index_name,
                namespace=config.thread_id,
            )
            logger.info("Namespace '%s' in index '%s' cleaned up successfully.", config.thread_id, config.index_name)
        except Exception as cleanup_err:
            logger.warning("Failed to clean up Pinecone namespace '%s': %s", config.thread_id, str(cleanup_err))

async def main():
    await asyncio.gather(
        evaluating_orchastrator(),
        evaluating_generator(),
        evaluating_retreiver(),
        evaluating_graph()
    )  

if __name__ == "__main__":
    # asyncio.run(evaluating_graph())
    asyncio.run(main())

