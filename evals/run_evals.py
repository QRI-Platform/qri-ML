import sys
import asyncio
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from langsmith import Client, aevaluate
from src.core.logger import logger
from src.core.exceptions import MyException
from src.utils.main_utils import load_json_file
from evals.metrics.orchastrator_eval import eval_orchastrator, eval_orchastrator_target
from evals.config import OrchastratorEvalConfig

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


if __name__ == "__main__":
    asyncio.run(evaluating_orchastrator())
