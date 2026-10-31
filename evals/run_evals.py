import os

from dotenv import load_dotenv

load_dotenv()

from evals.eval_application import run_application_eval
from evals.eval_generator import run_generator_eval
from evals.eval_retriever import run_retriever_eval
from evals.harness import push_results_to_langfuse
from src.core.logger import logger

os.environ["DEEPEVAL_CONCURRENCY_LIMIT"] = "1"
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "600")




def run_evals():
    logger.info("Running Retriever Eval")
    summary, results = run_retriever_eval(verbose=True)
    push_results_to_langfuse(
        result=results,
        title="retriever",
        summary=summary,
        run_name="retriever_eval",
        metadata={"dataset_name": "Retriever_Eval"},
    )

    logger.info("Running Generator Eval")
    summary, results = run_generator_eval(verbose=True)
    push_results_to_langfuse(
        result=results,
        title="generator",
        summary=summary,
        run_name="generator_eval",
        metadata={"dataset_name": "Generator_Eval"},
    )

    logger.info("Running Application Eval")
    summary, results = run_application_eval(verbose=True)
    push_results_to_langfuse(
        result=results,
        title="application",
        summary=summary,
        run_name="application_eval",
        metadata={"dataset_name": "Application_Eval"},
    )


if __name__ == "__main__":
    run_evals()
