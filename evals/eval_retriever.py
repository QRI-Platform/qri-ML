
import asyncio
import glob

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, ErrorConfig
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase

from evals.common import OLLAMA_JUDGE_MODEL, get_judge
from evals.harness import (
    load_goldens,
    print_summary,
    summarize_by_metric,
)
from src.core.logger import logger
from src.domain.config_entities import DataIngestionConfig, RetrieverConfig
from src.retrievers.pinecone_retriever import Retriever, get_retriever
from src.services.data_ingestion_service import DataIngestion

# ==================== Constants ====================
GOLDEN_PATH = "Retriever_Eval"
JUDGE_MODEL = OLLAMA_JUDGE_MODEL
THRESHOLD = 0.7
RETRIEVER_NAME = "MyRetriever"
EMBEDDING_MODEL = "text-embedding-3-large"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 3
RETRIEVER_NAMESPACE = "retriever_eval"
DATA_OVERRIDE = False
BATCH_SIZE = 3



# ==================== Ingest Document ====================

async def ingest_document(folder_path: str, retriever: Retriever | None = None):
    """Ingest all PDFs in folder_path into the evaluation namespace using DataIngestion."""
    file_paths = glob.glob(f"{folder_path}/*.pdf")
    if not file_paths:
        logger.warning("No PDF files found in %s", folder_path)
        return None

    r = retriever or get_retriever(
        RetrieverConfig(
            model_name=EMBEDDING_MODEL,
            k=TOP_K,
            namespace=RETRIEVER_NAMESPACE,
        )
    )

    ingest_config = DataIngestionConfig(
        files_path=file_paths,
        index_name=r.retriever_config.index_name,
        namespace=r.retriever_config.namespace,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    ingestion = DataIngestion(data_ingestion_config=ingest_config, retriever=r)
    logger.info("Ingesting %d files into namespace '%s'...", len(file_paths), r.retriever_config.namespace)
    artifact = await ingestion.ingest()
    logger.info("Data ingestion completed.")
    return artifact


# tell deep eval to use llm
judge = get_judge(model=JUDGE_MODEL)


async def run(retriever: Retriever, run_name=None):
    try:
        loaded_goldens = load_goldens(GOLDEN_PATH)

        test_cases = []
        async def process_query(g):
            query = g["query"]
            vector_store = await retriever.create_retriever()

            filters: list[str] | None = None
            retrieved = []

            search_task = retriever.get_similar_documents(
                vector_store=vector_store, query=query, filter=filters
            )
            gathered = await asyncio.gather(search_task)
            for docs in gathered:
                retrieved.extend(docs)

            retrieval_context = [doc.page_content for doc in retrieved]

            return \
                LLMTestCase(
                    input=g["query"],
                    expected_output=g["ideal_answer"],
                    retrieval_context=retrieval_context,
                    actual_output="",
                )


        for i in range(0, len(loaded_goldens), BATCH_SIZE):
            tasks = [process_query(g) for g in loaded_goldens[i:i+BATCH_SIZE]]

            test_cases.extend(await asyncio.gather(*tasks))

        metrics = [
            ContextualRecallMetric(
                threshold=THRESHOLD,
                strict_mode=True,
                model=judge,
                include_reason=True,
                async_mode=False,
            ),
            ContextualPrecisionMetric(
                threshold=THRESHOLD,
                strict_mode=True,
                model=judge,
                include_reason=True,
                async_mode=False,
            ),
        ]

        hyperparameters = {
            "retriever": RETRIEVER_NAME,
            "embedding_model": EMBEDDING_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "top_k": TOP_K,
            "judge_model": JUDGE_MODEL,
            "golden_set": GOLDEN_PATH,
        }

        result = evaluate(
            test_cases=test_cases,
            metrics=metrics,
            hyperparameters=hyperparameters,
            async_config=AsyncConfig(run_async=False, max_concurrent=1),
            error_config=ErrorConfig(ignore_errors=True),
        )

        summary = summarize_by_metric(result)
        return summary, result

    finally:
        # Clean up created namespace
        logger.info("Deleting namespace")
        await retriever.delete_namespace(
            index_name=retriever.retriever_config.index_name,
            namespace=RETRIEVER_NAMESPACE,
        )




def run_local(run_name=None, folder_to_ingest: str | None = None):
    """Standalone convenience: optionally ingest, build the retriever, then run."""
    retriever_config = RetrieverConfig(
        model_name=EMBEDDING_MODEL,
        k=TOP_K,
        namespace=RETRIEVER_NAMESPACE,
    )

    retriever = get_retriever(retriever_config)

    async def _execute():
        if folder_to_ingest:
            await ingest_document(folder_to_ingest, retriever=retriever)
        return await run(retriever, run_name=run_name)

    return asyncio.run(_execute())


def run_retriever_eval(verbose=False):
    summary, result = run_local(folder_to_ingest="data/evals")
    if verbose:
        print_summary("retriever", summary)
    return summary,result

if __name__ == "__main__":
    run_retriever_eval(verbose=True)


