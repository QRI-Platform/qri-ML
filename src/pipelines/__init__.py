import sys
from functools import lru_cache
from src.logger import logger
from src.exception import MyException
from src.pipelines.graph_runner_pipeline import GraphRunnerPipeline


@lru_cache
def get_pipeline() -> GraphRunnerPipeline:
    logger.debug("Initializing GraphRunnerPipeline singleton")
    try:
        pipeline = GraphRunnerPipeline()
        logger.info("GraphRunnerPipeline singleton initialized")
        return pipeline
    except Exception as e:
        raise MyException(e, sys)
