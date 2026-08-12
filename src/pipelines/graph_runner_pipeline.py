import sys
from functools import lru_cache
from langchain_core.messages import HumanMessage

from src.core.logger import logger
from src.core.exceptions import MyException
from src.domain.enums import Pipeline
from src.graphs.builder import get_graph
from src.domain.state import State
from langsmith import traceable


class GraphRunnerPipeline(Pipeline):
    def __init__(self):
        self.graph = get_graph()
        logger.info("GraphRunnerPipeline initialized")

    @traceable(name="GraphRunnerPipeline.initiate", run_type="chain")
    async def initiate(self, user_id: str, thread_id: str, file_paths: list = None, message: str = None,has_documents:bool=False):
        try:
            logger.info("Pipeline.initiate called: user=%s thread=%s files=%d message=%s",
                        user_id, thread_id, len(file_paths or []), bool(message))
            state = {
                "file_paths": file_paths or [],
                "messages": [HumanMessage(content=message)] if message else [],
                "has_documents":has_documents
            }

            config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
            logger.debug("Streaming graph events with config=%s", config)

            async for chunk in self.graph.astream_events(state, config=config, version="v2"):
                yield chunk

        except Exception as e:
            logger.error("Pipeline.initiate failed: %s", str(e))
            raise MyException(e, sys)

    

@lru_cache
def get_graph_runner_pipeline() -> GraphRunnerPipeline:
    return GraphRunnerPipeline()
