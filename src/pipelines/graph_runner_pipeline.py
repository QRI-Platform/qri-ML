import sys
from src.logger import logger
from src.exception import MyException
from src.utils.abstract_class import Pipeline
from src.graphs.builder import get_graph
from src.models.workflow_models import State
from langchain_core.messages import HumanMessage


class GraphRunnerPipeline(Pipeline):
    def __init__(self):
        self.graph = get_graph()
        logger.info("GraphRunnerPipeline initialized")

    async def initiate(self, user_id: str, thread_id: str, file_paths: list = None, message: str = None):
        try:
            logger.info("Pipeline.initiate called: user=%s thread=%s files=%d message=%s",
                        user_id, thread_id, len(file_paths or []), bool(message))
            state = State(
                user_id=user_id,
                thread_id=thread_id,
                file_paths=file_paths or [],
            )
            if message:
                state.messages.append(HumanMessage(content=message))

            config = {"configurable": {"thread_id": thread_id}}
            logger.debug("Streaming graph events with config=%s", config)

            async for chunk in self.graph.astream_events(state, config=config, version="v2"):
                yield chunk

        except Exception as e:
            logger.error("Pipeline.initiate failed: %s", str(e))
            raise MyException(e, sys)
