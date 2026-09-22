import sys
from functools import lru_cache
from langchain_core.messages import HumanMessage

from src.core.logger import logger
from src.core.exceptions import MyException
from src.domain.enums import Pipeline
from src.graphs.builder import get_graph
from src.domain.state import State
from langfuse import observe, propagate_attributes
from langfuse.langchain import CallbackHandler
from src.llm.llm_loader import get_llm
from typing import Literal

class GraphRunnerPipeline(Pipeline):
    def __init__(self):
        self.graph = get_graph()
        logger.info("GraphRunnerPipeline initialized")

    @observe(name="GraphRunnerPipeline.initiate")
    async def initiate(self, user_id: str, 
                       thread_id: str, 
                       file_paths: list = None, 
                       message: str = None,
                       metadata: dict = {}, 
                       need_title: bool = False,
                       need_test_paper:bool=False,
                       total_no_of_questions:int=10,
                       no_of_easy_questions:int=2,
                       no_of_medium_questions:int=4,
                       no_of_hard_questions:int=4,
                       level:Literal['easy','medium','hard']='medium',
                       subject_name:str="maths",
                       exam_type:str="IIT_JEE"
                       ):
        try:
            logger.info("Pipeline.initiate called: user=%s thread=%s files=%d message=%s",
                        user_id, thread_id, len(file_paths or []), bool(message))

            if not file_paths:
                state = {
                    "file_paths": file_paths or [],
                    "messages": [HumanMessage(content=message)] if message else [],
                    "need_title": need_title,
                    "need_test_paper":need_test_paper,
                    "total_no_of_questions":total_no_of_questions,
                    "no_of_easy_questions":no_of_easy_questions,
                    "no_of_medium_questions":no_of_medium_questions,
                    "no_of_hard_questions":no_of_hard_questions,
                    "level":level,
                    "subject_name":subject_name,
                    "exam_type":exam_type
                }
            else:
                state = {
                    "file_paths": file_paths or [],
                    "messages": [HumanMessage(content=message)] if message else [],
                    "has_documents":True,
                    "need_title": need_title,
                    "need_test_paper":need_test_paper,
                    "total_no_of_questions":total_no_of_questions,
                    "no_of_easy_questions":no_of_easy_questions,
                    "no_of_medium_questions":no_of_medium_questions,
                    "no_of_hard_questions":no_of_hard_questions,
                    "level":level,
                    "subject_name":subject_name,
                    "exam_type":exam_type
                }
                

            callbacks = []
            try:
                callbacks.append(CallbackHandler())
            except Exception as fe:
                logger.warning("Could not initialize Langfuse callback handler: %s", fe)

            config = {
                "configurable": {"thread_id": thread_id, "user_id": user_id, "metadata": metadata},
                "callbacks": callbacks,
            }
            logger.debug("Streaming graph events with config=%s", config)

            with propagate_attributes(session_id=thread_id, user_id=user_id):
                async for chunk in self.graph.astream_events(state, config=config, version="v2"):
                    yield chunk

        except Exception as e:
            logger.error("Pipeline.initiate failed: %s", str(e))
            raise MyException(e, sys)




@lru_cache
def get_graph_runner_pipeline() -> GraphRunnerPipeline:
    return GraphRunnerPipeline()
