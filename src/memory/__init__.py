import sys
from functools import lru_cache
from langgraph.checkpoint.memory import MemorySaver
from src.logger import logger
from src.exception import MyException


@lru_cache
def get_checkpointer() -> MemorySaver:
    try:
        logger.debug("Initializing MemorySaver checkpointer singleton")
        cp = MemorySaver()
        logger.info("MemorySaver checkpointer initialized")
        return cp
    except Exception as e:
        raise MyException(e, sys)


checkpointer = get_checkpointer()