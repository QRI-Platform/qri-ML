import sys
from functools import lru_cache
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from src.core.logger import logger
from src.core.exceptions import MyException


@lru_cache
def get_checkpointer() -> MemorySaver:
    try:
        logger.debug("Initializing MemorySaver checkpointer singleton")
        cp = MemorySaver()
        logger.info("MemorySaver checkpointer initialized")
        return cp
    except Exception as e:
        raise MyException(e, sys)


@lru_cache
def get_store() -> InMemoryStore:
    try:
        logger.debug("Initializing InMemoryStore singleton")
        store_obj = InMemoryStore()
        logger.info("InMemoryStore initialized")
        return store_obj
    except Exception as e:
        raise MyException(e, sys)


checkpointer = get_checkpointer()
store = get_store()
