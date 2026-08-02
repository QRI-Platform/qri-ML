import os
import sys
from functools import lru_cache
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from src.core.logger import logger
from src.core.exceptions import MyException
from src.core.config import get_app_config
from src.core.constants import MAXIMUM_CONNECTION_POOL_SIZE

# Neon DB Connection String
DB_URI = get_app_config().postgres_sql_url

# Connection Pool for DB Efficiency
pool = ConnectionPool(conninfo=DB_URI, max_size=MAXIMUM_CONNECTION_POOL_SIZE, kwargs={"autocommit": True})


@lru_cache
def get_checkpointer() -> BaseCheckpointSaver:
    try:
        logger.debug("Initializing PostgresSaver checkpointer singleton")
        # Setup pool connection
        checkpointer_obj = PostgresSaver(pool)
        # Table Creation (Must be run once initially)
        checkpointer_obj.setup()
        logger.info("PostgresSaver checkpointer initialized successfully")
        return checkpointer_obj
    except Exception as e:
        raise MyException(e, sys)


@lru_cache
def get_store() -> BaseStore:
    try:
        logger.debug("Initializing PostgresStore singleton")
        store_obj = PostgresStore(pool)
        store_obj.setup()
        logger.info("PostgresStore initialized successfully")
        return store_obj
    except Exception as e:
        raise MyException(e, sys)


checkpointer = get_checkpointer()
store = get_store()