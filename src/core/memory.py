import sys
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from langgraph.store.base.batch import AsyncBatchedBaseStore

if not hasattr(AsyncBatchedBaseStore, "_task"):
    AsyncBatchedBaseStore._task = None

from src.core.logger import logger
from src.core.exceptions import MyException
from src.core.config import get_app_config
from src.core.constants import MAXIMUM_CONNECTION_POOL_SIZE

# Neon DB Connection String
DB_URI = get_app_config().postgres_sql_url

# Async Connection Pool
pool = AsyncConnectionPool(
    conninfo=DB_URI, 
    max_size=MAXIMUM_CONNECTION_POOL_SIZE, 
    kwargs={"autocommit": True},
    open=False
)

# Global variables (will be initialized inside init_db_services when event loop starts)
_checkpointer = None
_store = None


def get_checkpointer() -> BaseCheckpointSaver:
    """Returns the initialized async checkpointer instance."""
    if _checkpointer is None:
        raise RuntimeError("Database services are not initialized yet! Ensure 'init_db_services()' ran during startup.")
    return _checkpointer


def get_store() -> BaseStore:
    """Returns the initialized async store instance."""
    if _store is None:
        raise RuntimeError("Database services are not initialized yet! Ensure 'init_db_services()' ran during startup.")
    return _store


async def init_db_services():
    """
    Call this inside FastAPI's lifespan on application startup 
    when the async event loop is active.
    """
    global _checkpointer, _store
    try:
        logger.debug("Initializing Async Connection Pool, Checkpointer, and Store...")
        
        # 1. Open Connection Pool
        await pool.open()
        
        # 2. Instantiate checkpointer and store inside active event loop
        _checkpointer = AsyncPostgresSaver(pool)
        _store = AsyncPostgresStore(pool)
        
        # 3. Setup Postgres tables
        await _checkpointer.setup()
        await _store.setup()
        
        logger.info("Async PostgresSaver and Store initialized successfully")
    except Exception as e:
        raise MyException(e, sys)


async def close_db_services():
    """
    Call this inside FastAPI's lifespan on application shutdown.
    """
    try:
        logger.debug("Closing Async Connection Pool...")
        await pool.close()
        logger.info("Async Connection Pool closed successfully")
    except Exception as e:
        raise MyException(e, sys)