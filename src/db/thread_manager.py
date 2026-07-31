import os
import sys
import sqlite3
from typing import List, Optional
from functools import lru_cache

from src.core.constants import SQLITE_DB_PATH, MAX_THREADS_PER_USER
from src.core.logger import logger
from src.core.exceptions import MyException
from src.core.memory import get_checkpointer


class ThreadManager:
    def __init__(self, db_path: str = SQLITE_DB_PATH):
        try:
            self.db_path = db_path
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            self._init_db()
            logger.info("ThreadManager initialized with db_path=%s", db_path)
        except Exception as e:
            raise MyException(e, sys)

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        try:
            with self._get_conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_threads (
                        id          INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id     TEXT    NOT NULL,
                        thread_id   TEXT    NOT NULL UNIQUE,
                        created_at  INTEGER NOT NULL DEFAULT (strftime('%s','now'))
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_user_threads_user ON user_threads(user_id)")
                conn.commit()
            logger.info("ThreadManager: DB schema ready at %s", self.db_path)
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def delete_graph_checkpoint(thread_id: str):
        try:
            logger.info("Deleting graph checkpointer memory for thread=%s", thread_id)
            get_checkpointer().delete_thread(thread_id)
            logger.info("Graph checkpointer memory deleted for thread=%s", thread_id)
        except Exception as e:
            logger.error("Failed to delete graph checkpoint for thread=%s: %s", thread_id, str(e))

    def register_thread(self, user_id: str, thread_id: str) -> Optional[str]:
        try:
            logger.info("Registering thread=%s for user=%s", thread_id, user_id)
            evicted_thread_id: Optional[str] = None

            with self._get_conn() as conn:
                existing = conn.execute(
                    "SELECT thread_id FROM user_threads WHERE thread_id = ?", (thread_id,)
                ).fetchone()

                if existing:
                    logger.info("thread=%s already registered for user=%s", thread_id, user_id)
                    return None

                threads = conn.execute(
                    "SELECT thread_id FROM user_threads WHERE user_id = ? ORDER BY created_at ASC",
                    (user_id,),
                ).fetchall()

                if len(threads) >= MAX_THREADS_PER_USER:
                    oldest = threads[0]["thread_id"]
                    conn.execute("DELETE FROM user_threads WHERE thread_id = ?", (oldest,))
                    evicted_thread_id = oldest
                    logger.info(
                        "user=%s exceeded limit=%d — evicting oldest thread=%s",
                        user_id, MAX_THREADS_PER_USER, oldest,
                    )

                conn.execute(
                    "INSERT INTO user_threads (user_id, thread_id) VALUES (?, ?)",
                    (user_id, thread_id),
                )
                conn.commit()

            if evicted_thread_id:
                self.delete_graph_checkpoint(evicted_thread_id)
                logger.info(f"deleted thread from graph {oldest}")

            logger.info("thread=%s registered for user=%s (evicted=%s)", thread_id, user_id, evicted_thread_id)
            return evicted_thread_id
        except Exception as e:
            logger.error("register_thread failed for user=%s thread=%s", user_id, thread_id)
            raise MyException(e, sys)

    def get_threads(self, user_id: str) -> List[str]:
        try:
            logger.debug("Getting threads for user=%s", user_id)
            with self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT thread_id FROM user_threads WHERE user_id = ? ORDER BY created_at ASC",
                    (user_id,),
                ).fetchall()
            thread_ids = [row["thread_id"] for row in rows]
            logger.debug("Found %d threads for user=%s", len(thread_ids), user_id)
            return thread_ids
        except Exception as e:
            logger.error("get_threads failed for user=%s", user_id)
            raise MyException(e, sys)

    def remove_thread(self, thread_id: str):
        try:
            logger.info("Removing thread=%s from DB", thread_id)
            with self._get_conn() as conn:
                conn.execute("DELETE FROM user_threads WHERE thread_id = ?", (thread_id,))
                conn.commit()
            logger.info("thread=%s removed from DB", thread_id)
            self.delete_graph_checkpoint(thread_id)
        except Exception as e:
            logger.error("remove_thread failed for thread=%s", thread_id)
            raise MyException(e, sys)

    def thread_exists(self, thread_id: str) -> bool:
        try:
            logger.debug("Checking existence of thread=%s", thread_id)
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT 1 FROM user_threads WHERE thread_id = ?", (thread_id,)
                ).fetchone()
            exists = row is not None
            logger.debug("thread=%s exists=%s", thread_id, exists)
            return exists
        except Exception as e:
            logger.error("thread_exists failed for thread=%s", thread_id)
            raise MyException(e, sys)


@lru_cache
def get_thread_manager() -> ThreadManager:
    try:
        logger.debug("Initializing ThreadManager singleton")
        tm = ThreadManager()
        logger.info("ThreadManager singleton initialized")
        return tm
    except Exception as e:
        raise MyException(e, sys)
