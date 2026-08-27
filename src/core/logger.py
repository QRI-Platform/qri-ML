import os
import sys
import atexit
import logging
from queue import Queue
from datetime import datetime
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
from src.core.constants import LOGS_DIR

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3

os.makedirs(LOGS_DIR, exist_ok=True)
log_file_path = os.path.join(LOGS_DIR, LOG_FILE)


def configure_non_blocking_logger():
    # 1. Main Logger instance
    _logger = logging.getLogger("app")
    _logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")

    # 2. Worker Handlers (Inko direct logger me nahi jodna, ye background thread chalayega)
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # 3. Lock-free in-memory Queue
    log_queue = Queue(-1)

    # 4. Logger par SIRF QueueHandler lagega (Main thread sirf RAM me object push karega)
    queue_handler = QueueHandler(log_queue)
    _logger.handlers.clear()
    _logger.addHandler(queue_handler)

    # 5. Dedicated Background Worker Thread jo Terminal & File dono handle karega
    listener = QueueListener(
        log_queue,
        file_handler,
        console_handler,
        respect_handler_level=True
    )
    listener.start()

    # Server shutdown par queue properly flush hokar close ho
    atexit.register(listener.stop)

    return _logger


logger = configure_non_blocking_logger()