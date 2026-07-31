import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
from src.core.constants import LOGS_DIR

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
MAX_FOLDER_SIZE = 2 * 1024 * 1024
MAX_LOG_SIZE = 5 * 1024 * 1024

os.makedirs(LOGS_DIR, exist_ok=True)
log_file_path = os.path.join(LOGS_DIR, LOG_FILE)


def cleanup_logs():
    if not os.path.exists(LOGS_DIR):
        return
    files = [os.path.join(LOGS_DIR, f) for f in os.listdir(LOGS_DIR) if f.endswith(".log")]
    files.sort(key=os.path.getmtime)
    total_size = sum(os.path.getsize(f) for f in files)
    while total_size > MAX_FOLDER_SIZE and files:
        oldest_file = files.pop(0)
        file_size = os.path.getsize(oldest_file)
        try:
            os.remove(oldest_file)
            total_size -= file_size
        except Exception:
            break


def configure_logger():
    cleanup_logs()
    _logger = logging.getLogger("app")
    _logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")

    file_handler = RotatingFileHandler(log_file_path, maxBytes=MAX_LOG_SIZE, backupCount=3)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    _logger.handlers.clear()
    _logger.addHandler(file_handler)
    _logger.addHandler(console_handler)
    return _logger


logger = configure_logger()
logger.info("Logger initialized. Logging to %s", log_file_path)
