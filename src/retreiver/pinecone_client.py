import sys
from functools import lru_cache
from pinecone import Pinecone
from src.config.app_config import get_app_config
from src.logger import logger
from src.exception import MyException


@lru_cache
def get_pinecone_client() -> Pinecone:
    try:
        logger.debug("Initializing Pinecone client singleton")
        cfg = get_app_config()
        pc = Pinecone(api_key=cfg.pine_cone_api_key)
        logger.info("Pinecone client initialized")
        return pc
    except Exception as e:
        raise MyException(e, sys)
