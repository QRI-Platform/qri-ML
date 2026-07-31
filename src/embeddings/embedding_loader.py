import sys
from functools import lru_cache
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.core.constants import EMBEDDING_MODEL_NAME
from src.core.logger import logger
from src.core.exceptions import MyException


@lru_cache
def get_embeddings() -> HuggingFaceEmbeddings:
    try:
        logger.debug("Initializing HuggingFaceEmbeddings singleton (may download model)")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        logger.info("HuggingFaceEmbeddings loaded with model: %s", EMBEDDING_MODEL_NAME)
        return embeddings
    except Exception as e:
        raise MyException(e, sys)
