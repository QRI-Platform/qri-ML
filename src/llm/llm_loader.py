import sys
from functools import lru_cache
from langchain_groq import ChatGroq
from src.core.config import get_app_config
from src.core.constants import LLM_MODEL_NAME
from src.core.logger import logger
from src.core.exceptions import MyException


@lru_cache
def get_llm() -> ChatGroq:
    try:
        logger.debug("Initializing ChatGroq LLM singleton")
        cfg = get_app_config()
        llm = ChatGroq(model=LLM_MODEL_NAME, api_key=cfg.groq_api_key)
        logger.info("ChatGroq LLM initialized with model: %s", LLM_MODEL_NAME)
        return llm
    except Exception as e:
        raise MyException(e, sys)