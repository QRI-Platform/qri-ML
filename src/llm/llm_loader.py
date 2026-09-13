import sys
from functools import lru_cache
from typing import Optional
from langchain_groq import ChatGroq
from src.core.config import get_app_config
from src.core.constants import LLM_MODEL_NAME
from src.core.exceptions import MyException
from src.core.logger import logger


@lru_cache
def get_llm(
    reasoning_format: Optional[str] = "parsed",  # "parsed" or "hidden" for tool-calling nodes
    reasoning_effort: Optional[str] = "medium",  # "low", "medium", "high"
    streaming: bool = False,
) -> ChatGroq:
    try:
        logger.debug(
            "Initializing ChatGroq LLM singleton with reasoning_format=%s, reasoning_effort=%s",
            reasoning_format,
            reasoning_effort,
        )
        cfg = get_app_config()

        kwargs = {
            "model": LLM_MODEL_NAME,
            "api_key": cfg.groq_api_key,
            "streaming": streaming,
        }

        if reasoning_format is not None:
            kwargs["reasoning_format"] = reasoning_format

        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort

        llm = ChatGroq(**kwargs)

        logger.info(
            "ChatGroq LLM initialized with model: %s (reasoning_format=%s, reasoning_effort=%s)",
            LLM_MODEL_NAME,
            reasoning_format,
            reasoning_effort,
        )
        return llm

    except Exception as e:
        raise MyException(e, sys)